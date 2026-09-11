// Ceres-based banded bundle adjustment: one global affine per scene plus
// per-band translation shifts (see rpc_bundle_adjust.hpp for the model,
// the gauge handling and the observability caveats).
//
// The corrections travel as RpcAffineBanded objects (one per scene, in and
// out); internally the solve unpacks them into per-scene SceneParams (the
// affine block, the band structure the object came with, and the free +
// derived shift blocks in local-band order) and writes the solved values
// back through the objects' mutators on success.
//
// The parameter structure is expressed with ONE generic dynamic-autodiff
// cost: every quantity the residual evaluates is an affine (linear)
// combination of parameter blocks --
//
//     effective affine = aff_const + sum(coef * scene affine block)
//     effective shift  = sum(coef * band shift block)
//
// which covers all the cases uniformly: a free scene's affine is a single
// coefficient-1 term; the zero-mean-derived last scene expands to
// S*identity - sum(others); a free band's shift is a coefficient-1 (or
// tent-weighted) term; the zero-mean-derived highest-center band of a
// scene expands to -sum(others). The blocks land in the camera group of
// the Schur ordering whatever they combine, so DENSE_SCHUR keeps
// eliminating only the 3-parameter ground blocks.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "zproj/crs/rpc_bundle_adjust.hpp"
#include "zproj/crs/rpc_ray.hpp"
#include "zproj/crs/wgs84.hpp"

#include "rpc_affine_detail.h"

namespace zproj::crs {
namespace {

using refine_detail::AffinePrior;
using refine_detail::kAffineIdentity;
using refine_detail::MakeAffineManifold;
using refine_detail::NewLoss;
using refine_detail::ResidualSq;

constexpr std::array<double, 6> kOnes6{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
constexpr std::array<double, 6> kNegOnes6{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0};

// A measure's residual as an affine combination of parameter blocks (the
// layout the functor and the double-precision evaluator share):
//   effective affine p = aff_const + sum(aff_terms: coef * 6-block)
//   effective shift  d = sum(shift_terms: coef * 2-block)
// Block pointers are unique within each term list, and the two lists are
// disjoint (6-blocks vs 2-blocks).
struct MeasureLayout {
    std::array<double, 6> aff_const{};
    std::vector<std::pair<double*, std::array<double, 6>>> aff_terms;
    std::vector<std::pair<double*, std::array<double, 2>>> shift_terms;
};

// The effective affine with the shift folded into the translations, as one
// plain RpcAffine (for the ground-block initialization triangulation and
// the before/after RMS loops).
RpcAffine EvalEffective(const MeasureLayout& lay) {
    RpcAffine eff;
    eff.p = lay.aff_const;
    for (const auto& [block, coef] : lay.aff_terms) {
        for (int k = 0; k < 6; ++k) {
            eff.p[static_cast<std::size_t>(k)] +=
                coef[static_cast<std::size_t>(k)] * block[k];
        }
    }
    double dx = 0.0;
    double dy = 0.0;
    for (const auto& [block, coef] : lay.shift_terms) {
        dx += coef[0] * block[0];
        dy += coef[1] * block[1];
    }
    eff.p[0] += dx;
    eff.p[3] += dy;
    return eff;
}

// Reprojection cost: projects one ground block through one scene's RPC and
// the measure's effective (affine + band shift) correction. Parameter
// blocks arrive as [aff_terms..., shift_terms..., ground]; residuals in
// units of pixel_sigma. Forward RPC evaluation only, so autodiff is exact.
// Dynamic autodiff because the block list varies (free vs derived scene,
// one-hot vs tent bands); the jet count stays ~(6 + 4 + 3), the same order
// as the plain solver's static (6, 3) costs.
struct BandedReprojError {
    BandedReprojError(const RpcInfo* info,
                      double obs_col,
                      double obs_row,
                      double pixel_sigma,
                      MeasureLayout layout)
        : info_(info),
          obs_col_(obs_col),
          obs_row_(obs_row),
          layout_(std::move(layout)) {
        inv_sigma_ = 1.0 / ((pixel_sigma > 0.0) ? pixel_sigma : 1.0);
    }

    template<typename T>
    bool operator()(T const* const* params, T* residuals) const {
        T p0 = T(layout_.aff_const[0]);
        T p1 = T(layout_.aff_const[1]);
        T p2 = T(layout_.aff_const[2]);
        T p3 = T(layout_.aff_const[3]);
        T p4 = T(layout_.aff_const[4]);
        T p5 = T(layout_.aff_const[5]);
        std::size_t idx = 0;
        for (const auto& term : layout_.aff_terms) {
            const auto& coef = term.second;
            p0 += T(coef[0]) * params[idx][0];
            p1 += T(coef[1]) * params[idx][1];
            p2 += T(coef[2]) * params[idx][2];
            p3 += T(coef[3]) * params[idx][3];
            p4 += T(coef[4]) * params[idx][4];
            p5 += T(coef[5]) * params[idx][5];
            ++idx;
        }
        T d0 = T(0.0);
        T d1 = T(0.0);
        for (const auto& term : layout_.shift_terms) {
            d0 += T(term.second[0]) * params[idx][0];
            d1 += T(term.second[1]) * params[idx][1];
            ++idx;
        }
        const T* lonlath = params[idx];
        T col;
        T row;
        detail::rpc_forward_point_core(
            *info_, lonlath[0], lonlath[1], lonlath[2], col, row);
        residuals[0] =
            (p0 + (p1 * col) + (p2 * row) + d0 - T(obs_col_)) * T(inv_sigma_);
        residuals[1] =
            (p3 + (p4 * col) + (p5 * row) + d1 - T(obs_row_)) * T(inv_sigma_);
        return true;
    }

    const RpcInfo* info_;
    double obs_col_;
    double obs_row_;
    double inv_sigma_;
    MeasureLayout layout_;
};

// Tikhonov prior on one band shift block: residuals = w * shift (pulling
// towards "the scene affine alone").
struct ShiftPrior {
    explicit ShiftPrior(double weight) : weight_(weight) {}

    template<typename T>
    bool operator()(const T* const shift, T* residuals) const {
        residuals[0] = T(weight_) * shift[0];
        residuals[1] = T(weight_) * shift[1];
        return true;
    }

    double weight_;
};

// Per-scene solve storage: everything one scene contributes, in LOCAL band
// order (the caller's object stores bands sorted by ascending center, so
// local band count-1 is the highest-center -- derived -- band). The
// containers are sized once and only their elements mutate afterwards, so
// the parameter blocks taken from them stay address-stable through the
// solve.
struct SceneParams {
    std::array<double, 6> aff;  // the scene affine block
    RpcAffineBandBasis basis = RpcAffineBandBasis::Linear;
    std::vector<std::array<double, 2>> shift;  // [i]: local band i's block
    std::vector<std::array<double, 2>> range;  // [i]: (row_lo, row_hi)
    std::vector<double> center;                // [i]: band center, ascending
};

}  // namespace

RpcBaReport solve_rpc_bundle_adjust_banded(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineBanded>& corrected,
    const RpcBaBandedOptions& options) {
    RpcBaReport report;
    const int num_scenes = static_cast<int>(scenes.size());
    if (num_scenes == 0) {
        report.message = "no scenes";
        return report;
    }
    if (corrected.size() != scenes.size()) {
        report.message = "corrected must have one entry per scene (got " +
                         std::to_string(corrected.size()) + ", expected " +
                         std::to_string(scenes.size()) + ")";
        return report;
    }

    // ---- Unpack the scene corrections into solve storage -----------------
    std::vector<SceneParams> scene_params(static_cast<std::size_t>(num_scenes));
    for (int s = 0; s < num_scenes; ++s) {
        const RpcAffineBanded& obj = corrected[static_cast<std::size_t>(s)];
        SceneParams& sp = scene_params[static_cast<std::size_t>(s)];
        sp.aff = obj.affine().p;
        sp.basis = obj.basis();
        sp.shift.resize(static_cast<std::size_t>(obj.num_bands()));
        sp.range.resize(static_cast<std::size_t>(obj.num_bands()));
        sp.center.resize(static_cast<std::size_t>(obj.num_bands()));
        for (int i = 0; i < obj.num_bands(); ++i) {
            sp.shift[static_cast<std::size_t>(i)] = {obj.band(i).dx,
                                                     obj.band(i).dy};
            sp.range[static_cast<std::size_t>(i)] = {obj.band(i).row_lo,
                                                     obj.band(i).row_hi};
            sp.center[static_cast<std::size_t>(i)] =
                0.5 * (obj.band(i).row_lo + obj.band(i).row_hi);
        }
    }
    const bool zero_mean = options.zero_mean_affines;
    const int derived_scene = zero_mean ? num_scenes - 1 : -1;

    // ---- Measure -> band terms, and the per-measure layout ----------------
    // (local band index, weight) pairs for one measure row; Constant basis
    // picks the containing (else nearest) band, Linear blends the two
    // bracketing band centers with constant extension outside them. A
    // scene without bands contributes no terms (plain affine scene).
    const auto BandTerms = [&](int scene, double row) {
        std::vector<std::pair<int, double>> terms;
        const SceneParams& sp = scene_params[static_cast<std::size_t>(scene)];
        const int count = static_cast<int>(sp.shift.size());
        if (count == 0) {
            return terms;
        }
        if (sp.basis == RpcAffineBandBasis::Constant || count < 2) {
            int pick = 0;
            for (int i = 0; i < count; ++i) {
                const auto& r = sp.range[static_cast<std::size_t>(i)];
                if (row >= r[0] && row < r[1]) {
                    pick = i;
                    break;
                }
            }
            const auto& r = sp.range[static_cast<std::size_t>(pick)];
            if (!(row >= r[0] && row < r[1])) {
                // Outside every band: nearest center.
                double best = std::numeric_limits<double>::infinity();
                for (int i = 0; i < count; ++i) {
                    const double d =
                        std::fabs(sp.center[static_cast<std::size_t>(i)] - row);
                    if (d < best) {
                        best = d;
                        pick = i;
                    }
                }
            }
            terms.emplace_back(pick, 1.0);
            return terms;
        }
        // Linear (tent) basis: bracket the row by band centers.
        int j = 0;
        while (j < count && sp.center[static_cast<std::size_t>(j)] < row) {
            ++j;
        }
        if (j == 0 || j == count) {
            // Constant extension outside the center span.
            terms.emplace_back(j == 0 ? 0 : count - 1, 1.0);
            return terms;
        }
        const int i = j - 1;
        const double span = sp.center[static_cast<std::size_t>(j)] -
                            sp.center[static_cast<std::size_t>(i)];
        if (!(span > 0.0)) {
            terms.emplace_back(j, 1.0);
            return terms;
        }
        const double w_i =
            (sp.center[static_cast<std::size_t>(j)] - row) / span;
        terms.emplace_back(i, w_i);
        terms.emplace_back(j, 1.0 - w_i);
        return terms;
    };

    const auto MakeLayout = [&](int scene, double row) {
        MeasureLayout lay;
        if (scene != derived_scene) {
            lay.aff_terms.emplace_back(
                scene_params[static_cast<std::size_t>(scene)].aff.data(),
                kOnes6);
        } else {
            for (int k = 0; k < 6; ++k) {
                lay.aff_const[static_cast<std::size_t>(k)] =
                    static_cast<double>(num_scenes) *
                    kAffineIdentity[static_cast<std::size_t>(k)];
            }
            for (int s = 0; s + 1 < num_scenes; ++s) {
                lay.aff_terms.emplace_back(
                    scene_params[static_cast<std::size_t>(s)].aff.data(),
                    kNegOnes6);
            }
        }
        // Shift terms with derived-band expansion (each scene's
        // highest-center band derives its shift from the others);
        // accumulate per block so tent brackets never repeat a pointer.
        SceneParams& sp = scene_params[static_cast<std::size_t>(scene)];
        const int count = static_cast<int>(sp.shift.size());
        for (const auto& [band, weight] : BandTerms(scene, row)) {
            if (band + 1 < count) {
                double* src = sp.shift[static_cast<std::size_t>(band)].data();
                const auto it = std::find_if(
                    lay.shift_terms.begin(),
                    lay.shift_terms.end(),
                    [src](const auto& t) { return t.first == src; });
                if (it != lay.shift_terms.end()) {
                    it->second[0] += weight;
                    it->second[1] += weight;
                } else {
                    lay.shift_terms.emplace_back(
                        src, std::array<double, 2>{weight, weight});
                }
                continue;
            }
            for (int f = 0; f + 1 < count; ++f) {
                double* src = sp.shift[static_cast<std::size_t>(f)].data();
                const auto it = std::find_if(
                    lay.shift_terms.begin(),
                    lay.shift_terms.end(),
                    [src](const auto& t) { return t.first == src; });
                if (it != lay.shift_terms.end()) {
                    it->second[0] -= weight;
                    it->second[1] -= weight;
                } else {
                    lay.shift_terms.emplace_back(
                        src, std::array<double, 2>{-weight, -weight});
                }
            }
        }
        return lay;
    };

    // ---- Sanitize the control network -------------------------------------
    // Measures reference scenes; a non-finite row cannot be banded.
    std::vector<RpcBaPoint> ties;
    std::vector<RpcBaPoint> gcps;
    int skipped = 0;
    for (const RpcBaPoint& pt : points) {
        if (pt.ground_fixed &&
            (!std::isfinite(pt.lon) || !std::isfinite(pt.lat) ||
             !std::isfinite(pt.height))) {
            ++skipped;
            continue;
        }
        RpcBaPoint clean = pt;
        clean.measures.clear();
        bool bad = false;
        for (const RpcBaMeasure& m : pt.measures) {
            if (m.view < 0 || m.view >= num_scenes || !std::isfinite(m.row)) {
                bad = true;
                break;
            }
            bool dup = false;
            for (const RpcBaMeasure& k : clean.measures) {
                dup = dup || (k.view == m.view);
            }
            if (!dup) {
                clean.measures.push_back(m);
            }
        }
        if (bad) {
            ++skipped;
            continue;
        }
        const std::size_t need = pt.ground_fixed ? 1 : 2;
        if (clean.measures.size() < need) {
            ++skipped;
            continue;
        }
        (pt.ground_fixed ? gcps : ties).push_back(std::move(clean));
    }

    // ---- Ground-block initialization for the tie points -------------------
    std::vector<RpcModel> models;
    models.reserve(scenes.size());
    for (const RpcInfo& s : scenes) {
        models.emplace_back(s);
    }
    std::vector<std::array<double, 3>> ground_all(ties.size());
    std::vector<unsigned char> init_ok(ties.size(), 0);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(ties.size());
         ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const RpcBaPoint& pt = ties[idx];
        std::vector<RpcRay> rays;
        rays.reserve(pt.measures.size());
        for (const RpcBaMeasure& m : pt.measures) {
            const RpcInfo& v = scenes[static_cast<std::size_t>(m.view)];
            const double span = std::min(0.9 * v.height_scale, 50.0);
            const RpcAffine eff = EvalEffective(MakeLayout(m.view, m.row));
            RpcRay ray;
            if (rpc_ray_affine(
                    v,
                    models[static_cast<std::size_t>(m.view)].inverse_init(),
                    eff,
                    m.col,
                    m.row,
                    v.height_off - span,
                    v.height_off + span,
                    ray)) {
                rays.push_back(ray);
            }
        }
        Ecef p;
        double rms = 0.0;
        std::array<double, 3> lonlath{};
        bool ok = rays.size() >= 2 &&
                  triangulate_nview(
                      rays.data(), static_cast<int>(rays.size()), p, rms) &&
                  std::isfinite(p.x()) && std::isfinite(p.y()) &&
                  std::isfinite(p.z());
        if (ok) {
            const Geodetic g = from_ecef(p);
            lonlath = {g.x() * kRadToDeg, g.y() * kRadToDeg, g.z()};
        }
        ground_all[idx] = lonlath;
        init_ok[idx] = ok ? 1 : 0;
    }
    std::vector<std::array<double, 3>> ground;
    ground.reserve(ties.size());
    std::size_t kept = 0;
    for (std::size_t i = 0; i < ties.size(); ++i) {
        if (!init_ok[i]) {
            ++skipped;
            continue;
        }
        if (std::isfinite(ties[i].height)) {
            ground_all[i][2] = ties[i].height;
        }
        ground.push_back(ground_all[i]);
        if (kept != i) {
            ties[kept] = std::move(ties[i]);
        }
        ++kept;
    }
    ties.resize(kept);
    report.num_points = static_cast<int>(ties.size());
    report.num_gcps = static_cast<int>(gcps.size());

    if (ties.empty() && gcps.empty()) {
        report.message =
            "no usable observations (every control point was malformed or "
            "failed to triangulate)";
        report.num_points_skipped = skipped;
        return report;
    }

    // ---- Constraint accounting --------------------------------------------
    // Free parameters: the free scene affines' dof plus 2 per free band
    // shift (each scene's derived band adds none; affine-only scenes add
    // none).
    int free_shift_params = 0;
    for (const SceneParams& sp : scene_params) {
        if (sp.shift.size() > 1) {
            free_shift_params += 2 * static_cast<int>(sp.shift.size() - 1);
        }
    }
    const int free_affine_params = (zero_mean ? num_scenes - 1 : num_scenes) *
                                   static_cast<int>(options.dof);
    const int n_free_params = free_affine_params + free_shift_params;
    const bool has_prior = options.affine_prior_weight > 0.0 ||
                           options.band_shift_prior_weight > 0.0;
    double net = 0.0;
    for (const RpcBaPoint& t : ties) {
        net += 2.0 * static_cast<double>(t.measures.size()) -
               (std::isfinite(t.height) ? 2.0 : 3.0);
    }
    for (const RpcBaPoint& g : gcps) {
        net += 2.0 * static_cast<double>(g.measures.size());
    }
    if (!has_prior && net < static_cast<double>(n_free_params)) {
        report.message = "under-determined: " + std::to_string(ties.size()) +
                         " tie points + " + std::to_string(gcps.size()) +
                         " GCPs net only " + std::to_string(net) +
                         " constraints and cannot float " +
                         std::to_string(n_free_params) + " free parameters (" +
                         std::to_string(free_affine_params) + " affine + " +
                         std::to_string(free_shift_params) + " band shift)";
        report.num_points_skipped = skipped;
        return report;
    }
    const std::vector<std::array<double, 3>> ground_init = ground;

    // ---- Problem assembly ---------------------------------------------------
    ceres::Problem problem;
    std::vector<double*> aff_ptrs;
    for (int s = 0; s < num_scenes; ++s) {
        if (s == derived_scene) {
            continue;
        }
        double* block = scene_params[static_cast<std::size_t>(s)].aff.data();
        problem.AddParameterBlock(block, 6);
        if (auto manifold = MakeAffineManifold(options.dof)) {
            problem.SetManifold(block, manifold.release());
        }
        aff_ptrs.push_back(block);
    }
    std::vector<double*> shift_ptrs;
    for (SceneParams& sp : scene_params) {
        for (std::size_t f = 0; f + 1 < sp.shift.size(); ++f) {
            double* block = sp.shift[f].data();
            problem.AddParameterBlock(block, 2);
            shift_ptrs.push_back(block);
        }
    }

    auto ordering = std::make_unique<ceres::ParameterBlockOrdering>();

    const auto AddMeasures = [&](const RpcBaPoint& pt, double* ground_block) {
        for (const RpcBaMeasure& m : pt.measures) {
            MeasureLayout lay = MakeLayout(m.view, m.row);
            std::vector<double*> blocks;
            blocks.reserve(lay.aff_terms.size() + lay.shift_terms.size() + 1);
            for (const auto& term : lay.aff_terms) {
                blocks.push_back(term.first);
            }
            for (const auto& term : lay.shift_terms) {
                blocks.push_back(term.first);
            }
            blocks.push_back(ground_block);
            // Capture the block sizes before the layout is moved from.
            const std::size_t num_aff = lay.aff_terms.size();
            const std::size_t num_shift = lay.shift_terms.size();
            auto* fn =
                new ceres::DynamicAutoDiffCostFunction<BandedReprojError>(
                    new BandedReprojError(
                        &scenes[static_cast<std::size_t>(m.view)],
                        m.col,
                        m.row,
                        options.pixel_sigma,
                        std::move(lay)));
            fn->SetNumResiduals(2);
            for (std::size_t i = 0; i < num_aff; ++i) {
                fn->AddParameterBlock(6);
            }
            for (std::size_t i = 0; i < num_shift; ++i) {
                fn->AddParameterBlock(2);
            }
            fn->AddParameterBlock(3);
            problem.AddResidualBlock(
                fn,
                NewLoss(options.robust_threshold_px, options.pixel_sigma),
                blocks);
        }
    };

    for (std::size_t i = 0; i < ties.size(); ++i) {
        double* g = ground[i].data();
        problem.AddParameterBlock(g, 3);
        if (std::isfinite(ties[i].height)) {
            problem.SetManifold(g, new ceres::SubsetManifold(3, {2}));
        }
        ordering->AddElementToGroup(g, 0);
        AddMeasures(ties[i], g);
    }
    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(gcps.size());
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        gcp_ground.push_back({gcps[i].lon, gcps[i].lat, gcps[i].height});
        double* g = gcp_ground.back().data();
        problem.AddParameterBlock(g, 3);
        problem.SetParameterBlockConstant(g);
        AddMeasures(gcps[i], g);
    }
    for (double* b : aff_ptrs) {
        ordering->AddElementToGroup(b, 1);
    }
    for (double* b : shift_ptrs) {
        ordering->AddElementToGroup(b, 1);
    }

    if (options.affine_prior_weight > 0.0) {
        for (double* b : aff_ptrs) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                    new AffinePrior(options.affine_prior_weight)),
                nullptr,
                b);
        }
    }
    if (options.band_shift_prior_weight > 0.0) {
        for (double* b : shift_ptrs) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShiftPrior, 2, 2>(
                    new ShiftPrior(options.band_shift_prior_weight)),
                nullptr,
                b);
        }
    }

    // ---- RMS before / solve / write-back -----------------------------------
    const auto ReprojectionRms =
        [&](const std::vector<std::array<double, 3>>& tie_ground) {
            double sum = 0.0;
            std::size_t n_res = 0;
            const auto Accum = [&](const RpcBaPoint& pt,
                                   const std::array<double, 3>& g) {
                for (const RpcBaMeasure& m : pt.measures) {
                    const RpcAffine eff =
                        EvalEffective(MakeLayout(m.view, m.row));
                    sum += ResidualSq(scenes[static_cast<std::size_t>(m.view)],
                                      eff.p.data(),
                                      g.data(),
                                      m.col,
                                      m.row);
                    n_res += 2;
                }
            };
            for (std::size_t i = 0; i < ties.size(); ++i) {
                Accum(ties[i], tie_ground[i]);
            }
            for (std::size_t i = 0; i < gcps.size(); ++i) {
                Accum(gcps[i], gcp_ground[i]);
            }
            return (n_res > 0) ? std::sqrt(sum / static_cast<double>(n_res))
                               : 0.0;
        };

    const double rms_before = ReprojectionRms(ground_init);

    // No affine write-back through SolveAndReport (the objects are
    // updated through their mutators below); ties present -> DENSE_SCHUR
    // eliminates the ground blocks, else dense QR.
    const ceres::LinearSolverType solver_type =
        ties.empty() ? ceres::DENSE_QR : ceres::DENSE_SCHUR;
    std::unique_ptr<ceres::ParameterBlockOrdering> solve_ordering;
    if (!ties.empty()) {
        solve_ordering = std::move(ordering);
    }
    RpcAffineReport solved =
        refine_detail::SolveAndReport(problem,
                                      options.max_iterations,
                                      options.num_threads,
                                      options.verbose,
                                      solver_type,
                                      {},
                                      std::move(solve_ordering));

    // Write the solved values back into the caller's objects, including
    // the derived quantities. Only on success -- a failed solve must
    // leave them untouched.
    if (solved.ok) {
        for (int s = 0; s < num_scenes; ++s) {
            if (s == derived_scene) {
                continue;
            }
            RpcAffine aff;
            aff.p = scene_params[static_cast<std::size_t>(s)].aff;
            corrected[static_cast<std::size_t>(s)].set_affine(aff);
        }
        if (zero_mean) {
            std::array<double, 6> derived{};
            for (int k = 0; k < 6; ++k) {
                derived[static_cast<std::size_t>(k)] =
                    static_cast<double>(num_scenes) *
                    kAffineIdentity[static_cast<std::size_t>(k)];
            }
            for (int s = 0; s + 1 < num_scenes; ++s) {
                for (int k = 0; k < 6; ++k) {
                    derived[static_cast<std::size_t>(k)] -=
                        scene_params[static_cast<std::size_t>(s)]
                            .aff[static_cast<std::size_t>(k)];
                }
            }
            RpcAffine aff;
            aff.p = derived;
            corrected[static_cast<std::size_t>(num_scenes - 1)].set_affine(aff);
        }
        for (int s = 0; s < num_scenes; ++s) {
            const SceneParams& sp = scene_params[static_cast<std::size_t>(s)];
            if (sp.shift.empty()) {
                continue;
            }
            std::array<double, 2> derived{0.0, 0.0};
            for (std::size_t f = 0; f + 1 < sp.shift.size(); ++f) {
                corrected[static_cast<std::size_t>(s)].set_band_shift(
                    static_cast<int>(f), sp.shift[f][0], sp.shift[f][1]);
                derived[0] -= sp.shift[f][0];
                derived[1] -= sp.shift[f][1];
            }
            corrected[static_cast<std::size_t>(s)].set_band_shift(
                static_cast<int>(sp.shift.size() - 1), derived[0], derived[1]);
        }
    }

    RpcBaReport out;
    out.ok = solved.ok;
    out.num_points = report.num_points;
    out.num_points_skipped = skipped;
    out.num_gcps = report.num_gcps;
    out.num_measures = 0;
    for (const RpcBaPoint& t : ties) {
        out.num_measures += static_cast<int>(t.measures.size());
    }
    for (const RpcBaPoint& g : gcps) {
        out.num_measures += static_cast<int>(g.measures.size());
    }
    out.rms_before_px = rms_before;
    out.rms_after_px = ReprojectionRms(ground);
    out.message = std::move(solved.message);
    return out;
}

}  // namespace zproj::crs
