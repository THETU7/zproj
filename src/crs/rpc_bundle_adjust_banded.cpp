// Ceres-based banded bundle adjustment: one global affine per scene plus
// per-band translation shifts (see rpc_bundle_adjust.hpp for the model,
// the gauge handling and the observability caveats).
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
// tent-weighted) term; the zero-mean-derived last band of a scene expands
// to -sum(others). The blocks land in the camera group of the Schur
// ordering whatever they combine, so DENSE_SCHUR keeps eliminating only
// the 3-parameter ground blocks.
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

// Per-scene band table: band indices sorted by row center, with the
// centers in parallel (the tent basis and the nearest-band fallback walk
// these).
struct SceneBands {
    std::vector<int> by_center;
    std::vector<double> centers;
};

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
    for (int k = 0; k < 6; ++k) {
        eff.p[static_cast<std::size_t>(k)] =
            lay.aff_const[static_cast<std::size_t>(k)];
    }
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

}  // namespace

RpcBaReport solve_rpc_bundle_adjust_banded(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaBand>& bands,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffine>& scene_affines,
    std::vector<std::array<double, 2>>& band_shifts,
    const RpcBaBandedOptions& options) {
    RpcBaReport report;
    const int num_scenes = static_cast<int>(scenes.size());
    const int num_bands = static_cast<int>(bands.size());
    if (num_scenes == 0) {
        report.message = "no scenes";
        return report;
    }
    if (num_bands == 0) {
        report.message = "no bands";
        return report;
    }
    if (scene_affines.size() != scenes.size()) {
        report.message = "scene_affines must have one entry per scene (got " +
                         std::to_string(scene_affines.size()) + ", expected " +
                         std::to_string(scenes.size()) + ")";
        return report;
    }
    if (band_shifts.size() != bands.size()) {
        report.message = "band_shifts must have one entry per band (got " +
                         std::to_string(band_shifts.size()) + ", expected " +
                         std::to_string(bands.size()) + ")";
        return report;
    }

    // ---- Band tables and the derived-block structure ----------------------
    // Per scene, the band listed LAST in input order derives its shift from
    // the others (the exact zero-mean-per-scene constraint); under
    // zero_mean_affines the last SCENE's affine is derived from the others.
    std::vector<SceneBands> scene_bands(static_cast<std::size_t>(num_scenes));
    for (int b = 0; b < num_bands; ++b) {
        const RpcBaBand& band = bands[static_cast<std::size_t>(b)];
        if (band.scene < 0 || band.scene >= num_scenes) {
            report.message = "band " + std::to_string(b) +
                             " references scene " + std::to_string(band.scene) +
                             " out of range [0, " + std::to_string(num_scenes) +
                             ")";
            return report;
        }
        if (!(band.row_lo < band.row_hi)) {
            report.message = "band " + std::to_string(b) +
                             " has an empty row range [" +
                             std::to_string(band.row_lo) + ", " +
                             std::to_string(band.row_hi) + ")";
            return report;
        }
        scene_bands[static_cast<std::size_t>(band.scene)].by_center.push_back(
            b);
    }
    for (SceneBands& sb : scene_bands) {
        if (sb.by_center.empty()) {
            report.message = "every scene needs at least one band";
            return report;
        }
        std::sort(
            sb.by_center.begin(), sb.by_center.end(), [&bands](int a, int c) {
                const RpcBaBand& ba = bands[static_cast<std::size_t>(a)];
                const RpcBaBand& bc = bands[static_cast<std::size_t>(c)];
                return (ba.row_lo + ba.row_hi) < (bc.row_lo + bc.row_hi);
            });
        sb.centers.reserve(sb.by_center.size());
        for (int b : sb.by_center) {
            const RpcBaBand& band = bands[static_cast<std::size_t>(b)];
            sb.centers.push_back(0.5 * (band.row_lo + band.row_hi));
        }
    }
    std::vector<int> derived_band(static_cast<std::size_t>(num_scenes), -1);
    std::vector<std::vector<int>> free_bands(
        static_cast<std::size_t>(num_scenes));
    {
        std::vector<int> last(static_cast<std::size_t>(num_scenes), -1);
        for (int b = 0; b < num_bands; ++b) {
            last[static_cast<std::size_t>(
                bands[static_cast<std::size_t>(b)].scene)] = b;
        }
        for (int b = 0; b < num_bands; ++b) {
            const int s = bands[static_cast<std::size_t>(b)].scene;
            if (last[static_cast<std::size_t>(s)] == b) {
                derived_band[static_cast<std::size_t>(s)] = b;
            } else {
                free_bands[static_cast<std::size_t>(s)].push_back(b);
            }
        }
    }
    const bool zero_mean = options.zero_mean_affines;
    const int derived_scene = zero_mean ? num_scenes - 1 : -1;

    // ---- Parameter-block storage (local, so failures leave callers intact)
    std::vector<std::array<double, 6>> aff_blocks(
        static_cast<std::size_t>(num_scenes));
    for (int s = 0; s < num_scenes; ++s) {
        aff_blocks[static_cast<std::size_t>(s)] =
            scene_affines[static_cast<std::size_t>(s)].p;
    }
    std::vector<std::array<double, 2>> shift_blocks(
        static_cast<std::size_t>(num_bands));
    for (int b = 0; b < num_bands; ++b) {
        shift_blocks[static_cast<std::size_t>(b)] =
            band_shifts[static_cast<std::size_t>(b)];
    }

    // ---- Measure -> band terms, and the per-measure layout ----------------
    // (band index, weight) pairs for one measure row; Constant basis picks
    // the containing (else nearest) band, Linear blends the two bracketing
    // band centers with constant extension outside them.
    const auto BandTerms = [&](int scene, double row) {
        const SceneBands& sb = scene_bands[static_cast<std::size_t>(scene)];
        std::vector<std::pair<int, double>> terms;
        if (options.basis == RpcBaBandBasis::Constant ||
            sb.centers.size() < 2) {
            int pick = sb.by_center.front();
            for (int b : sb.by_center) {
                const RpcBaBand& band = bands[static_cast<std::size_t>(b)];
                if (row >= band.row_lo && row < band.row_hi) {
                    pick = b;
                    break;
                }
            }
            if (!(row >= bands[static_cast<std::size_t>(pick)].row_lo &&
                  row < bands[static_cast<std::size_t>(pick)].row_hi)) {
                // Outside every band: nearest center.
                double best = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i < sb.by_center.size(); ++i) {
                    const double d = std::fabs(sb.centers[i] - row);
                    if (d < best) {
                        best = d;
                        pick = sb.by_center[i];
                    }
                }
            }
            terms.emplace_back(pick, 1.0);
            return terms;
        }
        // Linear (tent) basis: bracket the row by band centers.
        const std::size_t n = sb.centers.size();
        std::size_t j = 0;
        while (j < n && sb.centers[j] < row) {
            ++j;
        }
        if (j == 0 || j == n) {
            // Constant extension outside the center span.
            terms.emplace_back(
                j == 0 ? sb.by_center.front() : sb.by_center.back(), 1.0);
            return terms;
        }
        const std::size_t i = j - 1;
        const double span = sb.centers[j] - sb.centers[i];
        if (!(span > 0.0)) {
            terms.emplace_back(sb.by_center[j], 1.0);
            return terms;
        }
        const double w_i = (sb.centers[j] - row) / span;
        terms.emplace_back(sb.by_center[i], w_i);
        terms.emplace_back(sb.by_center[j], 1.0 - w_i);
        return terms;
    };

    const auto MakeLayout = [&](int scene, double row) {
        MeasureLayout lay;
        if (scene != derived_scene) {
            lay.aff_terms.emplace_back(
                aff_blocks[static_cast<std::size_t>(scene)].data(), kOnes6);
        } else {
            for (int k = 0; k < 6; ++k) {
                lay.aff_const[static_cast<std::size_t>(k)] =
                    static_cast<double>(num_scenes) *
                    kAffineIdentity[static_cast<std::size_t>(k)];
            }
            for (int s = 0; s + 1 < num_scenes; ++s) {
                lay.aff_terms.emplace_back(
                    aff_blocks[static_cast<std::size_t>(s)].data(), kNegOnes6);
            }
        }
        // Shift terms with derived-band expansion (a derived band
        // contributes -weight to each free band of its scene); accumulate
        // per block so tent brackets never repeat a pointer.
        for (const auto& [band, weight] : BandTerms(scene, row)) {
            const int s = bands[static_cast<std::size_t>(band)].scene;
            if (band != derived_band[static_cast<std::size_t>(s)]) {
                const double* src =
                    shift_blocks[static_cast<std::size_t>(band)].data();
                const auto it = std::find_if(
                    lay.shift_terms.begin(),
                    lay.shift_terms.end(),
                    [src](const auto& t) { return t.first == src; });
                if (it != lay.shift_terms.end()) {
                    it->second[0] += weight;
                    it->second[1] += weight;
                } else {
                    lay.shift_terms.emplace_back(
                        shift_blocks[static_cast<std::size_t>(band)].data(),
                        std::array<double, 2>{weight, weight});
                }
                continue;
            }
            for (int f : free_bands[static_cast<std::size_t>(s)]) {
                double* src = shift_blocks[static_cast<std::size_t>(f)].data();
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
    // Measures reference scenes here; a non-finite row cannot be banded.
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
    // shift (each scene's derived band adds none).
    const int n_free_params = (zero_mean ? num_scenes - 1 : num_scenes) *
                                  static_cast<int>(options.dof) +
                              2 * (num_bands - num_scenes);
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
        report.message =
            "under-determined: " + std::to_string(ties.size()) +
            " tie points + " + std::to_string(gcps.size()) + " GCPs net only " +
            std::to_string(net) + " constraints and cannot float " +
            std::to_string(n_free_params) + " free parameters (" +
            std::to_string((zero_mean ? num_scenes - 1 : num_scenes) *
                           static_cast<int>(options.dof)) +
            " affine + " + std::to_string(2 * (num_bands - num_scenes)) +
            " band shift)";
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
        double* block = aff_blocks[static_cast<std::size_t>(s)].data();
        problem.AddParameterBlock(block, 6);
        if (auto manifold = MakeAffineManifold(options.dof)) {
            problem.SetManifold(block, manifold.release());
        }
        aff_ptrs.push_back(block);
    }
    std::vector<double*> shift_ptrs;
    for (int s = 0; s < num_scenes; ++s) {
        for (int f : free_bands[static_cast<std::size_t>(s)]) {
            double* block = shift_blocks[static_cast<std::size_t>(f)].data();
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

    // No affine write-back through SolveAndReport (the banded blocks are
    // written manually below); ties present -> DENSE_SCHUR eliminates the
    // ground blocks, else dense QR.
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

    // Write back scene affines and band shifts, including the derived ones.
    // Only on success -- a failed solve must leave the callers' values
    // untouched.
    if (solved.ok) {
        for (int s = 0; s < num_scenes; ++s) {
            if (s == derived_scene) {
                continue;
            }
            scene_affines[static_cast<std::size_t>(s)].p =
                aff_blocks[static_cast<std::size_t>(s)];
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
                        aff_blocks[static_cast<std::size_t>(s)]
                                  [static_cast<std::size_t>(k)];
                }
            }
            scene_affines[static_cast<std::size_t>(num_scenes - 1)].p = derived;
        }
        for (int s = 0; s < num_scenes; ++s) {
            for (int f : free_bands[static_cast<std::size_t>(s)]) {
                band_shifts[static_cast<std::size_t>(f)] =
                    shift_blocks[static_cast<std::size_t>(f)];
            }
            const int d = derived_band[static_cast<std::size_t>(s)];
            if (d >= 0) {
                std::array<double, 2> derived{0.0, 0.0};
                for (int f : free_bands[static_cast<std::size_t>(s)]) {
                    derived[0] -= shift_blocks[static_cast<std::size_t>(f)][0];
                    derived[1] -= shift_blocks[static_cast<std::size_t>(f)][1];
                }
                band_shifts[static_cast<std::size_t>(d)] = derived;
            }
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
