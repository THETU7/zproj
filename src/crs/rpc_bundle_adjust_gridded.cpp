// Ceres-based gridded bundle adjustment: one global affine per scene plus
// per-cell translation shifts over a 2D (col, row) grid, and the two-stage
// driver that stages the decomposition (see rpc_bundle_adjust.hpp for the
// models, the gauge handling and the observability caveats).
//
// The corrections travel as RpcAffineGridded objects (one per scene, in
// and out); internally the solve unpacks them into per-scene SceneParams
// (the affine block, the grid structure the object came with, and the
// free + derived shift blocks in canonical cell order) and writes the
// solved values back through the objects' mutators on success.
//
// The parameter structure is expressed with ONE generic dynamic-autodiff
// cost: every quantity the residual evaluates is an affine (linear)
// combination of parameter blocks --
//
//     effective affine = aff_const + sum(coef * scene affine block)
//     effective shift  = sum(coef * cell shift block)
//
// which covers all the cases uniformly: a free scene's affine is a single
// coefficient-1 term; the zero-mean-derived last scene expands to
// S*identity - sum(others); a free cell's shift is a coefficient-1 (or
// bilinear-weighted) term; the zero-mean-derived canonically-last cell of
// a scene expands to -sum(others). The blocks land in the camera group of
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
using refine_detail::IdentityPriorPx;
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
// the measure's effective (affine + cell shift) correction. Parameter
// blocks arrive as [aff_terms..., shift_terms..., ground]; residuals in
// units of pixel_sigma. Forward RPC evaluation only, so autodiff is exact.
// Dynamic autodiff because the block list varies (free vs derived scene,
// one-hot vs bilinear cells); the jet count stays ~(6 + 8 + 3), the same
// order as the plain solver's static (6, 3) costs.
struct GriddedReprojError {
    GriddedReprojError(const RpcInfo* info,
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

// Tikhonov prior on one cell shift block: residuals = w * shift (pulling
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

// First-difference prior between two adjacent cell shift blocks:
// residuals = w * (a - b), pulling the shift field of neighboring cells
// towards each other (the smooth-field regularizer behind
// cell_shift_smoothness_weight).
struct ShiftSmoothness {
    explicit ShiftSmoothness(double weight) : weight_(weight) {}

    template<typename T>
    bool operator()(const T* const a, const T* const b, T* residuals) const {
        residuals[0] = T(weight_) * (a[0] - b[0]);
        residuals[1] = T(weight_) * (a[1] - b[1]);
        return true;
    }

    double weight_;
};

// Per-scene solve storage: everything one scene contributes, in CANONICAL
// cell order (the caller's object stores the regular row-major grid, so
// local cell count-1 is the canonically last -- derived -- cell). The
// containers are sized once and only their elements mutate afterwards, so
// the parameter blocks taken from them stay address-stable through the
// solve.
struct SceneParams {
    std::array<double, 6> aff;  // the scene affine block
    RpcAffineGridBasis basis = RpcAffineGridBasis::Linear;
    std::vector<std::array<double, 2>> shift;      // [i]: local cell i's block
    std::vector<std::array<double, 2>> col_range;  // [i]: (col_lo, col_hi)
    std::vector<std::array<double, 2>> row_range;  // [i]: (row_lo, row_hi)
    std::vector<double> col_center;                // [i]: cell col center
    std::vector<double> row_center;                // [i]: cell row center
    int num_cols = 1;  // cells per grid row (the tensor layout)
};

}  // namespace

RpcBaReport solve_rpc_bundle_adjust_gridded(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaGridOptions& options,
    std::vector<RpcBaMeasureResidual>* residuals) {
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
        const RpcAffineGridded& obj = corrected[static_cast<std::size_t>(s)];
        SceneParams& sp = scene_params[static_cast<std::size_t>(s)];
        sp.aff = obj.affine().p;
        sp.basis = obj.basis();
        sp.num_cols = obj.num_cols();
        sp.shift.resize(static_cast<std::size_t>(obj.num_cells()));
        sp.col_range.resize(static_cast<std::size_t>(obj.num_cells()));
        sp.row_range.resize(static_cast<std::size_t>(obj.num_cells()));
        sp.col_center.resize(static_cast<std::size_t>(obj.num_cells()));
        sp.row_center.resize(static_cast<std::size_t>(obj.num_cells()));
        for (int i = 0; i < obj.num_cells(); ++i) {
            const auto& c = obj.cell(i);
            sp.shift[static_cast<std::size_t>(i)] = {c.dx, c.dy};
            sp.col_range[static_cast<std::size_t>(i)] = {c.col_lo, c.col_hi};
            sp.row_range[static_cast<std::size_t>(i)] = {c.row_lo, c.row_hi};
            sp.col_center[static_cast<std::size_t>(i)] =
                0.5 * (c.col_lo + c.col_hi);
            sp.row_center[static_cast<std::size_t>(i)] =
                0.5 * (c.row_lo + c.row_hi);
        }
    }
    // fix_scene_affines (stage 2) holds every affine constant; the
    // zero-mean derivation is then moot and ignored.
    const bool zero_mean =
        options.zero_mean_affines && !options.fix_scene_affines;
    const int derived_scene = zero_mean ? num_scenes - 1 : -1;

    // ---- Measure -> cell terms, and the per-measure layout ----------------
    // The CONTAINING cell of one measure pixel (else the nearest center),
    // -1 for a scene without cells -- the cell a measure "belongs to" for
    // support counting, median init and residual grouping. Constant basis
    // and single-cell scenes resolve through it directly.
    const auto ContainingCell = [&](int scene, double col, double row) {
        const SceneParams& sp = scene_params[static_cast<std::size_t>(scene)];
        const int count = static_cast<int>(sp.shift.size());
        if (count == 0) {
            return -1;
        }
        int pick = -1;
        for (int i = 0; i < count; ++i) {
            const auto& cr = sp.col_range[static_cast<std::size_t>(i)];
            const auto& rr = sp.row_range[static_cast<std::size_t>(i)];
            if (col >= cr[0] && col < cr[1] && row >= rr[0] && row < rr[1]) {
                pick = i;
                break;
            }
        }
        if (pick < 0) {
            // Outside every cell: nearest center (pixel-space 2D
            // distance).
            double best = std::numeric_limits<double>::infinity();
            for (int i = 0; i < count; ++i) {
                const double dc =
                    sp.col_center[static_cast<std::size_t>(i)] - col;
                const double dr =
                    sp.row_center[static_cast<std::size_t>(i)] - row;
                const double d = (dc * dc) + (dr * dr);
                if (d < best) {
                    best = d;
                    pick = i;
                }
            }
        }
        return pick;
    };

    // (local cell index, weight) pairs for one measure pixel; Constant
    // basis picks the containing cell, Linear blends the up to four
    // bilinearly bracketing cells of the regular grid (constant extension
    // outside the center spans, per axis). A scene without cells
    // contributes no terms (plain affine scene).
    const auto CellTerms = [&](int scene, double col, double row) {
        std::vector<std::pair<int, double>> terms;
        const SceneParams& sp = scene_params[static_cast<std::size_t>(scene)];
        const int count = static_cast<int>(sp.shift.size());
        if (count == 0) {
            return terms;
        }
        if (sp.basis == RpcAffineGridBasis::Constant || count < 2) {
            terms.emplace_back(ContainingCell(scene, col, row), 1.0);
            return terms;
        }
        // Linear (tensor tent) basis: bracket each axis of the regular
        // grid by its centers (col centers are cells [0, num_cols); row
        // centers are cells [k * num_cols, ...)), then combine the
        // weights. A clamped axis collapses to its boundary cell.
        const int cols = sp.num_cols;
        const int rows = count / cols;
        int ic = 0;
        int jc = 0;
        double wci = 1.0;
        double wcj = 0.0;
        if (cols > 1) {
            while (jc < cols &&
                   sp.col_center[static_cast<std::size_t>(jc)] < col) {
                ++jc;
            }
            if (jc == 0 || jc == cols) {
                ic = jc = (jc == 0) ? 0 : cols - 1;
                wci = 1.0;
                wcj = 0.0;
            } else {
                ic = jc - 1;
                const double span =
                    sp.col_center[static_cast<std::size_t>(jc)] -
                    sp.col_center[static_cast<std::size_t>(ic)];
                if (!(span > 0.0)) {
                    ic = jc;
                    wci = 1.0;
                    wcj = 0.0;
                } else {
                    wci = (sp.col_center[static_cast<std::size_t>(jc)] - col) /
                          span;
                    wcj = 1.0 - wci;
                }
            }
        }
        int ir = 0;
        int jr = 0;
        double wri = 1.0;
        double wrj = 0.0;
        if (rows > 1) {
            while (jr < rows &&
                   sp.row_center[static_cast<std::size_t>(jr * cols)] < row) {
                ++jr;
            }
            if (jr == 0 || jr == rows) {
                ir = jr = (jr == 0) ? 0 : rows - 1;
                wri = 1.0;
                wrj = 0.0;
            } else {
                ir = jr - 1;
                const double span =
                    sp.row_center[static_cast<std::size_t>(jr * cols)] -
                    sp.row_center[static_cast<std::size_t>(ir * cols)];
                if (!(span > 0.0)) {
                    ir = jr;
                    wri = 1.0;
                    wrj = 0.0;
                } else {
                    wri = (sp.row_center[static_cast<std::size_t>(jr * cols)] -
                           row) /
                          span;
                    wrj = 1.0 - wri;
                }
            }
        }
        for (int a = 0; a < 2; ++a) {
            const int r = (a == 0) ? ir : jr;
            const double wr = (a == 0) ? wri : wrj;
            if (wr == 0.0) {
                continue;
            }
            for (int b = 0; b < 2; ++b) {
                const int c = (b == 0) ? ic : jc;
                const double wc = (b == 0) ? wci : wcj;
                if (wc == 0.0) {
                    continue;
                }
                terms.emplace_back(r * cols + c, wr * wc);
            }
        }
        return terms;
    };

    const auto MakeLayout = [&](int scene, double col, double row) {
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
        // Shift terms with derived-cell expansion (each scene's
        // canonically-last cell derives its shift from the others);
        // accumulate per block so bilinear brackets never repeat a
        // pointer.
        SceneParams& sp = scene_params[static_cast<std::size_t>(scene)];
        const int count = static_cast<int>(sp.shift.size());
        for (const auto& [cell, weight] : CellTerms(scene, col, row)) {
            if (cell + 1 < count) {
                double* src = sp.shift[static_cast<std::size_t>(cell)].data();
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
    // Measures reference scenes; a non-finite pixel cannot be gridded. The
    // kept points carry their ORIGINAL indices (point into `points`,
    // measure into that point's measures) so the residual out-record can
    // address the caller's data; duplicate-view drops desync the measure
    // indices, so they are tracked per kept measure.
    std::vector<RpcBaPoint> ties;
    std::vector<RpcBaPoint> gcps;
    std::vector<int> tie_point_orig;
    std::vector<int> gcp_point_orig;
    std::vector<std::vector<int>> tie_measure_orig;
    std::vector<std::vector<int>> gcp_measure_orig;
    int skipped = 0;
    for (std::size_t pi = 0; pi < points.size(); ++pi) {
        const RpcBaPoint& pt = points[pi];
        if (pt.ground_fixed &&
            (!std::isfinite(pt.lon) || !std::isfinite(pt.lat) ||
             !std::isfinite(pt.height))) {
            ++skipped;
            continue;
        }
        RpcBaPoint clean = pt;
        clean.measures.clear();
        std::vector<int> measure_orig;
        bool bad = false;
        for (std::size_t mi = 0; mi < pt.measures.size(); ++mi) {
            const RpcBaMeasure& m = pt.measures[mi];
            if (m.view < 0 || m.view >= num_scenes || !std::isfinite(m.col) ||
                !std::isfinite(m.row)) {
                bad = true;
                break;
            }
            bool dup = false;
            for (const RpcBaMeasure& k : clean.measures) {
                dup = dup || (k.view == m.view);
            }
            if (!dup) {
                clean.measures.push_back(m);
                measure_orig.push_back(static_cast<int>(mi));
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
        if (pt.ground_fixed) {
            gcps.push_back(std::move(clean));
            gcp_point_orig.push_back(static_cast<int>(pi));
            gcp_measure_orig.push_back(std::move(measure_orig));
        } else {
            ties.push_back(std::move(clean));
            tie_point_orig.push_back(static_cast<int>(pi));
            tie_measure_orig.push_back(std::move(measure_orig));
        }
    }

    // ---- Ground-block initialization for the tie points -------------------
    // median_cell_init discards the incoming shifts first, so the
    // triangulation runs shift-free against the scene affines alone and
    // the medians below measure offsets relative to a clean reference.
    if (options.median_cell_init) {
        for (SceneParams& sp : scene_params) {
            for (auto& s : sp.shift) {
                s = {0.0, 0.0};
            }
        }
    }
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
            const RpcAffine eff =
                EvalEffective(MakeLayout(m.view, m.col, m.row));
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
            tie_point_orig[kept] = tie_point_orig[i];
            tie_measure_orig[kept] = std::move(tie_measure_orig[i]);
        }
        ++kept;
    }
    ties.resize(kept);
    tie_point_orig.resize(kept);
    tie_measure_orig.resize(kept);
    report.num_points = static_cast<int>(ties.size());
    report.num_gcps = static_cast<int>(gcps.size());

    if (ties.empty() && gcps.empty()) {
        report.message =
            "no usable observations (every control point was malformed or "
            "failed to triangulate)";
        report.num_points_skipped = skipped;
        return report;
    }

    // ---- Median cell warm start / support counting / pinning ---------------
    // The scene affines alone (shift-free projection reference; the derived
    // scene's affine expands like MakeLayout's aff part).
    const auto SceneAffineOnly = [&](int scene) {
        RpcAffine eff;
        if (scene != derived_scene) {
            eff.p = scene_params[static_cast<std::size_t>(scene)].aff;
            return eff;
        }
        for (int k = 0; k < 6; ++k) {
            eff.p[static_cast<std::size_t>(k)] =
                static_cast<double>(num_scenes) *
                kAffineIdentity[static_cast<std::size_t>(k)];
        }
        for (int s = 0; s + 1 < num_scenes; ++s) {
            for (int k = 0; k < 6; ++k) {
                eff.p[static_cast<std::size_t>(k)] -=
                    scene_params[static_cast<std::size_t>(s)]
                        .aff[static_cast<std::size_t>(k)];
            }
        }
        return eff;
    };
    // GCP ground blocks are the survey truth, independent of the solve.
    std::vector<std::array<double, 3>> gcp_ground;
    gcp_ground.reserve(gcps.size());
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        gcp_ground.push_back({gcps[i].lon, gcps[i].lat, gcps[i].height});
    }

    // One offset (obs - projected-without-shift, px) per measure, grouped
    // by (scene, containing cell) -- the median warm start's and the
    // support floor's shared pass over the network.
    std::vector<std::vector<std::vector<std::array<double, 2>>>> offsets(
        scene_params.size());
    for (std::size_t s = 0; s < scene_params.size(); ++s) {
        offsets[s].resize(scene_params[s].shift.size());
    }
    const auto CollectOffsets = [&]() {
        const auto AddPoint = [&](const RpcBaPoint& pt,
                                  const std::array<double, 3>& g) {
            for (const RpcBaMeasure& m : pt.measures) {
                const int cell = ContainingCell(m.view, m.col, m.row);
                if (cell < 0) {
                    continue;
                }
                double col = 0.0;
                double row = 0.0;
                rpc_forward_point(scenes[static_cast<std::size_t>(m.view)],
                                  g[0],
                                  g[1],
                                  g[2],
                                  col,
                                  row);
                const RpcAffine aff = SceneAffineOnly(m.view);
                const double pc =
                    aff.p[0] + (aff.p[1] * col) + (aff.p[2] * row);
                const double pr =
                    aff.p[3] + (aff.p[4] * col) + (aff.p[5] * row);
                offsets[static_cast<std::size_t>(m.view)]
                       [static_cast<std::size_t>(cell)]
                           .push_back({m.col - pc, m.row - pr});
            }
        };
        for (std::size_t i = 0; i < ties.size(); ++i) {
            AddPoint(ties[i], ground[i]);
        }
        for (std::size_t i = 0; i < gcps.size(); ++i) {
            AddPoint(gcps[i], gcp_ground[i]);
        }
    };
    CollectOffsets();

    if (options.median_cell_init) {
        for (std::size_t s = 0; s < scene_params.size(); ++s) {
            SceneParams& sp = scene_params[s];
            for (std::size_t f = 0; f + 1 < sp.shift.size(); ++f) {
                std::vector<std::array<double, 2>>& group = offsets[s][f];
                if (group.empty()) {
                    continue;  // no data: keep the (zeroed) warm start
                }
                const auto Median = [](std::vector<double>& v) {
                    std::sort(v.begin(), v.end());
                    const std::size_t n = v.size();
                    return (n % 2 == 1) ? v[n / 2]
                                        : 0.5 * (v[n / 2 - 1] + v[n / 2]);
                };
                std::vector<double> xs;
                std::vector<double> ys;
                xs.reserve(group.size());
                ys.reserve(group.size());
                for (const auto& o : group) {
                    xs.push_back(o[0]);
                    ys.push_back(o[1]);
                }
                sp.shift[f] = {Median(xs), Median(ys)};
            }
        }
    }

    // Support floor: free cells below min_measures_per_cell measures fall
    // back to the scene affine (shift zeroed and held constant through the
    // solve). The derived cell is not a parameter block and never pins.
    std::vector<std::vector<char>> cell_pinned(scene_params.size());
    for (std::size_t s = 0; s < scene_params.size(); ++s) {
        SceneParams& sp = scene_params[s];
        cell_pinned[s].assign(sp.shift.size(), 0);
        if (options.min_measures_per_cell <= 0) {
            continue;
        }
        for (std::size_t f = 0; f + 1 < sp.shift.size(); ++f) {
            if (offsets[s][f].size() <
                static_cast<std::size_t>(options.min_measures_per_cell)) {
                cell_pinned[s][f] = 1;
                sp.shift[f] = {0.0, 0.0};
            }
        }
    }

    // ---- Constraint accounting --------------------------------------------
    // Free parameters: the free scene affines' dof (none under
    // fix_scene_affines -- the affines are constants) plus 2 per free,
    // un-pinned cell shift (each scene's derived cell adds none;
    // affine-only scenes add none).
    int free_shift_params = 0;
    for (std::size_t s = 0; s < scene_params.size(); ++s) {
        for (std::size_t f = 0; f + 1 < scene_params[s].shift.size(); ++f) {
            if (!cell_pinned[s][f]) {
                free_shift_params += 2;
            }
        }
    }
    const int free_affine_params =
        options.fix_scene_affines ? 0
                                  : (zero_mean ? num_scenes - 1 : num_scenes) *
                                        static_cast<int>(options.dof);
    const int n_free_params = free_affine_params + free_shift_params;
    const bool has_prior = options.affine_prior_weight > 0.0 ||
                           options.identity_prior_px > 0.0 ||
                           options.cell_shift_prior_weight > 0.0 ||
                           options.cell_shift_smoothness_weight > 0.0;
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
                         std::to_string(free_shift_params) + " cell shift)";
        report.num_points_skipped = skipped;
        return report;
    }
    const std::vector<std::array<double, 3>> ground_init = ground;

    // ---- Problem assembly ---------------------------------------------------
    ceres::Problem problem;
    std::vector<double*> aff_ptrs;
    for (int s = 0; s < num_scenes; ++s) {
        if (s == derived_scene) {
            // Derived under zero-mean: residuals reference the other
            // scenes' blocks, so this one is never a parameter block.
            continue;
        }
        double* block = scene_params[static_cast<std::size_t>(s)].aff.data();
        problem.AddParameterBlock(block, 6);
        if (options.fix_scene_affines) {
            problem.SetParameterBlockConstant(block);
            continue;
        }
        if (auto manifold = MakeAffineManifold(options.dof)) {
            problem.SetManifold(block, manifold.release());
        }
        aff_ptrs.push_back(block);
    }
    std::vector<double*> shift_ptrs;
    for (std::size_t s = 0; s < scene_params.size(); ++s) {
        SceneParams& sp = scene_params[s];
        for (std::size_t f = 0; f + 1 < sp.shift.size(); ++f) {
            double* block = sp.shift[f].data();
            problem.AddParameterBlock(block, 2);
            if (cell_pinned[s][f]) {
                problem.SetParameterBlockConstant(block);
            } else {
                shift_ptrs.push_back(block);
            }
        }
    }

    auto ordering = std::make_unique<ceres::ParameterBlockOrdering>();

    const auto AddMeasures = [&](const RpcBaPoint& pt, double* ground_block) {
        for (const RpcBaMeasure& m : pt.measures) {
            MeasureLayout lay = MakeLayout(m.view, m.col, m.row);
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
                new ceres::DynamicAutoDiffCostFunction<GriddedReprojError>(
                    new GriddedReprojError(
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
            problem.AddResidualBlock(fn,
                                     NewLoss(options.loss_kind,
                                             options.robust_threshold_px,
                                             options.pixel_sigma),
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
    for (std::size_t i = 0; i < gcps.size(); ++i) {
        double* g = gcp_ground[i].data();
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

    if (!options.fix_scene_affines) {
        for (int s = 0; s < num_scenes; ++s) {
            if (s == derived_scene) {
                continue;
            }
            double* b = scene_params[static_cast<std::size_t>(s)].aff.data();
            if (options.identity_prior_px > 0.0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<IdentityPriorPx, 6, 6>(
                        new IdentityPriorPx(
                            options.identity_prior_px,
                            scenes[static_cast<std::size_t>(s)].samp_scale,
                            scenes[static_cast<std::size_t>(s)].line_scale)),
                    nullptr,
                    b);
            }
            if (options.affine_prior_weight > 0.0) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<AffinePrior, 6, 6>(
                        new AffinePrior(options.affine_prior_weight)),
                    nullptr,
                    b);
            }
        }
    }
    if (options.cell_shift_prior_weight > 0.0) {
        for (double* b : shift_ptrs) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ShiftPrior, 2, 2>(
                    new ShiftPrior(options.cell_shift_prior_weight)),
                nullptr,
                b);
        }
    }
    // Smoothness prior between adjacent FREE cells of the regular tensor
    // grid (Linear basis only: Constant grids carry discontinuous
    // stitching structure that smoothing across seams would erase, and
    // ragged Constant layouts have no meaningful adjacency). Pairs
    // touching the derived (canonically last) cell are omitted: its shift
    // is minus the sum of the free cells, so a spike in any free cell
    // already moves it oppositely.
    if (options.cell_shift_smoothness_weight > 0.0) {
        for (std::size_t s = 0; s < scene_params.size(); ++s) {
            SceneParams& sp = scene_params[s];
            if (sp.basis != RpcAffineGridBasis::Linear || sp.shift.size() < 2) {
                continue;
            }
            const int cols = sp.num_cols;
            const int count = static_cast<int>(sp.shift.size());
            const auto Free = [&](int i) {
                return i + 1 < count &&
                       !cell_pinned[s][static_cast<std::size_t>(i)];
            };
            const auto AddPair = [&](int a, int b) {
                if (!Free(a) || !Free(b)) {
                    return;
                }
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<ShiftSmoothness, 2, 2, 2>(
                        new ShiftSmoothness(
                            options.cell_shift_smoothness_weight)),
                    nullptr,
                    sp.shift[static_cast<std::size_t>(a)].data(),
                    sp.shift[static_cast<std::size_t>(b)].data());
            };
            for (int i = 0; i < count; ++i) {
                if ((i % cols) + 1 < cols) {
                    AddPair(i, i + 1);  // horizontal neighbor
                }
                if (i + cols < count) {
                    AddPair(i, i + cols);  // vertical neighbor
                }
            }
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
                        EvalEffective(MakeLayout(m.view, m.col, m.row));
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
    // leave them untouched. (Under fix_scene_affines the affine blocks
    // are constants, so this writes their incoming values back.)
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
                corrected[static_cast<std::size_t>(s)].set_cell_shift(
                    static_cast<int>(f), sp.shift[f][0], sp.shift[f][1]);
                derived[0] -= sp.shift[f][0];
                derived[1] -= sp.shift[f][1];
            }
            corrected[static_cast<std::size_t>(s)].set_cell_shift(
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

    // Per-measure residuals against the final model (scene_params/ground
    // hold the solved values; gcp_ground is constant), in the caller's
    // point/measure indexing. Only on success -- a failed solve leaves
    // the out-record empty.
    if (residuals != nullptr) {
        residuals->clear();
        if (solved.ok) {
            residuals->reserve(static_cast<std::size_t>(out.num_measures));
            const auto AddPoint = [&](const RpcBaPoint& pt,
                                      int point_orig,
                                      const std::vector<int>& measure_orig,
                                      const std::array<double, 3>& g) {
                for (std::size_t k = 0; k < pt.measures.size(); ++k) {
                    const RpcBaMeasure& m = pt.measures[k];
                    const RpcAffine eff =
                        EvalEffective(MakeLayout(m.view, m.col, m.row));
                    double col = 0.0;
                    double row = 0.0;
                    rpc_forward_point(scenes[static_cast<std::size_t>(m.view)],
                                      g[0],
                                      g[1],
                                      g[2],
                                      col,
                                      row);
                    residuals->push_back(RpcBaMeasureResidual{
                        point_orig,
                        measure_orig[k],
                        m.view,
                        ContainingCell(m.view, m.col, m.row),
                        eff.p[0] + (eff.p[1] * col) + (eff.p[2] * row) - m.col,
                        eff.p[3] + (eff.p[4] * col) + (eff.p[5] * row) -
                            m.row});
                }
            };
            for (std::size_t i = 0; i < ties.size(); ++i) {
                AddPoint(
                    ties[i], tie_point_orig[i], tie_measure_orig[i], ground[i]);
            }
            for (std::size_t i = 0; i < gcps.size(); ++i) {
                AddPoint(gcps[i],
                         gcp_point_orig[i],
                         gcp_measure_orig[i],
                         gcp_ground[i]);
            }
        }
    }
    return out;
}

RpcBaTwoStageReport solve_rpc_bundle_adjust_two_stage(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaGridOptions& options,
    std::vector<RpcBaMeasureResidual>* residuals) {
    RpcBaTwoStageReport out;
    if (scenes.empty()) {
        out.affine_stage.message = "no scenes";
        return out;
    }
    if (corrected.size() != scenes.size()) {
        out.affine_stage.message =
            "corrected must have one entry per scene (got " +
            std::to_string(corrected.size()) + ", expected " +
            std::to_string(scenes.size()) + ")";
        return out;
    }

    // Stage 1: the plain per-scene affine solve over the same network,
    // held near identity by identity_prior_px (the caller sets it to the
    // trusted absolute accuracy of the RPC products -- the staging's
    // whole point, see rpc_bundle_adjust.hpp). The affine_* loss
    // overrides arm a LOOSER stage-1 loss: at this stage the grid's
    // local structure is still unmodeled and shows up as large-but-
    // genuine residuals a tight loss would suppress.
    RpcBaOptions affine_options = options;
    if (options.affine_robust_threshold_px.has_value()) {
        affine_options.robust_threshold_px =
            *options.affine_robust_threshold_px;
    }
    if (options.affine_loss_kind.has_value()) {
        affine_options.loss_kind = *options.affine_loss_kind;
    }
    std::vector<RpcAffine> affines;
    affines.reserve(scenes.size());
    for (const RpcAffineGridded& g : corrected) {
        affines.push_back(g.affine());
    }
    out.affine_stage =
        solve_rpc_bundle_adjust(scenes, points, affines, affine_options);
    if (!out.affine_stage.ok) {
        // corrected untouched: a failed stage 1 leaves the whole staging
        // moot.
        return out;
    }
    for (std::size_t s = 0; s < scenes.size(); ++s) {
        corrected[s].set_affine(affines[s]);
    }

    // Stage 2: affines frozen at stage 1's solution; only the cells'
    // translation shifts float.
    RpcBaGridOptions grid_options = options;
    grid_options.fix_scene_affines = true;
    grid_options.zero_mean_affines = false;
    out.grid_stage = solve_rpc_bundle_adjust_gridded(
        scenes, points, corrected, grid_options, residuals);
    return out;
}

}  // namespace zproj::crs
