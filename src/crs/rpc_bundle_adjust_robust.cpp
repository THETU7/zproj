// The robust multi-pass driver over the two-stage gridded solve (see
// rpc_bundle_adjust.hpp's "robust variant" notes): graduated per-pass
// loss settings, (scene, cell) median/MAD measure trimming between
// passes, warm-started throughout. This file holds only the ladder and
// the trim -- every solve is solve_rpc_bundle_adjust_two_stage, every
// residual comes back through the grid stage's out-record, so the
// driver owns no solver internals.
//
// The trim groups measures by (scene, cell) and compares each residual
// to its group's MEDIAN, not to zero: a cell with a genuine large local
// offset sits centered on that offset (its good measures stay), while an
// outlier deviates from the group regardless of the offset -- the
// per-cell-offset inconsistency that defeats absolute pixel thresholds.
// The scale is the group's own MAD (1.4826 * MAD ~ sigma for Gaussian
// noise), floored at trim_floor_px so a clean, near-zero-MAD group trims
// nothing, and groups of 1-2 measures keep everything (no scale to
// estimate).
//
// Candidates are then applied PER POINT, worst-first but one per pass:
// a bad measure contaminates its whole point's ground block (the
// triangulation is plain least squares), misfitting the point's CLEAN
// measures too, so an independent per-measure drop would cull those
// clean measures alongside the outlier. Dropping only the point's worst
// candidate keeps the clean majority, and the next pass's fresh
// triangulation heals the spared measures. A point that cannot spare
// any candidate (a 2-measure tie point with one bad measure has no
// majority to heal from) goes entirely.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "zproj/crs/rpc_bundle_adjust.hpp"

namespace zproj::crs {
namespace {

// Median of a double vector (copies; trim groups are small).
double MedianOf(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// One trim candidate: the residual record's index and its deviation
// from the group median, in units of the trim radius (> 1 = outside).
struct TrimCandidate {
    std::size_t residual;
    double excess;
};

// Collect the trim candidates: measures whose residual deviates from
// their (scene, cell) group's median by more than k robust sigmas
// (1.4826 * MAD, floored at floor_px), worst first. Groups of 1-2
// measures keep everything (no scale to estimate); scenes without cells
// group under cell -1 -- still a meaningful per-scene group.
std::vector<TrimCandidate> TrimCandidates(
    const std::vector<RpcBaMeasureResidual>& residuals,
    double k,
    double floor_px) {
    std::map<std::pair<int, int>, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        const RpcBaMeasureResidual& r = residuals[i];
        groups[{r.scene, r.cell}].push_back(i);
    }

    std::vector<TrimCandidate> candidates;
    for (const auto& [key, idxs] : groups) {
        (void)key;
        if (idxs.size() < 3) {
            continue;
        }
        double med[2] = {0.0, 0.0};
        double radius[2] = {0.0, 0.0};
        for (int axis = 0; axis < 2; ++axis) {
            std::vector<double> vals;
            vals.reserve(idxs.size());
            for (std::size_t i : idxs) {
                const RpcBaMeasureResidual& r = residuals[i];
                vals.push_back(axis == 0 ? r.res_col : r.res_row);
            }
            med[axis] = MedianOf(vals);
            for (double& v : vals) {
                v = std::fabs(v - med[axis]);
            }
            radius[axis] = std::max(k * 1.4826 * MedianOf(vals), floor_px);
        }
        for (std::size_t i : idxs) {
            const RpcBaMeasureResidual& r = residuals[i];
            const double excess =
                std::max(std::fabs(r.res_col - med[0]) / radius[0],
                         std::fabs(r.res_row - med[1]) / radius[1]);
            if (excess > 1.0) {
                candidates.push_back(TrimCandidate{i, excess});
            }
        }
    }
    std::sort(candidates.begin(),
              candidates.end(),
              [](const TrimCandidate& a, const TrimCandidate& b) {
                  return a.excess > b.excess;
              });
    return candidates;
}

// Apply the candidates to the network (in place), per point: drop each
// point's worst candidate while the point keeps its measure floor (2
// ties / 1 GCP), sparing the rest to heal next pass; a point that
// cannot spare any candidate goes entirely. Returns the measure and
// point counts removed.
std::pair<int, int> ApplyTrim(
    std::vector<RpcBaPoint>& points,
    const std::vector<RpcBaMeasureResidual>& residuals,
    const std::vector<TrimCandidate>& candidates) {
    std::vector<char> drop_measure(residuals.size(), 0);
    std::vector<char> drop_point(points.size(), 0);
    // Candidates are globally worst-first, so each point's first entry
    // is its worst measure.
    std::map<int, std::vector<std::size_t>> by_point;
    for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
        by_point[residuals[candidates[ci].residual].point].push_back(ci);
    }

    int trimmed = 0;
    int dropped_points = 0;
    for (const auto& [pi, cand_idx] : by_point) {
        if (pi < 0 || pi >= static_cast<int>(points.size())) {
            continue;
        }
        RpcBaPoint& pt = points[static_cast<std::size_t>(pi)];
        const std::size_t need = pt.ground_fixed ? 1 : 2;
        // A point already at its measure floor cannot spare ANY
        // candidate: a 2-measure tie point with one bad measure has no
        // majority to heal from, so the whole point goes -- but only on
        // a GROSS flag (a real mismatch sits many radii out); a marginal
        // one is more likely a noise tail or leftover contamination,
        // which the next pass heals or tolerates.
        if (pt.measures.size() <= need) {
            if (candidates[cand_idx[0]].excess > 3.0) {
                drop_point[static_cast<std::size_t>(pi)] = 1;
                ++dropped_points;
            }
            continue;
        }
        // Drop the worst candidates while the floor holds; any remaining
        // candidates are spared this pass (they are usually the clean
        // measures misfit by the dropped outlier's dragged ground block,
        // and the next pass's fresh triangulation heals them).
        std::size_t kept = pt.measures.size();
        for (std::size_t k = 0; k < cand_idx.size() && kept > need; ++k) {
            drop_measure[candidates[cand_idx[k]].residual] = 1;
            ++trimmed;
            --kept;
        }
    }

    // Rebuild: drop the marked measures, then the starved points.
    std::vector<RpcBaPoint> next;
    next.reserve(points.size());
    for (std::size_t pi = 0; pi < points.size(); ++pi) {
        if (drop_point[pi]) {
            continue;
        }
        RpcBaPoint pt = points[pi];
        // One residual record per used measure; map the surviving
        // measures to their original indices through the residuals.
        std::vector<char> drop(pt.measures.size(), 0);
        for (std::size_t i = 0; i < residuals.size(); ++i) {
            if (drop_measure[i] && residuals[i].point == static_cast<int>(pi)) {
                if (residuals[i].measure >= 0 &&
                    residuals[i].measure < static_cast<int>(drop.size())) {
                    drop[static_cast<std::size_t>(residuals[i].measure)] = 1;
                }
            }
        }
        std::vector<RpcBaMeasure> keep;
        keep.reserve(pt.measures.size());
        for (std::size_t mi = 0; mi < pt.measures.size(); ++mi) {
            if (!drop[mi]) {
                keep.push_back(pt.measures[mi]);
            }
        }
        pt.measures = std::move(keep);
        next.push_back(std::move(pt));
    }
    points = std::move(next);
    return {trimmed, dropped_points};
}

}  // namespace

RpcBaRobustReport solve_rpc_bundle_adjust_robust(
    const std::vector<RpcInfo>& scenes,
    const std::vector<RpcBaPoint>& points,
    std::vector<RpcAffineGridded>& corrected,
    const RpcBaRobustOptions& options) {
    RpcBaRobustReport report;
    if (scenes.empty()) {
        report.message = "no scenes";
        return report;
    }
    if (corrected.size() != scenes.size()) {
        report.message = "corrected must have one entry per scene (got " +
                         std::to_string(corrected.size()) + ", expected " +
                         std::to_string(scenes.size()) + ")";
        return report;
    }

    // The working ladder: an empty `passes` means one default pass (the
    // base loss settings, no trim) -- the pass knobs then never override.
    std::vector<RpcBaRobustPass> ladder = options.passes;
    const bool tuned_passes = !ladder.empty();
    if (ladder.empty()) {
        ladder.emplace_back();
    }

    std::vector<RpcBaPoint> work = points;
    std::vector<RpcAffineGridded> best = corrected;
    for (std::size_t p = 0; p < ladder.size(); ++p) {
        const RpcBaRobustPass& pass = ladder[p];
        RpcBaGridOptions stage_options = options;
        if (tuned_passes) {
            stage_options.affine_robust_threshold_px =
                pass.affine_robust_threshold_px;
            stage_options.robust_threshold_px = pass.grid_robust_threshold_px;
        }

        std::vector<RpcBaMeasureResidual> residuals;
        RpcBaTwoStageReport stage = solve_rpc_bundle_adjust_two_stage(
            scenes, work, best, stage_options, &residuals);
        report.passes.push_back(stage);
        if (!stage.affine_stage.ok || !stage.grid_stage.ok) {
            // A pass that cannot solve must not contribute: `best`
            // (hence `corrected`) stays at the last successful pass.
            report.message =
                "pass " + std::to_string(p) + " failed: " +
                (stage.affine_stage.ok ? stage.grid_stage.message
                                       : stage.affine_stage.message);
            return report;
        }

        const bool last = (p + 1 == ladder.size());
        if (last || pass.trim_mad_k <= 0.0 || residuals.empty()) {
            break;
        }

        const std::vector<TrimCandidate> candidates =
            TrimCandidates(residuals, pass.trim_mad_k, options.trim_floor_px);
        if (candidates.empty()) {
            continue;  // nothing to shrink; the ladder marches on
        }
        const auto [trimmed, dropped] = ApplyTrim(work, residuals, candidates);
        report.num_measures_trimmed += trimmed;
        report.num_points_dropped += dropped;
    }

    corrected = std::move(best);
    report.ok = true;
    return report;
}

}  // namespace zproj::crs
