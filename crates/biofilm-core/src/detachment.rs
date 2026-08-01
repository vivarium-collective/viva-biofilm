use crate::agent::Agent;

pub fn detach_above_height(agents: &mut Vec<Agent>, max_height: f64) -> usize {
    let before = agents.len();
    agents.retain(|a| a.y <= max_height);
    before - agents.len()
}

/// Bucket index for `x` along the cyclic X axis, mirroring
/// `relaxation::cell_index_x`'s convention: canonicalize into `[0, domain_x)`
/// via `rem_euclid` FIRST (an agent's `x` may transiently sit outside that
/// range, e.g. a freshly-divided daughter before the end-of-step wrap runs),
/// THEN floor-divide by `bin_width` and wrap the raw bin index into
/// `[0, n_bins)`. Skipping the pre-canonicalization would let an
/// off-canonical `x` land in a bin that differs from its canonical
/// counterpart's whenever `domain_x` isn't an exact multiple of `bin_width`.
fn bin_index_x(x: f64, bin_width: f64, n_bins: usize, domain_x: f64) -> usize {
    let canon = x.rem_euclid(domain_x);
    let raw = (canon / bin_width).floor() as i64;
    raw.rem_euclid(n_bins as i64) as usize
}

/// Wanner-Gujer height-proportional surface erosion, applied per column.
///
/// The biofilm-liquid interface recedes at a rate proportional to the
/// SQUARE of local thickness (the standard continuum erosion closure:
/// `u_det = k_det * h_local^2`), reflecting that shear stress at a taller
/// column's surface grows faster than shear at a shorter one. Discretely,
/// each call recedes the top of every column by `Delta = k_det * h_max^2 *
/// dt`, where `h_max` is that column's current surface height (max `y`
/// among its agents); an agent detaches once the receding front has
/// undercut its ENTIRE body (see the radius note below), not merely
/// dipped below its center point.
///
/// Deterministic and pure: agents are binned into `n_bins =
/// floor(domain_x / bin_width).max(1)` columns by `x.rem_euclid(domain_x)`
/// (the same cyclic-X convention `relaxation::relax` uses), each column's
/// `h_max`/`y_min` are reduced via a single ordered pass over `agents`, and
/// per-column deltas are a plain `Vec<f64>` indexed by bin — no
/// `HashMap`, no RNG, so which agents survive depends only on positions.
///
/// `bin_width` is the grid's `dx` (µm) at the call site in `World::step`:
/// this keeps erosion decisions aligned with the same columns the
/// reaction-diffusion solve and `cell_of` already use, rather than
/// introducing a second, independent spatial discretization. `density` is
/// the representative packing density used to convert an agent's `mass`
/// into its radius (`Agent::radius`) — the same single representative
/// density `World::step` already uses for `relaxation::relax` (per-species
/// shove mechanics are out of scope here too).
///
/// Radius-undercut rule (why NOT a bare `y <= h_max - Delta` point cut):
/// with point agents, `y <= h_max - Delta` for the SAME `h_max` that is by
/// definition one agent's exact `y` is violated by that very agent for
/// ANY `Delta > 0`, no matter how small — i.e. a naive point-threshold cut
/// unconditionally evicts the current tallest agent every single call
/// regardless of `k_det`'s magnitude (verified empirically: identical
/// collapse-to-monolayer behavior for `k_det` spanning 8 orders of
/// magnitude). Real biofilm surface recession over one `dt` is typically a
/// small fraction of a single agent's own diameter, so that point-cut
/// artifact reads as a fixed "evict exactly the tip every tick" erosion
/// rate independent of `k_det`, defeating the "rate-based, calibratable"
/// requirement. Requiring recession to clear an agent's own radius (its
/// LOWER edge, `a.y - a.radius(density)`, past the new front `h_max -
/// Delta`) before that agent detaches fixes this: an agent survives while
/// still physically anchored (partially) above the front, and only
/// several small `Delta`s accumulating (via `h_max` growing between calls)
/// eventually erode it away. This makes the erosion rate genuinely
/// `k_det`-dependent (see `tests/detachment.rs`'s plateau test and
/// `.superpowers/sdd/substrate-limitation/task2-erosion-report.md` for the
/// calibration this enables).
///
/// Clamp: an agent at/tied-with its column's `y_min` (the bottom-most
/// height present) is never removed, however large `Delta` is — the
/// substratum-adjacent monolayer of every non-empty column always survives
/// one call. Empty columns are simply skipped.
///
/// Returns the number of agents removed. `k_det <= 0.0` is a strict no-op
/// (returns 0 immediately, vector unchanged), so the erosion knob defaults
/// off and cannot perturb any existing behavior unless a spec opts in.
pub fn erode_surface(
    agents: &mut Vec<Agent>,
    k_det: f64,
    dt: f64,
    bin_width: f64,
    domain_x: f64,
    density: f64,
) -> usize {
    if k_det <= 0.0 {
        return 0;
    }
    let before = agents.len();
    if before == 0 {
        return 0;
    }

    let n_bins = ((domain_x / bin_width).floor() as usize).max(1);
    let mut h_max = vec![f64::NEG_INFINITY; n_bins];
    let mut y_min = vec![f64::INFINITY; n_bins];
    for a in agents.iter() {
        let b = bin_index_x(a.x, bin_width, n_bins, domain_x);
        if a.y > h_max[b] {
            h_max[b] = a.y;
        }
        if a.y < y_min[b] {
            y_min[b] = a.y;
        }
    }

    let mut delta = vec![0.0; n_bins];
    for b in 0..n_bins {
        if h_max[b] == f64::NEG_INFINITY {
            continue; // empty column: nothing to erode
        }
        delta[b] = (k_det * h_max[b] * h_max[b] * dt).max(0.0);
    }

    agents.retain(|a| {
        let b = bin_index_x(a.x, bin_width, n_bins, domain_x);
        if delta[b] <= 0.0 {
            return true;
        }
        if a.y <= y_min[b] {
            return true; // substratum-monolayer safety clamp
        }
        let front = h_max[b] - delta[b];
        let lower_edge = a.y - a.radius(density);
        lower_edge <= front
    });

    before - agents.len()
}
