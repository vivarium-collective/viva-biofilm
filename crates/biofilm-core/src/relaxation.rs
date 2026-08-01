use crate::agent::Agent;

fn dx_cyclic(a: f64, b: f64, span: f64) -> f64 {
    let mut d = a - b;
    if d > span / 2.0 {
        d -= span;
    } else if d < -span / 2.0 {
        d += span;
    }
    d
}

/// Apply the same overlap-push used in the O(n^2) version to a candidate
/// pair (i, j), mutating agents in place. Caller guarantees i != j.
fn push_pair(agents: &mut [Agent], i: usize, j: usize, density: f64, domain_x: f64, k: f64) {
    let ri = agents[i].radius(density);
    let rj = agents[j].radius(density);
    let min_d = ri + rj;
    let dx = dx_cyclic(agents[i].x, agents[j].x, domain_x);
    let dy = agents[i].y - agents[j].y;
    let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
    let overlap = min_d - dist;
    if overlap > 0.0 {
        let push = 0.5 * k * overlap;
        let ux = dx / dist;
        let uy = dy / dist;
        agents[i].x += push * ux;
        agents[i].y += push * uy;
        agents[j].x -= push * ux;
        agents[j].y -= push * uy;
    }
}

/// Uniform spatial-hash grid over agent indices, rebuilt once per
/// iteration. The domain is cyclic in X (wrapped via `rem_euclid` on the
/// cell index) and open in Y. Cell size is chosen so that any pair of
/// agents that can possibly overlap lies in the same cell or one of its 8
/// neighbors, making the neighbor search O(1) per agent at roughly-uniform
/// density.
struct Grid {
    cell: f64,
    n_x: usize,
    domain_x: f64,
    // y can be negative-ish only transiently; we offset by a bias so all
    // used y-cell indices are non-negative isize before casting to usize
    // for the row key. We store cells in a Vec<Vec<usize>> keyed by
    // (cell_x, cell_y_bucket) via a flat map. This is a HashMap for O(1)
    // buckets, but it is ONLY ever touched via `.entry()`/`.get()` on
    // exact keys -- never iterated -- so hash-order is not load-bearing
    // and determinism is preserved (the deterministic part is the fixed
    // (dx, dy) visit order in `neighbors_of`, not map iteration order).
    buckets: std::collections::HashMap<(usize, i64), Vec<usize>>,
}

impl Grid {
    fn build(agents: &[Agent], cell: f64, n_x: usize, domain_x: f64) -> Self {
        let mut buckets: std::collections::HashMap<(usize, i64), Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, a) in agents.iter().enumerate() {
            let cx = cell_index_x(a.x, cell, n_x, domain_x);
            let cy = (a.y / cell).floor() as i64;
            buckets.entry((cx, cy)).or_default().push(idx);
        }
        Grid {
            cell,
            n_x,
            domain_x,
            buckets,
        }
    }

    /// Deterministically ordered candidate neighbor indices (including the
    /// agent's own cell) for the cell containing agent `idx` at position
    /// (x, y): the 3x3 block of cells centered on it, visited in a fixed
    /// (dx, dy) order, with cyclic wrap on X.
    fn neighbors_of(&self, x: f64, y: f64) -> Vec<usize> {
        let cx = cell_index_x(x, self.cell, self.n_x, self.domain_x) as i64;
        let cy = (y / self.cell).floor() as i64;
        let mut out = Vec::new();
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let nx = ((cx + dx).rem_euclid(self.n_x as i64)) as usize;
                let ny = cy + dy;
                if let Some(v) = self.buckets.get(&(nx, ny)) {
                    out.extend_from_slice(v);
                }
            }
        }
        out
    }
}

/// Bucket index for `x` along the X axis. `x` may be transiently outside
/// `[0, domain_x)` (e.g. a freshly-divided daughter agent placed at
/// `parent.x + r*cos(angle)` before the end-of-iteration wrap runs), so we
/// canonicalize into `[0, domain_x)` via `rem_euclid` FIRST, before
/// dividing by `cell`. Skipping this step means an off-canonical x can
/// floor-divide to a raw cell index that differs from its canonical
/// counterpart by exactly `n_x` (or more), which then survives the
/// `rem_euclid(n_x)` wrap as a *different* bucket than the canonical
/// position would use -- silently missing genuinely-overlapping pairs that
/// straddle the wrap boundary whenever `domain_x` isn't an exact multiple
/// of `cell`.
fn cell_index_x(x: f64, cell: f64, n_x: usize, domain_x: f64) -> usize {
    let canon = x.rem_euclid(domain_x);
    let raw = (canon / cell).floor() as i64;
    raw.rem_euclid(n_x as i64) as usize
}

pub fn relax(agents: &mut Vec<Agent>, density: f64, domain_x: f64, iters: usize, k: f64) {
    let n = agents.len();
    if n == 0 {
        return;
    }
    for _ in 0..iters {
        // Cell size: >= the maximum possible interaction distance (sum of
        // the two largest radii), so any genuinely-overlapping pair falls
        // in the same or an adjacent cell. All agents share `density`, so
        // radius is a monotonic function of mass alone.
        let max_r = agents
            .iter()
            .map(|a| a.radius(density))
            .fold(0.0_f64, f64::max);
        let cell = (2.0 * max_r).max(1e-9);
        let n_x = ((domain_x / cell).floor() as usize).max(1);

        let grid = Grid::build(agents, cell, n_x, domain_x);

        for i in 0..n {
            let (xi, yi) = (agents[i].x, agents[i].y);
            let mut candidates = grid.neighbors_of(xi, yi);
            candidates.sort_unstable();
            candidates.dedup();
            for j in candidates {
                if j > i {
                    push_pair(agents, i, j, density, domain_x, k);
                }
            }
        }

        for a in agents.iter_mut() {
            // wrap X
            a.x = a.x.rem_euclid(domain_x);
            // substratum constraint
            let r = a.radius(density);
            if a.y < r {
                a.y = r;
            }
        }
    }
}
