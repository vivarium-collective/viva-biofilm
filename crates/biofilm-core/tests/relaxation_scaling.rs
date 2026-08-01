use biofilm_core::agent::Agent;
use biofilm_core::relaxation::relax;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

const DENSITY: f64 = 0.15;
const MASS: f64 = 0.2;

fn agent_radius() -> f64 {
    (MASS / (std::f64::consts::PI * DENSITY)).sqrt()
}

/// N agents scattered over a domain whose side scales with sqrt(n), so the
/// *packing density* (and thus the average number of neighbors per agent)
/// stays roughly constant as n grows. This is what makes O(n) vs O(n^2)
/// show up as a clean time ratio rather than being swamped by a
/// simultaneously-growing neighbor count.
fn packed_agents(n: usize, seed: u64) -> (Vec<Agent>, f64) {
    let r = agent_radius();
    let side = (n as f64).sqrt() * 6.0 * r;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let agents: Vec<Agent> = (0..n)
        .map(|_| Agent {
            x: rng.gen_range(0.0..side),
            y: rng.gen_range(0.0..side),
            mass: MASS,
            species: 0,
        })
        .collect();
    (agents, side)
}

fn time_relax(n: usize) -> f64 {
    let (mut agents, side) = packed_agents(n, 7);
    let start = Instant::now();
    relax(&mut agents, DENSITY, side, 30, 0.5);
    start.elapsed().as_secs_f64()
}

#[test]
fn relax_scales_roughly_linearly_not_quadratically() {
    // Warm up so the first JIT-free but cache-cold call doesn't skew n=100.
    let _ = time_relax(50);

    let t100 = time_relax(100);
    let t500 = time_relax(500);
    let t2000 = time_relax(2000);

    println!(
        "relax timing: n=100 {:.6}s, n=500 {:.6}s, n=2000 {:.6}s",
        t100, t500, t2000
    );

    // n grows 4x from 500 -> 2000. A true O(n^2) algorithm scales ~16x over
    // that range; O(n) scales ~4x. Allow generous slack for timing noise.
    let ratio = t2000 / t500.max(1e-9);
    assert!(
        ratio < 8.0,
        "relax time should grow roughly linearly (not quadratically): \
         t(500)={:.6}s t(2000)={:.6}s ratio={:.2} (expected < 8.0)",
        t500,
        t2000,
        ratio
    );
}

#[test]
fn dense_cluster_fully_separates_no_residual_overlap() {
    // A tight cluster of mutually-overlapping agents should end up with
    // essentially no overlapping pairs after enough iterations. This
    // proves the grid neighbor search finds ALL genuinely-overlapping
    // pairs each iteration (a cell-size mistake would silently leave some
    // pairs unchecked and permanently overlapping).
    let r = agent_radius();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let n = 60;
    let mut agents: Vec<Agent> = (0..n)
        .map(|_| Agent {
            x: 50.0 + rng.gen_range(-0.3..0.3),
            y: 5.0 + rng.gen_range(-0.3..0.3),
            mass: MASS,
            species: 0,
        })
        .collect();

    relax(&mut agents, DENSITY, 200.0, 400, 0.5);

    let tol = 1e-2;
    let min_d = 2.0 * r;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = agents[i].x - agents[j].x;
            let dy = agents[i].y - agents[j].y;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(
                dist >= min_d - tol,
                "agents {} and {} still overlap after relaxation: dist={:.4} min_d={:.4}",
                i,
                j,
                dist,
                min_d
            );
        }
    }
}

fn cyclic_dist_x(xa: f64, xb: f64, domain_x: f64) -> f64 {
    let mut d = xa - xb;
    if d > domain_x / 2.0 {
        d -= domain_x;
    } else if d < -domain_x / 2.0 {
        d += domain_x;
    }
    d.abs()
}

#[test]
fn straddling_wrap_boundary_overlap_is_found_and_pushed_apart() {
    // Reproduces the missed-overlap bug found by brute-force review: a
    // fresh daughter agent from `divide_with_density` is placed at
    // `parent.x + r*cos(angle)` WITHOUT wrapping into [0, domain_x), so on
    // the very first `relax` iteration after a division near the wrap
    // boundary, agent x can be transiently outside [0, domain_x). If the
    // grid buckets cells from the *raw* (un-canonicalized) x, two agents
    // that are genuinely overlapping under the cyclic-X metric can land in
    // non-adjacent cells and never get checked as a candidate pair in that
    // iteration (self-healing only kicks in on the NEXT iteration, once
    // the end-of-iteration `rem_euclid` re-canonicalizes x -- but the
    // overlap should have been resolved on iteration 1).
    //
    // Numbers below are chosen so the ratio domain_x/cell is non-integer
    // (9.6 / 2.0 = 4.8, n_x = 4) and the buggy `cell_index_x` (no
    // canonicalization) buckets these two agents into cell 3 and cell 1
    // (not adjacent mod 4), while canonicalizing x by `domain_x` first
    // buckets both into cell 0.
    let density = 0.15;
    let r = 1.0_f64;
    let mass = r * r * std::f64::consts::PI * density;
    let cell = 2.0 * r; // matches relax()'s cell = 2 * max_r
    let domain_x = 4.8 * cell; // = 9.6; non-integer domain_x/cell ratio

    let mut agents = vec![
        Agent {
            x: -0.5, // just outside [0, domain_x) on the low side
            y: r,
            mass,
            species: 0,
        },
        Agent {
            x: domain_x + 0.5, // just outside [0, domain_x) on the high side
            y: r,
            mass,
            species: 0,
        },
    ];

    let min_d = 2.0 * r;
    let dist0 = cyclic_dist_x(agents[0].x, agents[1].x, domain_x);
    assert!(
        dist0 < min_d,
        "test setup bug: agents aren't actually overlapping (dist={:.4} min_d={:.4})",
        dist0,
        min_d
    );

    // A single iteration: if the pair is found, push_pair moves them apart
    // by `k * overlap` total (k=0.5, overlap=1.0 -> new dist = 1.5). If the
    // pair is missed (the bug), nothing touches their x and dist is
    // unchanged at 1.0.
    relax(&mut agents, density, domain_x, 1, 0.5);

    let dist = cyclic_dist_x(agents[0].x, agents[1].x, domain_x);
    let expected = 1.5;
    assert!(
        (dist - expected).abs() < 1e-6,
        "overlapping pair straddling the wrap boundary was not separated by relax() \
         (grid must have missed it as a candidate pair): dist={:.6} expected~={:.6} (started at {:.6})",
        dist,
        expected,
        dist0
    );
}
