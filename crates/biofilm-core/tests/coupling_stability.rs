use biofilm_core::World;

/// Table-K-like stiff single-species world: mu_max/Ks/yield chosen to match
/// the diagnosed period-2 oxygen limit-cycle regime (see
/// `.superpowers/sdd/substrate-limitation/task1-coupling-brief.md`). A small
/// grid, real oxygen diffusivity (2000 um^2/s, converted internally to
/// um^2/day), and enough agents to drive the substratum oxygen toward
/// depletion each macro-step.
fn build_stiff_world() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    w.set_species(0.1363, 0.08);
    w.add_reaction(3.94, vec![(o, 0.6)], vec![(o, -43.478)]);
    w.spawn_agents(60, 1.0, 0);
    w.finalize(7);
    w
}

/// Pre-fix (lagged single-Picard sink<->solve coupling), this reproduces the
/// diagnosed period-2 oxygen limit cycle: the substratum-row mean oxygen
/// swings ~0.0 <-> ~0.96 every other step under stiff Table-K-like kinetics,
/// which defeats Monod substrate limitation and lets the population run
/// away unbounded. Post-fix (Picard-iterate sink<->solve to
/// self-consistency each macro-step), the field should settle to a smooth,
/// low, non-oscillating steady state and growth should stay bounded.
#[test]
fn stiff_kinetics_do_not_oscillate_or_run_away() {
    let mut w = build_stiff_world();
    let dt = 1.0 / 24.0;

    let mut row0_means: Vec<f64> = Vec::with_capacity(100);
    let mut pop_at_60: usize = 0;
    let mut converged_steps_60_100: usize = 0;
    for i in 0..100 {
        w.step(dt);
        row0_means.push(w.solute_row_mean("oxygen", 0));
        if i == 59 {
            pop_at_60 = w.population();
        }
        if i >= 60 && w.last_coupling_converged() {
            converged_steps_60_100 += 1;
        }
    }
    let pop_final = w.population();

    // Guard against a regression to a non-convergent coupling loop (e.g. too
    // weak a `COUPLING_RELAXATION`): if the loop is riding the
    // `MAX_COUPLING_ITERS` cap instead of breaking on `COUPLING_TOL`, growth
    // reads an arbitrary non-converged iterate of what can be a still-
    // cycling limit cycle rather than the true self-consistent field — which
    // happened to still pass the amplitude/population checks below by a
    // thin margin during development (w=0.5: 0/40 post-transient steps
    // converged) but is not the self-consistency the fix is supposed to
    // deliver. Require the large majority of post-transient steps to
    // actually converge.
    let window_len = 100 - 60;
    assert!(
        converged_steps_60_100 * 10 >= window_len * 9,
        "coupling loop is not converging (riding MAX_COUPLING_ITERS instead of \
         breaking on COUPLING_TOL): only {converged_steps_60_100}/{window_len} \
         post-transient steps broke on tol — growth is reading arbitrary \
         non-converged iterates, not the self-consistent field"
    );

    // Post-transient window: no period-2 swing left. Pre-fix this window
    // swings ~0.0 <-> ~0.96 (amplitude ~0.96); post-fix, with a genuinely
    // convergent coupling loop, the field is smoothly low (amplitude small,
    // measured ~0.018 at COUPLING_RELAXATION=0.2 — well under this
    // threshold, unlike the ~0.14 seen with a non-convergent w=0.5).
    let window = &row0_means[60..100];
    let max = window.iter().cloned().fold(f64::MIN, f64::max);
    let min = window.iter().cloned().fold(f64::MAX, f64::min);
    let amplitude = max - min;
    assert!(
        amplitude < 0.05,
        "oxygen row-0 mean oscillating over steps 60..100 (period-2 limit cycle): \
         max={max} min={min} amplitude={amplitude}, window={window:?}"
    );

    // Population must stay bounded, not run away under the (correctly
    // scaled, correctly local) Monod substrate limitation.
    assert!(
        pop_final < 8 * pop_at_60.max(1),
        "population ran away: pop_at_step60={pop_at_60} pop_final={pop_final}"
    );
}
