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
    for i in 0..100 {
        w.step(dt);
        row0_means.push(w.solute_row_mean("oxygen", 0));
        if i == 59 {
            pop_at_60 = w.population();
        }
    }
    let pop_final = w.population();

    // Post-transient window: no period-2 swing left. Pre-fix this window
    // swings ~0.0 <-> ~0.96 (amplitude ~0.96); post-fix the field is smoothly
    // low (amplitude small).
    let window = &row0_means[60..100];
    let max = window.iter().cloned().fold(f64::MIN, f64::max);
    let min = window.iter().cloned().fold(f64::MAX, f64::min);
    let amplitude = max - min;
    assert!(
        amplitude < 0.15,
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
