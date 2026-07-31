use biofilm_core::World;
use biofilm_core::grid::{Grid, SoluteField, solve_steady_state};

/// A looser tolerance should converge in strictly fewer iterations than a
/// tight one on the same setup, and the resulting field should still be
/// close (within a few %) to the tight-tolerance solution -- proving the
/// speed/accuracy tradeoff is safe for the visualization/simulation regime.
#[test]
fn looser_tolerance_converges_faster_and_field_stays_close() {
    let g = Grid::new(4, 16, 1.0);
    let sink = vec![0.5; g.nx * g.ny]; // uniform consumption

    let mut f_tight = SoluteField::new(&g, 5.0, 2000.0);
    let iters_tight = solve_steady_state(&mut f_tight, &g, &sink, 5.0, 1.4, 1e-9, 50_000);

    let mut f_loose = SoluteField::new(&g, 5.0, 2000.0);
    let iters_loose = solve_steady_state(&mut f_loose, &g, &sink, 5.0, 1.8, 1e-4, 2_000);

    assert!(
        iters_loose < iters_tight,
        "expected loose tol to converge faster: loose={iters_loose} tight={iters_tight}"
    );

    for idx in 0..f_tight.conc.len() {
        let a = f_tight.conc[idx];
        let b = f_loose.conc[idx];
        let denom = a.abs().max(1e-9);
        let rel_diff = (a - b).abs() / denom;
        assert!(
            rel_diff < 0.05,
            "cell {idx} diverged beyond 5%: tight={a} loose={b} rel_diff={rel_diff}"
        );
    }
}

fn build_world() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(30, 1.0, 0);
    w
}

/// `set_pde_params` must be settable pre-`finalize` and the resulting run
/// must remain fully deterministic (same seed -> identical population and
/// biomass), matching the guarantee already covered for the default knobs
/// in `tests/world.rs`.
#[test]
fn set_pde_params_is_configurable_and_deterministic() {
    let run = || {
        let mut w = build_world();
        w.set_pde_params(1e-5, 5_000, 1.7);
        w.finalize(1234);
        for _ in 0..20 {
            w.step(0.05);
        }
        (w.population(), w.total_biomass())
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "custom pde params must remain deterministic");
}

/// The fast defaults (tol=1e-4, max_iter=2000, omega=1.8), used implicitly
/// when `set_pde_params` is never called, must still be an unambiguous
/// improvement over the old defaults in the sense that they converge inside
/// the new, much smaller `max_iter` budget for a representative setup.
#[test]
fn default_pde_params_converge_within_new_fast_budget() {
    let g = Grid::new(32, 48, 2.0);
    let sink = vec![0.05; g.nx * g.ny];
    let mut f = SoluteField::new(&g, 1.0, 2000.0 * 86_400.0);
    let iters = solve_steady_state(&mut f, &g, &sink, 1.0, 1.8, 1e-4, 2_000);
    assert!(iters < 2_000, "expected convergence within budget, used all {iters} iterations");
}
