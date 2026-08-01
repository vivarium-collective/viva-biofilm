use biofilm_core::World;

fn two_strategy_world(n_each: usize) -> World {
    let mut w = World::new();
    w.set_domain(50, 50, 2.0, 40.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    // Configure pre-populated species[0] for RS (higher mu_max, lower yield)
    let rs = 0;
    w.set_species(0.15, 0.2);
    w.add_reaction_for(rs, 3.0, vec![(o, 0.3)], vec![(o, -2.0)]);
    // Add new species[1] for YS (lower mu_max, higher yield)
    let ys = w.add_species(0.15, 0.2);
    w.add_reaction_for(ys, 1.5, vec![(o, 0.3)], vec![(o, -1.0)]);
    w.spawn_distributed(rs, n_each, 1.0, 0);
    w.spawn_distributed(ys, n_each, 1.0, 1);
    w.finalize(42);
    w
}

#[test]
fn spawns_equal_counts_and_tracks_per_strategy() {
    let w = two_strategy_world(10);
    assert_eq!(w.population_of(0), 10);
    assert_eq!(w.population_of(1), 10);
    assert_eq!(w.population(), 20);
    assert!(w.biomass_of(0) > 0.0 && w.biomass_of(1) > 0.0);
}

#[test]
fn competition_is_deterministic() {
    let run = || {
        let mut w = two_strategy_world(10);
        for _ in 0..30 {
            w.step(0.05);
        }
        (w.population_of(0), w.population_of(1))
    };
    assert_eq!(run(), run());
}
