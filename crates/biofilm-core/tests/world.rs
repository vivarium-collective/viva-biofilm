use biofilm_core::World;

fn build() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(30, 1.0, 0);
    w.finalize(1234);
    w
}

#[test]
fn biofilm_grows_and_is_deterministic() {
    let run = || {
        let mut w = build();
        for _ in 0..20 {
            w.step(0.05);
        }
        (w.population(), w.total_biomass())
    };
    let (p0, b0) = run();
    assert!(p0 >= 30, "population should not shrink below seed: {}", p0);
    assert!(b0 > 0.0);
    // determinism: identical seed -> identical result
    assert_eq!(run(), (p0, b0));
}

#[test]
fn thickness_within_domain() {
    let mut w = build();
    for _ in 0..30 {
        w.step(0.05);
    }
    assert!(w.biofilm_thickness() <= 32.0 * 2.0);
}
