use biofilm_core::World;

fn build_dev_biofilm() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    // diffusivities are given in µm²/s at the schema layer; here pass the SAME
    // numbers the schema will pass. Task-1's job is to make the solver treat
    // time consistently in days (see world.rs unit note).
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(30, 1.0, 0);
    w.finalize(1234);
    w
}

#[test]
fn developed_biofilm_has_partial_substrate_gradient() {
    let mut w = build_dev_biofilm();
    for _ in 0..40 { w.step(0.05); }
    // A biofilm consuming substrate must draw the substratum concentration
    // DOWN relative to the bulk boundary, but NOT to zero everywhere
    // (a flat field or a fully-depleted field are both calibration failures).
    let sub_bottom = w.solute_row_mean("solute", 0);       // near substratum
    let sub_top = w.solute_row_mean("solute", 31);         // near boundary
    assert!(sub_top > sub_bottom, "expected gradient: top {} > bottom {}", sub_top, sub_bottom);
    assert!(sub_bottom < 0.9 * sub_top, "gradient too weak (nearly flat): {} vs {}", sub_bottom, sub_top);
    assert!(sub_bottom > 0.0, "substrate fully depleted — sink too strong");
}

#[test]
fn biofilm_grows_over_time() {
    let mut w = build_dev_biofilm();
    let p0 = w.population();
    for _ in 0..40 { w.step(0.05); }
    assert!(w.population() > p0, "biofilm should grow: {} -> {}", p0, w.population());
    assert!(w.total_biomass() > 0.0);
}
