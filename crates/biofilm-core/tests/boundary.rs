use biofilm_core::World;

#[test]
fn set_bulk_changes_steady_state_toward_new_boundary() {
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    w.set_species(0.15, 0.2);
    w.finalize(1);
    w.step(0.05);
    let before = w.solute_row_mean("solute", 15);
    w.set_bulk(s, 5.0);
    w.step(0.05);
    let after = w.solute_row_mean("solute", 15);
    assert!(after > before, "raising bulk should raise the boundary row: {} -> {}", before, after);
}
