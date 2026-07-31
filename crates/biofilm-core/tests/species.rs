use biofilm_core::World;

#[test]
fn two_species_grow_by_their_own_mu_max() {
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    let fast = w.add_species(0.15, 1e9); // huge division mass so they don't divide, just grow
    let slow = w.add_species(0.15, 1e9);
    w.add_reaction_for(fast, 4.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    w.add_reaction_for(slow, 1.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    // seed one agent of each species at the same spot-ish
    w.spawn_distributed(fast, 1, 1.0, 0);
    w.spawn_distributed(slow, 1, 1.0, 1);
    w.finalize(1);
    let m0: Vec<f64> = w.agents().iter().map(|a| a.mass).collect();
    for _ in 0..5 { w.step(0.02); }
    let m1: Vec<f64> = w.agents().iter().map(|a| a.mass).collect();
    // both grew; the fast-species agent gained more mass than the slow one
    let gained: Vec<f64> = m1.iter().zip(&m0).map(|(a,b)| a-b).collect();
    assert!(gained.iter().all(|&g| g > 0.0), "both should grow: {:?}", gained);
    // identify which agent is which species
    let sp: Vec<u16> = w.agents().iter().map(|a| a.species).collect();
    let fast_gain: f64 = gained.iter().zip(&sp).filter(|(_,s)| **s == fast as u16).map(|(g,_)| *g).sum();
    let slow_gain: f64 = gained.iter().zip(&sp).filter(|(_,s)| **s == slow as u16).map(|(g,_)| *g).sum();
    assert!(fast_gain > slow_gain, "fast species should gain more: {} vs {}", fast_gain, slow_gain);
}

#[test]
fn backward_compat_single_species_still_works() {
    // old API path: set_species + add_reaction + spawn_agents
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    w.add_reaction(2.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(10, 1.0, 0);
    w.finalize(1);
    let p0 = w.population();
    for _ in 0..10 { w.step(0.05); }
    assert!(w.population() >= p0);
}
