// crates/biofilm-core/tests/chemostat.rs
use biofilm_core::chemostat::{Chemostat, LinearReaction};

#[test]
fn first_order_decay_matches_analytic() {
    let mut c = Chemostat::new(vec![2.0, 2.0]);
    c.add_reaction(LinearReaction { substrate: 0, k: 0.1, stoich: vec![(0, -1.0), (1, 0.5)] });
    let dt = 0.01;
    for _ in 0..6000 {
        c.step_heun(dt);
    }
    let t = 60.0;
    let analytic_s1 = 2.0 * (-0.1_f64 * t).exp();
    assert!((c.conc(0) - analytic_s1).abs() < 1e-3, "got {} want {}", c.conc(0), analytic_s1);
    // solute2 rises by 0.5 * (consumed solute1)
    let consumed = 2.0 - c.conc(0);
    assert!((c.conc(1) - (2.0 + 0.5 * consumed)).abs() < 1e-3);
}
