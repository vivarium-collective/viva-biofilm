use biofilm_core::reaction::{Monod, Reaction};

#[test]
fn monod_saturates_to_mu_max() {
    let m = Monod { mu_max: 2.0, terms: vec![(0, 1.0)] };
    // Very high substrate -> fraction ~ 1 -> rate ~ mu_max
    assert!((m.specific_rate(&[1e6]) - 2.0).abs() < 1e-3);
    // At S = Ks the fraction is exactly 0.5
    assert!((m.specific_rate(&[1.0]) - 1.0).abs() < 1e-9);
    // Zero substrate -> zero rate
    assert_eq!(m.specific_rate(&[0.0]), 0.0);
}

#[test]
fn dual_monod_multiplies_fractions() {
    let m = Monod { mu_max: 1.0, terms: vec![(0, 1.0), (1, 1.0)] };
    // both at Ks -> 0.5 * 0.5 = 0.25
    assert!((m.specific_rate(&[1.0, 1.0]) - 0.25).abs() < 1e-9);
}

#[test]
fn biomass_rate_scales_with_mass() {
    let r = Reaction {
        kinetics: Monod { mu_max: 2.0, terms: vec![(0, 1.0)] },
        yield_per_solute: vec![(0, -4.2)],
    };
    assert!((r.biomass_rate(3.0, &[1e6]) - 6.0).abs() < 1e-2);
}
