use biofilm_core::agent::{Agent, grow, divide};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn growth_is_euler() {
    let mut a = vec![Agent { x: 0.0, y: 0.0, mass: 1.0, species: 0 }];
    grow(&mut a, &[0.5], 2.0); // +1.0
    assert!((a[0].mass - 2.0).abs() < 1e-12);
}

#[test]
fn division_splits_mass_and_conserves_total() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut a = vec![Agent { x: 10.0, y: 10.0, mass: 0.3, species: 0 }];
    divide(&mut a, 0.2, &mut rng);
    assert_eq!(a.len(), 2);
    let total: f64 = a.iter().map(|x| x.mass).sum();
    assert!((total - 0.3).abs() < 1e-12);
    assert!((a[0].mass - 0.15).abs() < 1e-12);
}

#[test]
fn division_is_deterministic_for_seed() {
    let run = || {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut a = vec![Agent { x: 5.0, y: 5.0, mass: 0.5, species: 0 }];
        divide(&mut a, 0.2, &mut rng);
        (a[1].x, a[1].y)
    };
    assert_eq!(run(), run());
}
