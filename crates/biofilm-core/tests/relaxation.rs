use biofilm_core::agent::Agent;
use biofilm_core::relaxation::relax;

#[test]
fn overlapping_agents_are_pushed_apart() {
    let mut a = vec![
        Agent { x: 10.0, y: 10.0, mass: 0.2, species: 0 },
        Agent { x: 10.2, y: 10.0, mass: 0.2, species: 0 },
    ];
    let density = 0.15;
    let sep0 = (a[0].x - a[1].x).abs();
    relax(&mut a, density, 100.0, 50, 0.5);
    let sep1 = (a[0].x - a[1].x).abs();
    let r = a[0].radius(density);
    assert!(sep1 > sep0, "agents should separate: {} -> {}", sep0, sep1);
    assert!(sep1 >= 2.0 * r - 1e-3, "final sep should reach ~2r");
}

#[test]
fn agents_stay_above_substratum() {
    let mut a = vec![Agent { x: 5.0, y: 0.0, mass: 0.2, species: 0 }];
    relax(&mut a, 0.15, 100.0, 10, 0.5);
    assert!(a[0].y >= a[0].radius(0.15) - 1e-6);
}
