use biofilm_core::agent::Agent;
use biofilm_core::detachment::detach_above_height;

#[test]
fn removes_agents_above_cap() {
    let mut a = vec![
        Agent { x: 0.0, y: 10.0, mass: 0.2, species: 0 },
        Agent { x: 0.0, y: 70.0, mass: 0.2, species: 0 },
    ];
    let removed = detach_above_height(&mut a, 64.0);
    assert_eq!(removed, 1);
    assert_eq!(a.len(), 1);
    assert!(a[0].y <= 64.0);
}
