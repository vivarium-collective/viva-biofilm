use biofilm_core::agent::Agent;
use biofilm_core::detachment::{detach_above_height, erode_surface};
use biofilm_core::World;

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

// ---- Task 2: rate-based Wanner-Gujer erosion ---------------------------

#[test]
fn erosion_is_noop_when_k_det_is_zero() {
    let mut a = vec![
        Agent { x: 0.0, y: 10.0, mass: 0.2, species: 0 },
        Agent { x: 2.0, y: 40.0, mass: 0.2, species: 0 },
        Agent { x: 4.0, y: 90.0, mass: 0.2, species: 0 },
    ];
    let before = a.clone();
    let removed = erode_surface(&mut a, 0.0, 0.05, 2.0, 16.0, 0.15);
    assert_eq!(removed, 0);
    assert_eq!(a.len(), before.len());
    for (got, want) in a.iter().zip(before.iter()) {
        assert_eq!(got.x, want.x);
        assert_eq!(got.y, want.y);
        assert_eq!(got.mass, want.mass);
    }
    // Negative rates are also a no-op (defensive: never erode "backwards").
    let mut b = before.clone();
    let removed_neg = erode_surface(&mut b, -1.0, 0.05, 2.0, 16.0, 0.15);
    assert_eq!(removed_neg, 0);
    assert_eq!(b.len(), before.len());
}

#[test]
fn erosion_never_strips_the_substratum_monolayer() {
    // A single tall column: agents stacked at x=0, y = 0, 5, 10, ..., 50.
    // Even with a huge k_det/dt (Delta >> h_max), the bottom-most agent
    // (y=0) must survive one erode_surface call.
    let mut a: Vec<Agent> = (0..=10)
        .map(|i| Agent { x: 0.0, y: (i * 5) as f64, mass: 0.2, species: 0 })
        .collect();
    let removed = erode_surface(&mut a, 1.0, 100.0, 2.0, 16.0, 0.15); // huge Delta
    assert!(removed > 0, "expected some erosion to occur");
    assert!(!a.is_empty(), "erosion must never empty a non-empty column");
    let y_min = a.iter().map(|x| x.y).fold(f64::INFINITY, f64::min);
    assert_eq!(y_min, 0.0, "bottom-most (substratum) agent must survive even under extreme erosion");
    // And erosion is top-down only: nothing above the (clamped) threshold
    // survives except the protected bottom layer.
    assert!(a.iter().all(|ag| ag.y <= 5.0), "erosion should have removed everything above the bottom-most layer");
}

fn build_dev_biofilm(ny: usize, k_det: f64) -> World {
    let mut w = World::new();
    w.set_domain(16, ny, 2.0, ny as f64 * 2.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.set_detachment_rate(k_det);
    w.spawn_agents(30, 1.0, 0);
    w.finalize(1234);
    w
}

#[test]
fn erosion_bounds_thickness_to_a_plateau() {
    // k_det=0.02 here is a TEST-ONLY calibration (larger dt -> larger
    // per-step Delta -> faster-converging, cheaper-to-run plateau) chosen
    // just to demonstrate the qualitative behavior quickly; it is NOT the
    // production `detachment_rate` (see run.py's BIOFILM_SPEC, calibrated
    // separately at the production dt=0.05 to plateau in the tens-of-um
    // range -- see task2-erosion-report.md).
    let mut eroded = build_dev_biofilm(32, 2e-2);
    let mut control = build_dev_biofilm(32, 0.0);

    const N_STEPS: usize = 80;
    const DT: f64 = 0.2;
    let mut eroded_heights = Vec::with_capacity(N_STEPS);
    for _ in 0..N_STEPS {
        eroded.step(DT);
        control.step(DT);
        eroded_heights.push(eroded.biofilm_thickness());
    }

    assert!(eroded.population() > 0, "erosion must not empty the biofilm");

    // Plateau check: the max height over the last third of the run should
    // not still be climbing -- compare the last-third window's max against
    // the window just before it; it should not have grown appreciably.
    let last_third = &eroded_heights[eroded_heights.len() - 25..];
    let prev_third = &eroded_heights[eroded_heights.len() - 50..eroded_heights.len() - 25];
    let last_max = last_third.iter().cloned().fold(0.0_f64, f64::max);
    let prev_max = prev_third.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        last_max <= prev_max * 1.15,
        "eroded biofilm height should have plateaued, not still be climbing: prev window max {}, last window max {}",
        prev_max,
        last_max
    );

    // The no-erosion control must end up taller than the eroded run --
    // erosion is actually doing something, not just coincidentally flat.
    assert!(
        control.biofilm_thickness() > eroded.biofilm_thickness(),
        "no-erosion control ({}) should exceed eroded run ({})",
        control.biofilm_thickness(),
        eroded.biofilm_thickness()
    );
}
