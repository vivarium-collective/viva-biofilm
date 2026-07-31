use biofilm_core::grid::{Grid, SoluteField, solve_steady_state};

#[test]
fn no_sink_gives_uniform_field_equal_to_boundary() {
    let g = Grid::new(8, 8, 1.0);
    let mut f = SoluteField::new(&g, 1.0, 2000.0); // conc init 1.0, D=2000
    let sink = vec![0.0; g.nx * g.ny];
    solve_steady_state(&mut f, &g, &sink, 5.0, 1.4, 1e-8, 10_000);
    // With no consumption and top fixed at 5.0, steady state is uniform 5.0.
    for j in 0..g.ny {
        for i in 0..g.nx {
            assert!((f.at(i, j) - 5.0).abs() < 1e-4, "cell {},{} = {}", i, j, f.at(i, j));
        }
    }
}

#[test]
fn uniform_sink_creates_gradient_decreasing_into_biofilm() {
    let g = Grid::new(4, 16, 1.0);
    let mut f = SoluteField::new(&g, 5.0, 2000.0);
    let sink = vec![0.5; g.nx * g.ny]; // uniform consumption
    solve_steady_state(&mut f, &g, &sink, 5.0, 1.4, 1e-9, 50_000);
    // Concentration at substratum (j=0) must be below the boundary value.
    assert!(f.at(0, 0) < f.at(0, g.ny - 1));
    assert!(f.at(0, 0) >= 0.0);
}
