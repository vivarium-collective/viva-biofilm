pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64, // isotropic cell size (µm)
}

impl Grid {
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Grid { nx, ny, dx }
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i + j * self.nx
    }
}

pub struct SoluteField {
    pub conc: Vec<f64>,
    pub diffusivity: f64,
    pub bulk: f64,
    nx: usize,
}

impl SoluteField {
    pub fn new(g: &Grid, init: f64, diffusivity: f64) -> Self {
        SoluteField {
            conc: vec![init; g.nx * g.ny],
            diffusivity,
            bulk: init,
            nx: g.nx,
        }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.conc[i + j * self.nx]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.conc[i + j * self.nx] = v;
    }
}

/// Red-black SOR relaxation of D * laplacian(C) - sink = 0.
/// X cyclic, j=0 no-flux (mirror neighbour), j=ny-1 Dirichlet.
pub fn solve_steady_state(
    f: &mut SoluteField,
    g: &Grid,
    sink: &[f64],
    top_dirichlet: f64,
    sor: f64,
    tol: f64,
    max_iter: usize,
) -> usize {
    let d = f.diffusivity;
    let h2 = g.dx * g.dx;
    // Fix Dirichlet top row.
    for i in 0..g.nx {
        f.set(i, g.ny - 1, top_dirichlet);
    }
    for iter in 0..max_iter {
        let mut max_res: f64 = 0.0;
        for color in 0..2 {
            for j in 0..(g.ny - 1) {
                for i in 0..g.nx {
                    if (i + j) % 2 != color {
                        continue;
                    }
                    let ip = f.at((i + 1) % g.nx, j);
                    let im = f.at((i + g.nx - 1) % g.nx, j);
                    let jp = f.at(i, j + 1);
                    // no-flux at substratum: neighbour below j=0 mirrors j itself
                    let jm = if j == 0 { f.at(i, j) } else { f.at(i, j - 1) };
                    // Discrete: D/h2 * (ip+im+jp+jm - 4C) - sink = 0
                    let s = sink[g.idx(i, j)];
                    let new = (d / h2 * (ip + im + jp + jm) - s) / (4.0 * d / h2);
                    let old = f.at(i, j);
                    let relaxed = old + sor * (new - old);
                    f.set(i, j, relaxed.max(0.0));
                    max_res = max_res.max((relaxed - old).abs());
                }
            }
        }
        if max_res < tol {
            return iter + 1;
        }
    }
    max_iter
}
