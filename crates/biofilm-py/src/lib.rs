use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::biofilm_core::chemostat::{Chemostat, LinearReaction};
use ::biofilm_core::World as CoreWorld;

#[pyclass]
pub struct World {
    inner: CoreWorld,
    // Metadata tracked here (not in core) so readbacks can resolve names/
    // shape/radius without changing biofilm-core's builder/accessor surface.
    grid: (usize, usize),
    solute_names: Vec<String>,
    solute_index: HashMap<String, usize>,
    density: f64,
}

#[pymethods]
impl World {
    #[new]
    fn new() -> Self {
        World {
            inner: CoreWorld::new(),
            grid: (0, 0),
            solute_names: Vec::new(),
            solute_index: HashMap::new(),
            density: 0.15,
        }
    }

    // ---- Builders ------------------------------------------------------

    fn set_domain(&mut self, nx: usize, ny: usize, dx: f64, layer_thickness: f64) {
        self.inner.set_domain(nx, ny, dx, layer_thickness);
        self.grid = (nx, ny);
    }

    fn add_solute(
        &mut self,
        name: &str,
        init: f64,
        diff_liquid: f64,
        diff_biofilm: f64,
        bulk: f64,
    ) -> usize {
        let idx = self.inner.add_solute(name, init, diff_liquid, diff_biofilm, bulk);
        self.solute_names.push(name.to_string());
        self.solute_index.insert(name.to_string(), idx);
        idx
    }

    fn add_reaction(&mut self, mu_max: f64, monod_terms: Vec<(usize, f64)>, yields: Vec<(usize, f64)>) {
        self.inner.add_reaction(mu_max, monod_terms, yields);
    }

    /// Set a solute's boundary (`bulk`) concentration by name, resolving
    /// via the wrapper's `solute_index` name map. Lets an external process
    /// (e.g. `BiofilmProcess`, driven by its `boundary_concentrations`
    /// input) perturb the environment at runtime. Raises `ValueError` if
    /// `name` is not a known solute.
    fn set_bulk_by_name(&mut self, name: &str, value: f64) -> PyResult<()> {
        let idx = self
            .solute_index
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown solute: {name}")))?;
        self.inner.set_bulk(*idx, value);
        Ok(())
    }

    fn set_species(&mut self, density: f64, division_mass: f64) {
        self.inner.set_species(density, division_mass);
        self.density = density;
    }

    /// Configure the PDE (SOR) solver knobs: `tol`, `max_iter`, `omega`.
    /// Defaults to the fast values (1e-4, 2000, 1.8) if never called.
    /// Raises `ValueError` (rather than letting an invalid value reach the
    /// solver and silently diverge to NaN/Inf) if `omega` is not in
    /// `[1.0, 2.0)`, `tol <= 0.0`, or `max_iter == 0`.
    fn set_pde_params(&mut self, tol: f64, max_iter: usize, omega: f64) -> PyResult<()> {
        if !(1.0..2.0).contains(&omega) {
            return Err(PyValueError::new_err(format!(
                "omega must be in [1.0, 2.0) for SOR convergence, got {omega}"
            )));
        }
        if !(tol > 0.0) {
            return Err(PyValueError::new_err(format!("tol must be > 0.0, got {tol}")));
        }
        if max_iter == 0 {
            return Err(PyValueError::new_err("max_iter must be > 0, got 0"));
        }
        self.inner.set_pde_params(tol, max_iter, omega);
        Ok(())
    }

    fn spawn_agents(&mut self, n: usize, band_height: f64, seed_offset: u64) {
        self.inner.spawn_agents(n, band_height, seed_offset);
    }

    fn finalize(&mut self, seed: u64) {
        self.inner.finalize(seed);
    }

    // ---- Advance ---------------------------------------------------------

    fn step(&mut self, dt: f64) {
        self.inner.step(dt);
    }

    // ---- Readbacks -------------------------------------------------------

    fn population(&self) -> usize {
        self.inner.population()
    }

    fn total_biomass(&self) -> f64 {
        self.inner.total_biomass()
    }

    fn biofilm_thickness(&self) -> f64 {
        self.inner.biofilm_thickness()
    }

    fn solute_means(&self) -> HashMap<String, f64> {
        self.solute_names
            .iter()
            .enumerate()
            .map(|(k, name)| (name.clone(), self.inner.solute_mean(k)))
            .collect()
    }

    fn agent_positions(&self) -> Vec<(f64, f64)> {
        self.inner.agents().iter().map(|a| (a.x, a.y)).collect()
    }

    fn agent_masses(&self) -> Vec<f64> {
        self.inner.agents().iter().map(|a| a.mass).collect()
    }

    fn agent_radii(&self) -> Vec<f64> {
        self.inner.agents().iter().map(|a| a.radius(self.density)).collect()
    }

    fn agent_species(&self) -> Vec<u16> {
        self.inner.agents().iter().map(|a| a.species).collect()
    }

    fn solute_field(&self, name: &str) -> PyResult<Vec<f64>> {
        let idx = self
            .solute_index
            .get(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown solute: {name}")))?;
        Ok(self.inner.solute_field(*idx).to_vec())
    }

    fn grid_shape(&self) -> (usize, usize) {
        self.grid
    }

    fn time(&self) -> f64 {
        self.inner.time()
    }
}

#[pyclass]
pub struct ChemostatWorld {
    inner: Chemostat,
}

#[pymethods]
impl ChemostatWorld {
    #[new]
    fn new(init: Vec<f64>) -> Self {
        ChemostatWorld { inner: Chemostat::new(init) }
    }

    fn add_linear_reaction(&mut self, substrate: usize, k: f64, stoich: Vec<(usize, f64)>) {
        self.inner.add_reaction(LinearReaction { substrate, k, stoich });
    }

    fn step(&mut self, dt: f64) {
        self.inner.step_heun(dt);
    }

    fn conc(&self, k: usize) -> f64 {
        self.inner.conc(k)
    }

    fn concs(&self) -> Vec<f64> {
        self.inner.concs.clone()
    }
}

#[pymodule]
fn biofilm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    m.add_class::<ChemostatWorld>()?;
    Ok(())
}
