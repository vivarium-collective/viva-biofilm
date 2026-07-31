use pyo3::prelude::*;
use ::biofilm_core::World as CoreWorld;

#[pyclass]
pub struct World {
    inner: CoreWorld,
}

#[pymethods]
impl World {
    #[new]
    fn new() -> Self {
        World { inner: CoreWorld::new() }
    }
    fn step(&mut self, dt: f64) {
        self.inner.step(dt);
    }
    fn time(&self) -> f64 {
        self.inner.time()
    }
}

#[pymodule]
fn biofilm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
