pub struct LinearReaction {
    pub substrate: usize,
    pub k: f64,
    pub stoich: Vec<(usize, f64)>,
}

impl LinearReaction {
    fn rate(&self, concs: &[f64]) -> f64 {
        self.k * concs[self.substrate].max(0.0)
    }
}

pub struct Chemostat {
    pub concs: Vec<f64>,
    pub rates: Vec<LinearReaction>,
}

impl Chemostat {
    pub fn new(init: Vec<f64>) -> Self {
        Chemostat { concs: init, rates: vec![] }
    }
    pub fn add_reaction(&mut self, r: LinearReaction) {
        self.rates.push(r);
    }
    fn deriv(&self, concs: &[f64]) -> Vec<f64> {
        let mut d = vec![0.0; concs.len()];
        for r in &self.rates {
            let v = r.rate(concs);
            for &(k, coeff) in &r.stoich {
                d[k] += coeff * v;
            }
        }
        d
    }
    pub fn step_heun(&mut self, dt: f64) {
        let k1 = self.deriv(&self.concs);
        let mid: Vec<f64> = self.concs.iter().zip(&k1).map(|(c, d)| c + d * dt).collect();
        let k2 = self.deriv(&mid);
        for i in 0..self.concs.len() {
            self.concs[i] += 0.5 * dt * (k1[i] + k2[i]);
            if self.concs[i] < 0.0 {
                self.concs[i] = 0.0;
            }
        }
    }
    pub fn conc(&self, k: usize) -> f64 {
        self.concs[k]
    }
}
