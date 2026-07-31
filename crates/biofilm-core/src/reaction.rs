pub struct Monod {
    pub mu_max: f64,
    pub terms: Vec<(usize, f64)>, // (solute_index, Ks)
}

impl Monod {
    pub fn specific_rate(&self, concs: &[f64]) -> f64 {
        let mut frac = 1.0;
        for &(k, ks) in &self.terms {
            let s = concs[k].max(0.0);
            frac *= s / (s + ks);
        }
        self.mu_max * frac
    }
}

pub struct Reaction {
    pub kinetics: Monod,
    pub yield_per_solute: Vec<(usize, f64)>, // (solute_index, coeff per unit biomass)
}

impl Reaction {
    pub fn biomass_rate(&self, mass: f64, concs: &[f64]) -> f64 {
        mass * self.kinetics.specific_rate(concs)
    }
}
