use crate::agent::Agent;

fn dx_cyclic(a: f64, b: f64, span: f64) -> f64 {
    let mut d = a - b;
    if d > span / 2.0 {
        d -= span;
    } else if d < -span / 2.0 {
        d += span;
    }
    d
}

pub fn relax(agents: &mut Vec<Agent>, density: f64, domain_x: f64, iters: usize, k: f64) {
    let n = agents.len();
    for _ in 0..iters {
        for i in 0..n {
            for j in (i + 1)..n {
                let ri = agents[i].radius(density);
                let rj = agents[j].radius(density);
                let min_d = ri + rj;
                let dx = dx_cyclic(agents[i].x, agents[j].x, domain_x);
                let dy = agents[i].y - agents[j].y;
                let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
                let overlap = min_d - dist;
                if overlap > 0.0 {
                    let push = 0.5 * k * overlap;
                    let ux = dx / dist;
                    let uy = dy / dist;
                    agents[i].x += push * ux;
                    agents[i].y += push * uy;
                    agents[j].x -= push * ux;
                    agents[j].y -= push * uy;
                }
            }
        }
        for a in agents.iter_mut() {
            // wrap X
            a.x = a.x.rem_euclid(domain_x);
            // substratum constraint
            let r = a.radius(density);
            if a.y < r {
                a.y = r;
            }
        }
    }
}
