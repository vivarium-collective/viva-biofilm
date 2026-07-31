use crate::agent::Agent;

pub fn detach_above_height(agents: &mut Vec<Agent>, max_height: f64) -> usize {
    let before = agents.len();
    agents.retain(|a| a.y <= max_height);
    before - agents.len()
}
