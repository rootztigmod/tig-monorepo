mod solver_1000_5;
mod solver_1000_10;
mod solver_1000_25;
mod solver_5000_10;
mod solver_5000_25;

use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

pub struct Solver;

impl Solver {
    pub fn solve(
        challenge: &Challenge,
        _save_solution: Option<&dyn Fn(&Solution) -> Result<()>>,
        hyperparameters: &Option<Map<String, Value>>,
    ) -> Result<Option<Solution>> {
        let solution = if challenge.num_items <= 1500 {
            let total_w: u32 = challenge.weights.iter().sum();
            let budget_pct = if total_w > 0 {
                (challenge.max_weight as f64 / total_w as f64 * 100.0) as u32
            } else {
                10
            };
            if budget_pct <= 7 {
                solver_1000_5::solve(challenge, hyperparameters)
            } else if budget_pct <= 17 {
                solver_1000_10::solve(challenge, hyperparameters)
            } else {
                solver_1000_25::solve(challenge, hyperparameters)
            }
        } else {
            let total_w: u32 = challenge.weights.iter().sum();
            let budget_pct = if total_w > 0 {
                (challenge.max_weight as f64 / total_w as f64 * 100.0) as u32
            } else {
                10
            };
            if budget_pct <= 17 {
                solver_5000_10::solve(challenge, hyperparameters)
            } else {
                solver_5000_25::solve(challenge, hyperparameters)
            }
        };
        Ok(Some(solution))
    }
}

#[allow(dead_code)]
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    if let Some(solution) = Solver::solve(challenge, Some(save_solution), hyperparameters)? {
        let _ = save_solution(&solution);
    }
    Ok(())
}

pub fn help() {
    println!("Quadratic Knapsack Problem - Per-track solvers for independent evolution");
    println!("Tracks: n_items=1000 budget=5/10/25, n_items=5000 budget=10/25");
}
