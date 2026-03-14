use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

const N_IT_CONSTRUCT: usize = 2;
const DIFF_LIM: usize = 9;
const MICRO_K: usize = 16;
const MICRO_RM_K: usize = 8;
const MICRO_ADD_K: usize = 8;

struct Params {
    effort: usize,
    stall_limit: usize,
    perturbation_strength: Option<usize>,
    perturbation_rounds: Option<usize>,
}

impl Params {
    fn initialize(h: &Option<Map<String, Value>>) -> Self {
        let mut p = Self {
            effort: 1,
            stall_limit: 6,
            perturbation_strength: None,
            perturbation_rounds: None,
        };
        if let Some(m) = h {
            if let Some(v) = m.get("effort").and_then(|v| v.as_u64()) {
                p.effort = (v as usize).clamp(1, 6);
            }
            if let Some(v) = m.get("stall_limit").and_then(|v| v.as_u64()) {
                p.stall_limit = (v as usize).clamp(1, 20);
            }
            if let Some(v) = m.get("perturbation_strength").and_then(|v| v.as_u64()) {
                p.perturbation_strength = Some((v as usize).clamp(1, 20));
            }
            if let Some(v) = m.get("perturbation_rounds").and_then(|v| v.as_u64()) {
                p.perturbation_rounds = Some((v as usize).clamp(1, 100));
            }
        }
        p
    }

    fn n_perturbation_rounds(&self) -> usize {
        if let Some(v) = self.perturbation_rounds { return v; }
        15
    }

    fn perturbation_strength_base(&self) -> usize {
        if let Some(v) = self.perturbation_strength { return v; }
        3
    }

    fn vnd_max_iterations(&self) -> usize {
        180
    }

    fn n_starts(&self, hard: bool) -> usize {
        if hard { 4 } else { 3 }
    }

    fn stall_limit_effective(&self) -> usize {
        self.stall_limit
    }
}

#[derive(Clone, Copy)]
struct Rng {
    state: u64,
}

impl Rng {
    fn from_seed(seed: &[u8; 32]) -> Self {
        let mut s: u64 = 0x9E3779B97F4A7C15;
        for (i, &b) in seed.iter().enumerate() {
            s ^= (b as u64) << ((i & 7) * 8);
            s = s.rotate_left(7).wrapping_mul(0xBF58476D1CE4E5B9);
        }
        if s == 0 { s = 1; }
        Self { state: s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

struct State<'a> {
    ch: &'a Challenge,
    selected_bit: Vec<bool>,
    contrib: Vec<i32>,
    support: Vec<u16>,
    total_interactions: &'a [i64],
    hubs_static: &'a [usize],
    neigh: &'a Vec<Vec<(u16, i16)>>,
    total_value: i64,
    total_weight: u32,
    window_locked: Vec<usize>,
    window_core: Vec<usize>,
    window_rejected: Vec<usize>,
    core_bins: Vec<(u32, Vec<usize>)>,
    usage: Vec<u16>,
    dp_cache: Vec<i64>,
    choose_cache: Vec<u8>,
    snap_bits: Vec<bool>,
    snap_contrib: Vec<i32>,
    snap_support: Vec<u16>,
}

impl<'a> State<'a> {
    fn new_empty(
        ch: &'a Challenge,
        total_interactions: &'a [i64],
        hubs_static: &'a [usize],
        neigh: &'a Vec<Vec<(u16, i16)>>,
    ) -> Self {
        let n = ch.num_items;
        let mut contrib = vec![0i32; n];
        for i in 0..n { contrib[i] = ch.values[i] as i32; }
        Self {
            ch,
            selected_bit: vec![false; n],
            contrib,
            support: vec![0u16; n],
            total_interactions,
            hubs_static,
            neigh,
            total_value: 0,
            total_weight: 0,
            window_locked: Vec::new(),
            window_core: Vec::new(),
            window_rejected: Vec::new(),
            core_bins: Vec::new(),
            usage: vec![0u16; n],
            dp_cache: Vec::new(),
            choose_cache: Vec::new(),
            snap_bits: vec![false; n],
            snap_contrib: vec![0i32; n],
            snap_support: vec![0u16; n],
        }
    }

    fn selected_items(&self) -> Vec<usize> {
        (0..self.ch.num_items).filter(|&i| self.selected_bit[i]).collect()
    }

    #[inline(always)]
    fn slack(&self) -> u32 {
        self.ch.max_weight - self.total_weight
    }

    #[inline(always)]
    fn add_item(&mut self, i: usize) {
        self.total_value += self.contrib[i] as i64;
        self.total_weight += self.ch.weights[i];
        let contrib_ptr = self.contrib.as_mut_ptr();
        let sup_ptr = self.support.as_mut_ptr();
        let row = unsafe { self.neigh.get_unchecked(i) };
        for &(k, v) in row.iter() {
            unsafe {
                let kk = k as usize;
                let ck = contrib_ptr.add(kk);
                *ck = (*ck).wrapping_add(v as i32);
                let sk = sup_ptr.add(kk);
                *sk = (*sk).saturating_add(1);
            }
        }
        self.selected_bit[i] = true;
    }

    #[inline(always)]
    fn remove_item(&mut self, j: usize) {
        self.total_value -= self.contrib[j] as i64;
        self.total_weight -= self.ch.weights[j];
        let contrib_ptr = self.contrib.as_mut_ptr();
        let sup_ptr = self.support.as_mut_ptr();
        let row = unsafe { self.neigh.get_unchecked(j) };
        for &(k, v) in row.iter() {
            unsafe {
                let kk = k as usize;
                let ck = contrib_ptr.add(kk);
                *ck = (*ck).wrapping_sub(v as i32);
                let sk = sup_ptr.add(kk);
                *sk = (*sk).saturating_sub(1);
            }
        }
        self.selected_bit[j] = false;
    }

    #[inline(always)]
    fn replace_item(&mut self, rm: usize, cand: usize) {
        self.remove_item(rm);
        self.add_item(cand);
    }

    fn restore_snapshot(&mut self, snap_value: i64, snap_weight: u32) {
        self.selected_bit.clone_from(&self.snap_bits);
        self.contrib.clone_from(&self.snap_contrib);
        self.support.clone_from(&self.snap_support);
        self.total_value = snap_value;
        self.total_weight = snap_weight;
    }
}

fn build_sparse_neighbors_and_totals(ch: &Challenge) -> (Vec<Vec<(u16, i16)>>, Vec<i64>) {
    let n = ch.num_items;
    let mut neigh: Vec<Vec<(u16, i16)>> = (0..n).map(|_| Vec::with_capacity(12)).collect();
    let mut totals: Vec<i64> = vec![0i64; n];
    for i in 0..n {
        let row_ptr = unsafe { ch.interaction_values.get_unchecked(i).as_ptr() };
        for j in 0..i {
            let val = unsafe { *row_ptr.add(j) };
            if val != 0 {
                let v16 = val as i16;
                neigh[i].push((j as u16, v16));
                neigh[j].push((i as u16, v16));
                let vv = val as i64;
                totals[i] += vv;
                totals[j] += vv;
            }
        }
    }
    for row in neigh.iter_mut() {
        row.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    }
    (neigh, totals)
}

fn core_half_for(ch: &Challenge) -> usize {
    let team_est = (ch.max_weight as usize) / 6;
    let by_budget = if team_est <= 140 {
        70
    } else if team_est >= 1100 {
        130
    } else if team_est >= 900 {
        120
    } else if team_est >= 450 {
        90
    } else if team_est >= 200 {
        75
    } else {
        0
    };
    if by_budget != 0 { by_budget.min(150) } else { 40 }
}

fn set_windows_from_density(
    state: &mut State,
    by_density: &[usize],
    idx_first_rejected: usize,
    idx_last_inserted: usize,
) {
    use std::collections::BTreeMap;
    let n = state.ch.num_items;
    let core_half = core_half_for(state.ch);
    let mut left = idx_first_rejected.saturating_sub(core_half + 1);
    let right = (idx_last_inserted + core_half + 1).min(n);
    if left > right { left = right; }
    state.window_locked = by_density[..left].to_vec();
    state.window_core = by_density[left..right].to_vec();
    state.window_rejected = by_density[right..].to_vec();
    let mut bins: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &i in &state.window_core {
        bins.entry(state.ch.weights[i]).or_default().push(i);
    }
    state.core_bins = bins.into_iter().collect();
}

fn rebuild_windows(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    let cap = state.ch.max_weight;
    let mut by_density: Vec<usize> = (0..n).collect();
    let contrib = &state.contrib;
    by_density.sort_unstable_by(|&a, &b| {
        let na = contrib[a] as i64;
        let nb = contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (na * wb).cmp(&(nb * wa)).reverse()
    });
    let mut rem = cap;
    let mut idx_last_inserted = 0usize;
    let mut idx_first_rejected = n;
    for (idx, &i) in by_density.iter().enumerate() {
        let w = state.ch.weights[i];
        if w <= rem {
            rem -= w;
            idx_last_inserted = idx;
        } else if idx_first_rejected == n {
            idx_first_rejected = idx;
        }
    }
    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}

fn build_initial_solution(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    let cap = state.ch.max_weight;
    let mut sum_values: i64 = 0;
    let mut sum_w: u32 = 0;
    for i in 0..n {
        state.selected_bit[i] = true;
        let c = unsafe { state.neigh.get_unchecked(i).len().min(u16::MAX as usize) as u16 };
        state.support[i] = c;
        let ti = state.total_interactions[i].min(i32::MAX as i64) as i32;
        state.contrib[i] = state.ch.values[i] as i32 + ti;
        sum_values += state.ch.values[i] as i64;
        sum_w += state.ch.weights[i];
    }
    state.total_weight = sum_w;
    let sum_inter: i64 = state.total_interactions.iter().sum();
    state.total_value = sum_values + sum_inter / 2;
    while state.total_weight > cap {
        let mut worst_item = 0;
        let mut worst_score = i64::MAX;
        for i in 0..n {
            if state.selected_bit[i] {
                let contrib = state.contrib[i] as i64;
                let weight = state.ch.weights[i] as i64;
                let score = if weight > 0 { (contrib * 1000) / weight } else { contrib * 1000 };
                if score < worst_score { worst_score = score; worst_item = i; }
            }
        }
        state.remove_item(worst_item);
    }
    let mut idx_last_inserted = 0;
    let mut idx_first_rejected = n;
    let mut by_density: Vec<usize> = (0..n).collect();
    for _ in 0..=N_IT_CONSTRUCT {
        idx_last_inserted = 0;
        idx_first_rejected = n;
        let contrib = &state.contrib;
        by_density.sort_unstable_by(|&a, &b| {
            let na = contrib[a] as i64;
            let nb = contrib[b] as i64;
            let wa = state.ch.weights[a] as i64;
            let wb = state.ch.weights[b] as i64;
            (na * wb).cmp(&(nb * wa)).reverse()
        });
        let mut target_sel: Vec<usize> = Vec::with_capacity(n);
        let mut rem = cap;
        for (idx, &i) in by_density.iter().enumerate() {
            let w = state.ch.weights[i];
            if w <= rem {
                target_sel.push(i);
                rem -= w;
                idx_last_inserted = idx;
            } else if idx_first_rejected == n {
                idx_first_rejected = idx;
            }
        }
        let mut in_target = vec![false; n];
        for &i in &target_sel { in_target[i] = true; }
        let mut to_remove: Vec<usize> = Vec::new();
        let mut to_add: Vec<usize> = Vec::new();
        for i in 0..n {
            if state.selected_bit[i] && !in_target[i] { to_remove.push(i); }
            if !state.selected_bit[i] && in_target[i] { to_add.push(i); }
        }
        if to_remove.is_empty() && to_add.is_empty() { break; }
        for &r in &to_remove { state.remove_item(r); }
        for &a in &to_add { state.add_item(a); }
    }
    set_windows_from_density(state, &by_density, idx_first_rejected, idx_last_inserted);
}

fn greedy_fill_with_beta(state: &mut State, rng: &mut Rng, noise_mask: u32, allow_seed: bool) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const SUP_BONUS: i64 = 70;
    const HUB_GLOBAL_K: usize = 64;
    const HUB_PAIR_K: usize = 12;
    let n = state.ch.num_items;
    let neigh = state.neigh;
    let mut hubs_g: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_GLOBAL_K);
    let lim = state.hubs_static.len().min(256);
    for &i in state.hubs_static.iter().take(lim) {
        if state.selected_bit[i] { continue; }
        let w_u = state.ch.weights[i];
        if w_u == 0 { continue; }
        let w = w_u as i64;
        let tot_i = state.total_interactions[i];
        let mut ss = (tot_i * 1000) / w.max(1);
        if noise_mask != 0 { ss += (rng.next_u32() & (noise_mask >> 1)) as i64; }
        hubs_g.push((ss, i, w_u));
    }
    hubs_g.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    if hubs_g.len() > HUB_GLOBAL_K { hubs_g.truncate(HUB_GLOBAL_K); }
    let mut in_frontier = vec![false; n];
    let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));
    for i in 0..n {
        if !state.selected_bit[i] { continue; }
        let row = unsafe { neigh.get_unchecked(i) };
        for &(k, _v) in row.iter() {
            let u = k as usize;
            if !state.selected_bit[u] && !in_frontier[u] { in_frontier[u] = true; frontier.push(u); }
        }
    }
    let push_frontier_of = |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, v: usize| {
        let row = unsafe { neigh.get_unchecked(v) };
        for &(k, _vv) in row.iter() {
            let u = k as usize;
            if !st.selected_bit[u] && !in_f[u] { in_f[u] = true; fr.push(u); }
        }
    };
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_pos: Option<(usize, i64)> = None;
        let f_len = frontier.len();
        let scan_lim = if f_len > 2048 { 2048 } else { f_len };
        if f_len != 0 && f_len <= 2048 {
            for &i in &frontier {
                if state.selected_bit[i] { continue; }
                let w_u = state.ch.weights[i];
                if w_u == 0 || w_u > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let tot_i = state.total_interactions[i];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);
                let mut s = (adj * 1000) / (w_u as i64).max(1) + (state.support[i] as i64) * SUP_BONUS;
                if noise_mask != 0 { s += (rng.next_u32() & noise_mask) as i64; }
                if best_pos.map_or(true, |(_, bs)| s > bs) { best_pos = Some((i, s)); }
            }
        } else if f_len > 2048 {
            for _ in 0..scan_lim {
                let idx = (rng.next_u32() as usize) % f_len;
                let i = frontier[idx];
                if state.selected_bit[i] { continue; }
                let w_u = state.ch.weights[i];
                if w_u == 0 || w_u > slack { continue; }
                let c = state.contrib[i] as i64;
                if c <= 0 { continue; }
                let tot_i = state.total_interactions[i];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);
                let mut s = (adj * 1000) / (w_u as i64).max(1) + (state.support[i] as i64) * SUP_BONUS;
                if noise_mask != 0 { s += (rng.next_u32() & noise_mask) as i64; }
                if best_pos.map_or(true, |(_, bs)| s > bs) { best_pos = Some((i, s)); }
            }
        }
        if let Some((i, _)) = best_pos {
            state.add_item(i);
            push_frontier_of(state, &mut in_frontier, &mut frontier, i);
            continue;
        }
        let mut best_pair: Option<(usize, usize, i64)> = None;
        if slack >= 2 && !hubs_g.is_empty() {
            let lim2 = hubs_g.len().min(HUB_PAIR_K);
            for t in 0..lim2 {
                let a = hubs_g[t].1;
                let wa = hubs_g[t].2;
                if state.selected_bit[a] || wa == 0 || wa >= slack { continue; }
                let row = unsafe { neigh.get_unchecked(a) };
                let pref = row.len().min(64);
                for u in 0..pref {
                    let (bb, vv) = row[u];
                    let b = bb as usize;
                    if a == b || state.selected_bit[b] { continue; }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > slack { continue; }
                    let v = vv as i64;
                    if v <= 0 { continue; }
                    let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                    if delta <= 0 { continue; }
                    let s = (delta * 1_000_000) / ((wa + wb) as i64).max(1);
                    if best_pair.map_or(true, |(_, _, bs)| s > bs) { best_pair = Some((a, b, s)); }
                }
            }
        }
        if let Some((a, b, _)) = best_pair {
            state.add_item(a);
            push_frontier_of(state, &mut in_frontier, &mut frontier, a);
            if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] {
                state.add_item(b);
                push_frontier_of(state, &mut in_frontier, &mut frontier, b);
            }
            continue;
        }
        if allow_seed {
            let mut best_seed: Option<(usize, i64)> = None;
            for &(ss, i, _w) in &hubs_g {
                if state.selected_bit[i] { continue; }
                let wi = state.ch.weights[i];
                if wi == 0 || wi > slack { continue; }
                if best_seed.map_or(true, |(_, bs)| ss > bs) { best_seed = Some((i, ss)); }
            }
            if let Some((i, _)) = best_seed {
                state.add_item(i);
                push_frontier_of(state, &mut in_frontier, &mut frontier, i);
                continue;
            }
        }
        break;
    }
}

fn construct_pair_seed_beta(state: &mut State, rng: &mut Rng) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    let neigh = state.neigh;
    if state.total_weight == 0 {
        let cap = state.ch.max_weight;
        let mut best_pair: Option<(usize, usize, i64)> = None;
        for i in 0..n {
            let wi = state.ch.weights[i];
            if wi == 0 || wi > cap { continue; }
            let row = unsafe { neigh.get_unchecked(i) };
            for &(jj, vv) in row.iter() {
                let j = jj as usize;
                if j >= i { continue; }
                let w = wi + state.ch.weights[j];
                if w == 0 || w > cap { continue; }
                let v = vv as i64;
                if v <= 0 { continue; }
                let ti = state.total_interactions[i];
                let tj = state.total_interactions[j];
                let noise: i64 = (((i as i64) * 1315423911i64) ^ ((j as i64) * 2654435761i64)) & 0x3Fi64;
                let s = (v * 1_000_000) / (w as i64) + (ti + tj) / 2000 + noise;
                if best_pair.map_or(true, |(_, _, bs)| s > bs) { best_pair = Some((i, j, s)); }
            }
        }
        if let Some((i, j, _)) = best_pair {
            state.add_item(i);
            if state.total_weight + state.ch.weights[j] <= cap && !state.selected_bit[j] {
                state.add_item(j);
            }
        }
    }
    greedy_fill_with_beta(state, rng, 0, true);
}

fn construct_frontier_cluster_grow(state: &mut State, rng: &mut Rng) {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    let n = state.ch.num_items;
    if n == 0 { return; }
    let cap = state.ch.max_weight;
    let neigh = state.neigh;
    let mut in_frontier = vec![false; n];
    let mut frontier: Vec<usize> = Vec::with_capacity(n.min(4096));
    let push_frontier_of = |st: &State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, i: usize| {
        let row = unsafe { neigh.get_unchecked(i) };
        for &(k, _v) in row.iter() {
            let u = k as usize;
            if !st.selected_bit[u] && !in_f[u] { in_f[u] = true; fr.push(u); }
        }
    };
    let add_seed = |st: &mut State, in_f: &mut Vec<bool>, fr: &mut Vec<usize>, seed: usize| {
        if st.selected_bit[seed] { return; }
        if st.total_weight + st.ch.weights[seed] > cap { return; }
        st.add_item(seed);
        push_frontier_of(st, in_f, fr, seed);
    };
    let mut best_seed: Option<usize> = None;
    let mut best_s: i64 = i64::MIN;
    let samples = n.min(512).max(64);
    for _ in 0..samples {
        let i = (rng.next_u32() as usize) % n;
        let w = state.ch.weights[i] as i64;
        if w <= 0 || w as u32 > cap { continue; }
        let s = (state.total_interactions[i] * 1000) / w;
        if s > best_s { best_s = s; best_seed = Some(i); }
    }
    if let Some(i) = best_seed {
        add_seed(state, &mut in_frontier, &mut frontier, i);
        let mut best_j: Option<(usize, i64)> = None;
        let row = unsafe { neigh.get_unchecked(i) };
        for &(jj, vv) in row.iter() {
            let j = jj as usize;
            if j == i || state.selected_bit[j] { continue; }
            let wsum = state.ch.weights[i] + state.ch.weights[j];
            if wsum == 0 || wsum > cap { continue; }
            let v = vv as i64;
            if v <= 0 { continue; }
            let s = (v * 1_000_000) / (wsum as i64) + state.total_interactions[j] / 2000;
            if best_j.map_or(true, |(_, bs)| s > bs) { best_j = Some((j, s)); }
        }
        if let Some((j, _)) = best_j { add_seed(state, &mut in_frontier, &mut frontier, j); }
    }
    let team_est = (cap as usize) / 6;
    let max_jumps: usize = if team_est >= 900 { 10 } else if team_est >= 450 { 8 } else if team_est >= 200 { 6 } else { 3 };
    let mut jumps_done: usize = 0;
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_cand: Option<(usize, i64, i64)> = None;
        for &u in &frontier {
            if state.selected_bit[u] { continue; }
            let wu = state.ch.weights[u];
            if wu == 0 || wu > slack { continue; }
            let c = state.contrib[u] as i64;
            if c <= 0 { continue; }
            let tot_u = state.total_interactions[u];
            let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_u);
            let s0 = (adj * 1000) / (wu as i64).max(1);
            let s = s0 + (rng.next_u32() & 0x0F) as i64;
            if best_cand.map_or(true, |(_, bs, _)| s > bs) { best_cand = Some((u, s, s0)); }
        }
        let allow_early_jump = jumps_done < max_jumps && slack > cap / 5;
        let mut jump: Option<(usize, i64)> = None;
        let mut jump_s: i64 = i64::MIN;
        if allow_early_jump || best_cand.is_none() {
            for _ in 0..samples {
                let i = (rng.next_u32() as usize) % n;
                if state.selected_bit[i] { continue; }
                let wi = state.ch.weights[i];
                if wi == 0 || wi >= slack { continue; }
                let row = unsafe { neigh.get_unchecked(i) };
                let mut ok = false;
                let lim = row.len().min(10);
                for t in 0..lim {
                    let j = row[t].0 as usize;
                    if state.selected_bit[j] { continue; }
                    let wj = state.ch.weights[j];
                    if wj != 0 && wi + wj <= slack { ok = true; break; }
                }
                if !ok { continue; }
                let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
                if s > jump_s { jump_s = s; jump = Some((i, s)); }
            }
        }
        let do_jump = if let (Some((_u, _s, s0)), Some((_j, js))) = (best_cand, jump) {
            allow_early_jump && js > s0.saturating_mul(2)
        } else {
            best_cand.is_none() && jump.is_some()
        };
        if do_jump {
            if let Some((seed, _)) = jump {
                jumps_done += 1;
                add_seed(state, &mut in_frontier, &mut frontier, seed);
                let slack1 = state.slack();
                if state.selected_bit[seed] && slack1 >= 1 {
                    let mut best_nb: Option<(usize, i64)> = None;
                    let row = unsafe { neigh.get_unchecked(seed) };
                    let pref = row.len().min(72);
                    for t in 0..pref {
                        let (jj, vv) = row[t];
                        let j = jj as usize;
                        if j == seed || state.selected_bit[j] { continue; }
                        let wj = state.ch.weights[j];
                        if wj == 0 || wj > slack1 { continue; }
                        let v = vv as i64;
                        if v <= 0 { continue; }
                        let wsum = (state.ch.weights[seed] + wj) as i64;
                        let s = (v * 1_000_000) / wsum.max(1) + state.total_interactions[j] / 2000;
                        if best_nb.map_or(true, |(_, bs)| s > bs) { best_nb = Some((j, s)); }
                    }
                    if let Some((j, _)) = best_nb { add_seed(state, &mut in_frontier, &mut frontier, j); }
                }
                continue;
            }
        }
        if let Some((u, _s, _s0)) = best_cand {
            state.add_item(u);
            push_frontier_of(state, &mut in_frontier, &mut frontier, u);
            continue;
        }
        if let Some((seed, _)) = jump {
            add_seed(state, &mut in_frontier, &mut frontier, seed);
            continue;
        }
        break;
    }
    greedy_fill_with_beta(state, rng, 0, true);
}

fn integer_core_target(
    ch: &Challenge,
    locked: &[usize],
    core: &[usize],
    core_val: &[i32],
    dp_cache: &mut Vec<i64>,
    choose_cache: &mut Vec<u8>,
) -> Vec<usize> {
    let used_locked: u64 = locked.iter().map(|&i| ch.weights[i] as u64).sum();
    let rem_cap = (ch.max_weight as u64).saturating_sub(used_locked) as usize;
    let myk = core.len();
    if myk == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        selected.sort_unstable();
        return selected;
    }
    let mut total_core_weight: usize = 0;
    let mut total_pos_weight: usize = 0;
    let mut all_pos_fit = true;
    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        total_core_weight += wt;
        if core_val[t] > 0 {
            total_pos_weight += wt;
            if total_pos_weight > rem_cap { all_pos_fit = false; }
        }
    }
    if rem_cap == 0 {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if ch.weights[it] == 0 && core_val[t] > 0 { selected.push(it); }
        }
        selected.sort_unstable();
        return selected;
    }
    if all_pos_fit {
        let mut selected: Vec<usize> = locked.to_vec();
        for (t, &it) in core.iter().enumerate() {
            if core_val[t] > 0 { selected.push(it); }
        }
        selected.sort_unstable();
        return selected;
    }
    let myw = rem_cap.min(total_core_weight);
    let dp_size = myw + 1;
    let choose_size = myk * dp_size;
    if dp_cache.len() < dp_size { dp_cache.resize(dp_size, i64::MIN / 4); }
    if choose_cache.len() < choose_size { choose_cache.resize(choose_size, 0); }
    for val in &mut dp_cache[0..dp_size] { *val = -1; }
    dp_cache[0] = 0;
    choose_cache[0..choose_size].fill(0);
    let mut w_hi: usize = 0;
    for (t, &it) in core.iter().enumerate() {
        let wt = ch.weights[it] as usize;
        if wt > myw { continue; }
        let val = core_val[t] as i64;
        let new_hi = (w_hi + wt).min(myw);
        for w in (wt..=new_hi).rev() {
            let prev = dp_cache[w - wt];
            if prev < 0 { continue; }
            let cand = prev + val;
            if cand > dp_cache[w] { dp_cache[w] = cand; choose_cache[t * dp_size + w] = 1; }
        }
        w_hi = new_hi;
    }
    let mut selected: Vec<usize> = locked.to_vec();
    let mut w_star = (0..=myw).max_by_key(|&w| dp_cache[w]).unwrap_or(0);
    for t in (0..myk).rev() {
        let it = core[t];
        let wt = ch.weights[it] as usize;
        if wt <= w_star && choose_cache[t * dp_size + w_star] == 1 {
            selected.push(it);
            w_star -= wt;
        }
    }
    selected.sort_unstable();
    selected
}

fn apply_dp_target_via_ops(state: &mut State, target_sel: &[usize]) {
    let n = state.ch.num_items;
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    let mut j = 0;
    let m = target_sel.len();
    for i in 0..n {
        let in_target = j < m && target_sel[j] == i;
        if in_target { j += 1; }
        if state.selected_bit[i] && !in_target { to_remove.push(i); }
        else if in_target && !state.selected_bit[i] { to_add.push(i); }
    }
    for &r in &to_remove { state.remove_item(r); }
    for &a in &to_add { state.add_item(a); }
}

fn dp_refinement(state: &mut State) {
    let passes = if state.window_core.len() <= 160 { 2 } else { 1 };
    let n = state.ch.num_items;
    let neigh = state.neigh;
    for _ in 0..passes {
        let mut core_val: Vec<i32> = Vec::with_capacity(state.window_core.len());
        if !state.window_core.is_empty() {
            let mut sel_core_bit = vec![false; n];
            for &i in &state.window_core {
                if state.selected_bit[i] { sel_core_bit[i] = true; }
            }
            for &it in &state.window_core {
                let mut sub: i32 = 0;
                let row = unsafe { neigh.get_unchecked(it) };
                for &(k, v) in row.iter() {
                    let j = k as usize;
                    if sel_core_bit[j] { sub += v as i32; }
                }
                let mut v0 = state.contrib[it] - (sub / 2);
                if !state.selected_bit[it] {
                    v0 += (state.total_interactions[it] / 320) as i32 + (state.usage[it] as i32) * 12;
                }
                core_val.push(v0);
            }
        }
        let target = integer_core_target(
            state.ch,
            &state.window_locked,
            &state.window_core,
            &core_val,
            &mut state.dp_cache,
            &mut state.choose_cache,
        );
        apply_dp_target_via_ops(state, &target);
    }
}

fn micro_qkp_refinement(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 || state.window_core.is_empty() { return; }
    let team_est = (state.ch.max_weight as usize) / 6;
    let big_team = team_est >= 850;
    let micro_k: usize = if big_team { 12 } else { MICRO_K };
    let rm_k: usize = if big_team { 6 } else { MICRO_RM_K };
    let add_k: usize = if big_team { 6 } else { MICRO_ADD_K };
    let neigh = state.neigh;
    let mut sel: Vec<usize> = Vec::new();
    let mut unsel: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if state.selected_bit[i] { sel.push(i); } else { unsel.push(i); }
    }
    {
        let mut guides: Vec<usize> = Vec::new();
        for &i in &state.window_core {
            if state.selected_bit[i] { guides.push(i); }
        }
        guides.sort_unstable_by(|&a, &b| {
            state.support[b].cmp(&state.support[a])
                .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
                .then_with(|| b.cmp(&a))
        });
        let push_unsel = |v: &mut Vec<usize>, x: usize| {
            for &y in v.iter() { if y == x { return; } }
            v.push(x);
        };
        let g = guides.len().min(if big_team { 4 } else { 6 });
        for t in 0..g {
            let vtx = guides[t];
            let row = unsafe { neigh.get_unchecked(vtx) };
            let pref = row.len().min(if big_team { 16 } else { 24 });
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] { push_unsel(&mut unsel, cand); }
            }
        }
        let hub_take: usize = if big_team { 5 } else { 8 };
        let mut added_hubs: usize = 0;
        let lim = state.hubs_static.len().min(192);
        for &h in state.hubs_static.iter().take(lim) {
            if added_hubs >= hub_take { break; }
            if state.selected_bit[h] { continue; }
            push_unsel(&mut unsel, h);
            added_hubs += 1;
            let row = unsafe { neigh.get_unchecked(h) };
            let pref = row.len().min(16);
            for u in 0..pref {
                let cand = row[u].0 as usize;
                if !state.selected_bit[cand] { push_unsel(&mut unsel, cand); }
            }
        }
    }
    let extra_r = state.window_rejected.len().min(if big_team { 12 } else { 24 });
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] { unsel.push(i); }
    }
    let extra_l = state.window_locked.len().min(24);
    let start_l = state.window_locked.len().saturating_sub(extra_l);
    for &i in &state.window_locked[start_l..] {
        if state.selected_bit[i] { sel.push(i); }
    }
    let score = |st: &State, i: usize| -> i64 {
        let w = (st.ch.weights[i] as i64).max(1);
        let dens = (st.contrib[i] as i64 * 1000) / w;
        dens + (st.support[i] as i64) * 120 + (st.total_interactions[i] / 320)
    };
    sel.sort_unstable_by(|&a, &b| score(state, a).cmp(&score(state, b)).then_with(|| a.cmp(&b)));
    unsel.sort_unstable_by(|&a, &b| score(state, b).cmp(&score(state, a)).then_with(|| b.cmp(&a)));
    let mut cand: Vec<usize> = Vec::with_capacity(micro_k);
    let push_u = |v: &mut Vec<usize>, x: usize| {
        for &y in v.iter() { if y == x { return; } }
        v.push(x);
    };
    for &i in sel.iter().take(rm_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k { break; }
    }
    for &i in unsel.iter().take(add_k) {
        push_u(&mut cand, i);
        if cand.len() >= micro_k { break; }
    }
    if cand.len() < 2 { return; }
    let k = cand.len();
    if k > 20 { return; }
    let mut sel_cand: Vec<usize> = Vec::new();
    let mut sel_cand_w: u32 = 0;
    for &it in &cand {
        if state.selected_bit[it] {
            sel_cand.push(it);
            sel_cand_w = sel_cand_w.saturating_add(state.ch.weights[it]);
        }
    }
    if state.total_weight < sel_cand_w { return; }
    let fixed_w = state.total_weight - sel_cand_w;
    if fixed_w > state.ch.max_weight { return; }
    let rem_cap: u32 = state.ch.max_weight - fixed_w;
    let mut w: Vec<u32> = vec![0; k];
    let mut base: Vec<i64> = vec![0; k];
    for t in 0..k {
        let it = cand[t];
        w[t] = state.ch.weights[it];
        let mut b = state.contrib[it] as i64;
        for &j in &sel_cand { b -= state.ch.interaction_values[it][j] as i64; }
        base[t] = b;
    }
    let mut inter: Vec<i64> = vec![0; k * k];
    for a in 0..k {
        let ia = cand[a];
        for b in 0..a {
            let ib = cand[b];
            let v = state.ch.interaction_values[ia][ib] as i64;
            inter[a * k + b] = v;
            inter[b * k + a] = v;
        }
    }
    let mut cur_mask: usize = 0;
    for t in 0..k {
        if state.selected_bit[cand[t]] { cur_mask |= 1usize << t; }
    }
    let mut cur_w_sum: u32 = 0;
    let mut cur_v: i64 = 0;
    for i in 0..k {
        if ((cur_mask >> i) & 1) == 0 { continue; }
        cur_w_sum += w[i];
        cur_v += base[i];
        for j in 0..i {
            if ((cur_mask >> j) & 1) != 0 { cur_v += inter[i * k + j]; }
        }
    }
    if cur_w_sum > rem_cap { return; }
    let mmax: usize = 1usize << k;
    let mut wmask: Vec<u32> = vec![0; mmax];
    let mut vmask: Vec<i64> = vec![0; mmax];
    let mut best_mask: usize = cur_mask;
    let mut best_val: i64 = cur_v;
    for mask in 1..mmax {
        let lsb = mask & mask.wrapping_neg();
        let bi = lsb.trailing_zeros() as usize;
        let prev = mask ^ lsb;
        let ww = wmask[prev].saturating_add(w[bi]);
        if ww > rem_cap { wmask[mask] = u32::MAX; vmask[mask] = i64::MIN / 4; continue; }
        let mut val = vmask[prev] + base[bi];
        let mut pm = prev;
        while pm != 0 {
            let l = pm & pm.wrapping_neg();
            let j = l.trailing_zeros() as usize;
            val += inter[bi * k + j];
            pm ^= l;
        }
        wmask[mask] = ww;
        vmask[mask] = val;
        if val > best_val { best_val = val; best_mask = mask; }
    }
    if best_mask == cur_mask { return; }
    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    for t in 0..k {
        let it = cand[t];
        let want = ((best_mask >> t) & 1) != 0;
        let have = state.selected_bit[it];
        if have && !want { to_remove.push(it); }
        else if !have && want { to_add.push(it); }
    }
    for &it in &to_remove { state.remove_item(it); }
    for &it in &to_add {
        if state.total_weight + state.ch.weights[it] <= state.ch.max_weight { state.add_item(it); }
    }
}

fn apply_best_add_windowed(state: &mut State) -> bool {
    const SUP_ADD_BONUS: i64 = 40;
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, i64)> = None;
    for (bw, items) in &state.core_bins {
        if *bw > slack { break; }
        for &cand in items {
            if state.selected_bit[cand] { continue; }
            let delta = state.contrib[cand];
            if delta <= 0 { continue; }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((cand, s)); }
        }
    }
    if best.is_none() {
        let lim = state.window_rejected.len().min(384);
        for &cand in &state.window_rejected[..lim] {
            if state.selected_bit[cand] { continue; }
            let w = state.ch.weights[cand];
            if w > slack { continue; }
            let delta = state.contrib[cand];
            if delta <= 0 { continue; }
            let s = (delta as i64) + (state.support[cand] as i64) * SUP_ADD_BONUS;
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((cand, s)); }
        }
    }
    if let Some((cand, _)) = best { state.add_item(cand); true } else { false }
}

fn apply_best_add_neigh_global(state: &mut State) -> bool {
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const SUP_ADD_BONUS: i64 = 40;
    let slack = state.slack();
    if slack == 0 { return false; }
    let neigh = state.neigh;
    let n = state.ch.num_items;
    if n == 0 { return false; }
    let edge_lim: usize = 12000;
    let node_lim: usize = 64;
    let start = (((state.total_value as u64) as usize) ^ ((state.total_weight as usize).wrapping_mul(911))) % n;
    let mut step = (n / 97).max(1);
    step |= 1;
    let mut best: Option<(usize, i64, i32)> = None;
    let mut scanned_edges: usize = 0;
    let mut scanned_nodes: usize = 0;
    let mut idx = start;
    let mut tries: usize = 0;
    while tries < n && scanned_nodes < node_lim && scanned_edges < edge_lim {
        if state.selected_bit[idx] {
            scanned_nodes += 1;
            let row = unsafe { neigh.get_unchecked(idx) };
            for &(cj, _vv) in row.iter() {
                scanned_edges += 1;
                if scanned_edges > edge_lim { break; }
                let cand = cj as usize;
                if state.selected_bit[cand] { continue; }
                let w_u = state.ch.weights[cand];
                if w_u == 0 || w_u > slack { continue; }
                let delta = state.contrib[cand];
                if delta <= 0 { continue; }
                let w = (w_u as i64).max(1);
                let c = delta as i64;
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / w + (state.support[cand] as i64) * SUP_ADD_BONUS;
                if best.map_or(true, |(_, bs, bd)| s > bs || (s == bs && delta > bd)) {
                    best = Some((cand, s, delta));
                }
            }
        }
        idx += step;
        if idx >= n { idx -= n; }
        tries += 1;
    }
    if let Some((cand, _, _)) = best { state.add_item(cand); true } else { false }
}

fn apply_best_replace12_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() { return false; }
    let slack0 = state.slack();
    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 64);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(64);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i]);
    add_pool.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 80;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 80;
        sb.cmp(&sa).then_with(|| b.cmp(&a))
    });
    if add_pool.len() > 28 { add_pool.truncate(28); }
    if add_pool.len() < 2 { return false; }
    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 120;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 120;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > 16 { rm.truncate(16); }
    let mut best: Option<(usize, usize, usize, i64)> = None;
    for &r in &rm {
        if !state.selected_bit[r] { continue; }
        let wr = state.ch.weights[r];
        let avail = wr.saturating_add(slack0);
        for x in 0..add_pool.len() {
            let a = add_pool[x];
            let wa = state.ch.weights[a];
            if wa == 0 || wa > avail { continue; }
            for y in (x + 1)..add_pool.len() {
                let b = add_pool[y];
                let wb = state.ch.weights[b];
                if wb == 0 || wa + wb > avail { continue; }
                let new_w = state.total_weight.saturating_sub(wr).saturating_add(wa).saturating_add(wb);
                if new_w > cap { continue; }
                let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64)
                    - (state.contrib[r] as i64)
                    - (state.ch.interaction_values[a][r] as i64)
                    - (state.ch.interaction_values[b][r] as i64)
                    + (state.ch.interaction_values[a][b] as i64);
                if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best = Some((r, a, b, delta));
                }
            }
        }
    }
    if let Some((r, a, b, _)) = best {
        state.remove_item(r); state.add_item(a); state.add_item(b); true
    } else { false }
}

fn apply_best_replace21_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 { return false; }
    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (ca * wb).cmp(&(cb * wa))
    });
    if rm.len() > 14 { rm.truncate(14); }
    let mut add_pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 48);
    add_pool.extend_from_slice(&state.window_core);
    let extra = state.window_rejected.len().min(48);
    add_pool.extend_from_slice(&state.window_rejected[..extra]);
    add_pool.retain(|&i| !state.selected_bit[i] && state.contrib[i] > 0);
    add_pool.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64;
        let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64;
        let wb = state.ch.weights[b] as i64;
        (cb * wa).cmp(&(ca * wb))
    });
    if add_pool.len() > 28 { add_pool.truncate(28); }
    if add_pool.is_empty() { return false; }
    let mut best: Option<(usize, usize, usize, i64)> = None;
    for &cand in &add_pool {
        let wc = state.ch.weights[cand];
        if wc == 0 { continue; }
        for x in 0..rm.len() {
            let a = rm[x];
            let wa = state.ch.weights[a];
            for y in (x + 1)..rm.len() {
                let b = rm[y];
                let wb = state.ch.weights[b];
                let new_w = state.total_weight.saturating_sub(wa).saturating_sub(wb).saturating_add(wc);
                if new_w > cap { continue; }
                let delta = (state.contrib[cand] as i64)
                    - (state.ch.interaction_values[cand][a] as i64)
                    - (state.ch.interaction_values[cand][b] as i64)
                    - (state.contrib[a] as i64)
                    - (state.contrib[b] as i64)
                    + (state.ch.interaction_values[a][b] as i64);
                if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
                    best = Some((cand, a, b, delta));
                }
            }
        }
    }
    if let Some((cand, a, b, _)) = best {
        state.remove_item(a);
        state.remove_item(b);
        if !state.selected_bit[cand] && state.total_weight + state.ch.weights[cand] <= cap {
            state.add_item(cand);
        }
        true
    } else { false }
}

#[inline]
fn apply_best_swap_diff_reduce_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let mut best: Option<(usize, usize, i32)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        if w_rm == 0 { continue; }
        let w_min = w_rm.saturating_sub(DIFF_LIM as u32);
        for (bw, items) in &state.core_bins {
            if *bw >= w_rm { break; }
            if *bw < w_min { continue; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm] - state.ch.interaction_values[cand][rm];
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) {
                    best = Some((cand, rm, delta));
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
}

#[inline]
fn apply_best_swap_diff_increase_windowed_cached(state: &mut State, used: &[usize]) -> bool {
    let slack = state.slack();
    if slack == 0 { return false; }
    let mut best: Option<(usize, usize, f64)> = None;
    for &rm in used {
        let w_rm = state.ch.weights[rm];
        let max_dw = (DIFF_LIM as u32).min(slack);
        let w_max = w_rm.saturating_add(max_dw);
        for (bw, items) in &state.core_bins {
            if *bw <= w_rm { continue; }
            if *bw > w_max { break; }
            let dw = *bw - w_rm;
            if dw > slack { break; }
            for &cand in items {
                if state.selected_bit[cand] { continue; }
                let delta = state.contrib[cand] - state.contrib[rm] - state.ch.interaction_values[cand][rm];
                if delta > 0 {
                    let ratio = (delta as f64) / (dw as f64);
                    if best.map_or(true, |(_, _, br)| ratio > br) { best = Some((cand, rm, ratio)); }
                }
            }
        }
    }
    if let Some((cand, rm, _)) = best { state.replace_item(rm, cand); true } else { false }
}

fn apply_best_swap_neigh_any(state: &mut State, used: &[usize]) -> bool {
    if used.is_empty() { return false; }
    let cap = state.ch.max_weight;
    let neigh = state.neigh;
    let mut best: Option<(usize, usize, i64, i64)> = None;
    for &rm in used {
        if !state.selected_bit[rm] { continue; }
        let wrm = state.ch.weights[rm];
        let row = unsafe { neigh.get_unchecked(rm) };
        for &(cj, vv) in row.iter() {
            let cand = cj as usize;
            if state.selected_bit[cand] { continue; }
            let wc = state.ch.weights[cand];
            if wc == 0 { continue; }
            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wrm as u64) { continue; }
            let delta = (state.contrib[cand] as i64) - (state.contrib[rm] as i64) - (vv as i64);
            if delta <= 0 { continue; }
            let score: i64 = if wc == wrm {
                delta * 1_000_000
            } else if wc < wrm {
                delta * 1000 + (wrm as i64 - wc as i64)
            } else {
                let dw = (wc - wrm) as i64;
                (delta * 1000) / dw.max(1)
            };
            if best.map_or(true, |(_, _, bs, bd)| score > bs || (score == bs && delta > bd)) {
                best = Some((cand, rm, score, delta));
            }
        }
    }
    if let Some((cand, rm, _, _)) = best { state.replace_item(rm, cand); true } else { false }
}

fn local_search_vnd(state: &mut State, params: &Params) {
    let mut iterations = 0;
    let max_iterations = params.vnd_max_iterations();
    let mut used: Vec<usize> = Vec::new();
    let mut micro_used = false;
    let max_frontier_swaps: usize = 0;
    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = 1;
    loop {
        iterations += 1;
        if iterations > max_iterations { break; }
        if apply_best_add_windowed(state) { continue; }
        if apply_best_add_neigh_global(state) { dirty_window = true; continue; }
        used.clear();
        for &i in &state.window_core {
            if state.selected_bit[i] { used.push(i); }
        }
        let extra = state.window_locked.len().min(24);
        let start = state.window_locked.len().saturating_sub(extra);
        for &i in state.window_locked[start..].iter() {
            if state.selected_bit[i] { used.push(i); }
        }
        if apply_best_swap_diff_reduce_windowed_cached(state, &used) { continue; }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) { continue; }
        if apply_best_swap_neigh_any(state, &used) { dirty_window = true; continue; }
        let _ = max_frontier_swaps;
        if apply_best_replace12_windowed(state, &used) { continue; }
        if apply_best_replace21_windowed(state, &used) { continue; }
        if dirty_window && n_rebuilds < max_rebuilds {
            n_rebuilds += 1;
            dirty_window = false;
            rebuild_windows(state);
            continue;
        }
        if !micro_used {
            micro_used = true;
            if dirty_window { rebuild_windows(state); dirty_window = false; }
            let old = state.total_value;
            micro_qkp_refinement(state);
            if state.total_value > old { rebuild_windows(state); continue; }
        }
        break;
    }
}

fn perturb_by_strategy(state: &mut State, rng: &mut Rng, strength: usize, stall_count: usize, strategy: usize) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }
    let cap = state.ch.max_weight;
    let neigh = state.neigh;
    let mut target: Option<(usize, usize, u32)> = None;
    if stall_count > 0 && (strategy == 0 || strategy == 6) {
        let lim = state.hubs_static.len().min(64);
        let extra = 48usize;
        let mut best_a: Option<(i64, usize, u32)> = None;
        for &a in state.hubs_static.iter().take(lim) {
            if state.selected_bit[a] { continue; }
            let wa = state.ch.weights[a];
            if wa == 0 || wa > cap { continue; }
            let mut s = (state.total_interactions[a] * 1000) / (wa as i64).max(1);
            s += (state.contrib[a] as i64) * 10;
            s += (rng.next_u32() & 0x3F) as i64;
            if best_a.map_or(true, |(bs, _, _)| s > bs) { best_a = Some((s, a, wa)); }
        }
        for _ in 0..extra {
            let a = (rng.next_u32() as usize) % state.ch.num_items;
            if state.selected_bit[a] { continue; }
            let wa = state.ch.weights[a];
            if wa == 0 || wa > cap { continue; }
            let mut s = (state.total_interactions[a] * 1000) / (wa as i64).max(1);
            s += (state.contrib[a] as i64) * 10;
            s += (rng.next_u32() & 0x3F) as i64;
            if best_a.map_or(true, |(bs, _, _)| s > bs) { best_a = Some((s, a, wa)); }
        }
        if let Some((_sa, a, wa)) = best_a {
            let row = unsafe { neigh.get_unchecked(a) };
            let pref = row.len().min(64);
            let mut best_b: Option<(i64, usize, u32)> = None;
            for t in 0..pref {
                let (bb, vv) = row[t];
                let b = bb as usize;
                if b == a || state.selected_bit[b] { continue; }
                let wb = state.ch.weights[b];
                if wb == 0 || wa + wb > cap { continue; }
                let v = vv as i64;
                if v <= 0 { continue; }
                let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64) + v;
                let mut s = (v * 1_000_000) / ((wa + wb) as i64).max(1);
                s += delta * 40;
                s += (rng.next_u32() & 0x1F) as i64;
                if best_b.map_or(true, |(bs, _, _)| s > bs) { best_b = Some((s, b, wb)); }
            }
            if let Some((_sb, b, wb)) = best_b { target = Some((a, b, wa + wb)); }
        }
    }
    let base_remove = (selected.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let strength_scaled = strength + (selected.len() / 40);
    let n_remove = (base_remove * adaptive_mult).min(strength_scaled).min(selected.len() / 3);
    let (ta, tb, tw) = if let Some((a, b, w)) = target {
        (a, b, w)
    } else {
        (usize::MAX, usize::MAX, 0u32)
    };
    let mut need_w: u32 = 0;
    if ta != usize::MAX {
        let slack = state.slack();
        if tw > slack { need_w = tw - slack; }
        need_w = need_w.saturating_add(((strength as u32) + (stall_count as u32)).min(10));
    }
    let mut removal_candidates: Vec<(i64, usize, u32)> = Vec::with_capacity(selected.len());
    for &i in &selected {
        let w = state.ch.weights[i];
        if w == 0 { continue; }
        let mut s: i64 = match strategy {
            0 => state.contrib[i] as i64,
            1 => -(w as i64),
            2 => (state.contrib[i] - state.ch.values[i] as i32) as i64,
            3 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1),
            4 => (state.contrib[i] as i64 * 1000) / (w as i64).max(1) + (state.support[i] as i64) * 200,
            5 => (state.support[i] as i64) * 500 - (w as i64) * 220 + (state.contrib[i] as i64) / 50,
            _ => (state.contrib[i] as i64) - (state.usage[i] as i64) * 50,
        };
        if ta != usize::MAX {
            let ia = state.ch.interaction_values[i][ta] as i64;
            let ib = state.ch.interaction_values[i][tb] as i64;
            s += (ia + ib) * 4 + (state.usage[i] as i64) * 80;
        } else if stall_count >= 3 {
            s += (state.usage[i] as i64) * 30;
        }
        removal_candidates.push((s, i, w));
    }
    removal_candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut freed: u32 = 0;
    let mut removed: usize = 0;
    for &(_s, i, w) in &removal_candidates {
        if removed >= n_remove && freed >= need_w { break; }
        if state.selected_bit[i] { state.remove_item(i); freed = freed.saturating_add(w); removed += 1; }
    }
    if ta != usize::MAX {
        let wa = state.ch.weights[ta];
        let wb = state.ch.weights[tb];
        if !state.selected_bit[ta] && state.total_weight + wa <= cap { state.add_item(ta); }
        if !state.selected_bit[tb] && state.total_weight + wb <= cap { state.add_item(tb); }
    }
}

fn greedy_reconstruct(state: &mut State, rng: &mut Rng, strategy: usize) {
    let n = state.ch.num_items;
    let cap = state.ch.max_weight;
    let mut candidates: Vec<usize> = (0..n).filter(|&i| !state.selected_bit[i]).collect();
    match strategy {
        0 => {
            const BETA_NUM: i64 = 3;
            const BETA_DEN: i64 = 20;
            candidates.sort_unstable_by(|&a, &b| {
                let wa = (state.ch.weights[a] as i64).max(1);
                let wb = (state.ch.weights[b] as i64).max(1);
                let ca = state.contrib[a] as i64;
                let cb = state.contrib[b] as i64;
                let ta = state.total_interactions[a];
                let tb = state.total_interactions[b];
                let adja = ca * BETA_DEN + BETA_NUM * (2 * ca - ta);
                let adjb = cb * BETA_DEN + BETA_NUM * (2 * cb - tb);
                let lhs = (adja as i128) * (wb as i128);
                let rhs = (adjb as i128) * (wa as i128);
                rhs.cmp(&lhs)
                    .then_with(|| state.support[b].cmp(&state.support[a]))
                    .then_with(|| tb.cmp(&ta))
                    .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        1 => {
            candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a].cmp(&state.ch.weights[b]).then(state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        2 => {
            candidates.sort_unstable_by_key(|&i| {
                -(state.total_interactions[i] + (state.contrib[i] as i64) * 10)
            });
        }
        3 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                if w > 0 { let eff = (state.contrib[i] as i64 * 100) / w; -eff } else { i64::MIN }
            });
        }
        4 => {
            candidates.sort_unstable_by_key(|&i| {
                let w = state.ch.weights[i] as i64;
                let c = state.contrib[i] as i64;
                -(c * w * w / 100)
            });
        }
        5 => {
            candidates.sort_unstable_by(|&a, &b| {
                let sa = state.support[a] as i64;
                let sb = state.support[b] as i64;
                let wa = (state.ch.weights[a] as i64).max(1);
                let wb = (state.ch.weights[b] as i64).max(1);
                let ca = state.contrib[a] as i64;
                let cb = state.contrib[b] as i64;
                let da = (ca * 1000) / wa + sa * 60 + state.total_interactions[a] / 500;
                let db = (cb * 1000) / wb + sb * 60 + state.total_interactions[b] / 500;
                db.cmp(&da).then_with(|| b.cmp(&a))
            });
        }
        _ => {
            candidates.sort_unstable_by_key(|&i| {
                let w = (state.ch.weights[i] as i64).max(1);
                let base = (state.contrib[i] as i64 * 10000) / (w * w);
                let penalty = (state.usage[i] as i64) * 50;
                -(base - penalty)
            });
        }
    }
    let allow_zero = strategy == 2;
    let mut added_any = false;
    for &i in &candidates {
        if state.selected_bit[i] { continue; }
        let w = state.ch.weights[i];
        if state.total_weight + w <= cap && (allow_zero || state.contrib[i] > 0) {
            state.add_item(i);
            added_any = true;
        }
    }
    let slack = state.slack();
    if slack >= 2 {
        let noise = if strategy == 0 { 0 } else { 0x0F };
        let allow_seed = slack >= 6;
        greedy_fill_with_beta(state, rng, noise, allow_seed);
    }
    let _ = added_any;
}

pub fn solve(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Solution {
    let params = Params::initialize(hyperparameters);
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    let (neigh_pre, total_pre) = build_sparse_neighbors_and_totals(challenge);
    let sample = n.min(96);
    let mut nz: u32 = 0;
    let mut tot: u32 = 0;
    for i in 0..sample {
        for j in 0..i {
            tot += 1;
            if challenge.interaction_values[i][j] != 0 { nz += 1; }
        }
    }
    let dens = if tot > 0 { (nz as f64) / (tot as f64) } else { 1.0 };
    let hard = dens < 0.10;
    let mut hubs_all: Vec<(i64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let w = challenge.weights[i] as i64;
        if w <= 0 { continue; }
        let s = (total_pre[i] * 1000) / w.max(1);
        hubs_all.push((s, i));
    }
    hubs_all.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let hubs_static: Vec<usize> = hubs_all.into_iter().take(320_usize.min(n)).map(|(_, i)| i).collect();
    let n_starts = params.n_starts(hard);
    let mut best: Option<State> = None;
    let mut second: Option<State> = None;
    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, &neigh_pre);
        match sid {
            0 => build_initial_solution(&mut st),
            1 => { construct_frontier_cluster_grow(&mut st, &mut rng); rebuild_windows(&mut st); }
            2 => { construct_pair_seed_beta(&mut st, &mut rng); rebuild_windows(&mut st); }
            _ => {
                let m = if hard { 5 } else { 4 };
                construct_forward_incremental(&mut st, m, &mut rng);
                rebuild_windows(&mut st);
            }
        }
        dp_refinement(&mut st);
        rebuild_windows(&mut st);
        micro_qkp_refinement(&mut st);
        local_search_vnd(&mut st, &params);
        if best.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = best; best = Some(st);
        } else if second.as_ref().map_or(true, |b| st.total_value > b.total_value) {
            second = Some(st);
        }
    }
    if best.is_some() && second.is_some() {
        let base_val = best.as_ref().unwrap().total_value;
        let mut best_new: Option<State> = None;
        let mut best_new_val = base_val;
        {
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, &neigh_pre);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] && b2.selected_bit[i] && hyb.total_weight + challenge.weights[i] <= challenge.max_weight {
                        hyb.add_item(i);
                    }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, &params);
            if hyb.total_value > best_new_val { best_new_val = hyb.total_value; best_new = Some(hyb); }
        }
        let (inter_cnt, union_cnt) = {
            let b1 = best.as_ref().unwrap();
            let b2 = second.as_ref().unwrap();
            let mut inter_cnt = 0usize;
            let mut union_cnt = 0usize;
            for i in 0..n {
                let a = b1.selected_bit[i];
                let b = b2.selected_bit[i];
                if a || b { union_cnt += 1; }
                if a && b { inter_cnt += 1; }
            }
            (inter_cnt, union_cnt)
        };
        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, &neigh_pre);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] || b2.selected_bit[i] { hyb.add_item(i); }
                }
            }
            if hyb.total_weight > challenge.max_weight {
                let mut sel: Vec<(i64, usize)> = Vec::new();
                for i in 0..n {
                    if !hyb.selected_bit[i] { continue; }
                    let c = hyb.contrib[i] as i64;
                    let w = challenge.weights[i] as i64;
                    let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                    sel.push((s, i));
                }
                sel.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                for &(_s, i) in &sel {
                    if hyb.total_weight <= challenge.max_weight { break; }
                    if hyb.selected_bit[i] { hyb.remove_item(i); }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, &params);
            if hyb.total_value > best_new_val { best_new = Some(hyb); }
        }
        if let Some(s) = best_new { best = Some(s); }
    }
    let mut state = best.unwrap();
    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        if state.selected_bit[i] { best_sel.push(i); }
    }
    let mut best_val = state.total_value;
    let mut stall_count = 0;
    let mut max_rounds = params.n_perturbation_rounds();
    max_rounds = max_rounds.min(if hard { 13 } else { 12 });
    let stall_limit = params.stall_limit_effective();
    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;
        state.snap_bits.clone_from(&state.selected_bit);
        state.snap_contrib.clone_from(&state.contrib);
        state.snap_support.clone_from(&state.support);
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;
        let apply_dp = !is_last_round && (perturbation_round < 3 || (perturbation_round % 4 == 0 && stall_count < 2));
        if apply_dp {
            rebuild_windows(&mut state);
            dp_refinement(&mut state);
            rebuild_windows(&mut state);
            micro_qkp_refinement(&mut state);
        }
        local_search_vnd(&mut state, &params);
        if state.total_value > best_val {
            best_val = state.total_value;
            best_sel.clear();
            for i in 0..n {
                if state.selected_bit[i] {
                    if state.usage[i] < u16::MAX { state.usage[i] += 1; }
                    best_sel.push(i);
                }
            }
            stall_count = 0;
        }
        if state.total_value <= prev_val {
            state.restore_snapshot(prev_val, prev_weight);
            if perturbation_round >= 7 && stall_count >= stall_limit { break; }
            if perturbation_round >= max_rounds - 1 { break; }
            stall_count += 1;
            let strategy = perturbation_round % 7;
            let strength = params.perturbation_strength_base() + (perturbation_round as usize) / 2;
            perturb_by_strategy(&mut state, &mut rng, strength, stall_count, strategy);
            greedy_reconstruct(&mut state, &mut rng, strategy);
            rebuild_windows(&mut state);
            dp_refinement(&mut state);
            rebuild_windows(&mut state);
            micro_qkp_refinement(&mut state);
            local_search_vnd(&mut state, &params);
            if state.total_value > best_val {
                best_val = state.total_value;
                best_sel.clear();
                for i in 0..n {
                    if state.selected_bit[i] {
                        if state.usage[i] < u16::MAX { state.usage[i] += 1; }
                        best_sel.push(i);
                    }
                }
                stall_count = 0;
            }
        }
    }
    Solution { items: best_sel }
}

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    if state.total_weight == 0 && n > 0 {
        let slack0 = state.slack();
        if slack0 > 0 {
            let tries = n.min(64);
            let samp = n.min(64);
            let mut best_seed: Option<usize> = None;
            let mut best_score: i64 = i64::MIN;
            for _ in 0..tries {
                let i = (rng.next_u32() as usize) % n;
                let wi = state.ch.weights[i];
                if wi == 0 || wi > slack0 { continue; }
                let mut est: i64 = 0;
                for _ in 0..samp {
                    let j = (rng.next_u32() as usize) % n;
                    est += state.ch.interaction_values[i][j] as i64;
                }
                let score = (est * 1000) / (wi as i64);
                if score > best_score { best_score = score; best_seed = Some(i); }
            }
            if let Some(i) = best_seed { state.add_item(i); }
        }
    }
    loop {
        let slack = state.slack();
        if slack == 0 { break; }
        let mut best_i: Option<usize> = None;
        let mut best_s: i64 = i64::MIN;
        let mut second_i: Option<usize> = None;
        let mut second_s: i64 = i64::MIN;
        for i in 0..n {
            if state.selected_bit[i] { continue; }
            let w_u = state.ch.weights[i];
            if w_u > slack { continue; }
            let c = state.contrib[i] as i64;
            if c <= 0 { continue; }
            let w = (w_u as i64).max(1);
            let mut s = match mode {
                2 => c,
                3 => (c * 1000) / w + (w_u as i64) * 3,
                _ => (c * 1000) / w,
            };
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }
            if s > best_s {
                second_s = best_s; second_i = best_i;
                best_s = s; best_i = Some(i);
            } else if s > second_s {
                second_s = s; second_i = Some(i);
            }
        }
        let pick = if mode >= 4 && second_i.is_some() {
            let m = if mode >= 5 { 1 } else { 3 };
            if (rng.next_u32() & m) == 0 { second_i } else { best_i }
        } else {
            best_i
        };
        if let Some(i) = pick { state.add_item(i); } else { break; }
    }
    if state.slack() >= 2 {
        let noise = if mode >= 4 { 0x1F } else { 0 };
        greedy_fill_with_beta(state, rng, noise, true);
    }
}
