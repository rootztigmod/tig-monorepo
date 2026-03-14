use std::collections::BTreeMap;
use serde_json::{Map, Value};
use tig_challenges::knapsack::*;

const N_IT_CONSTRUCT: usize = 2;
const CORE_HALF: usize = 25;
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
        if let Some(v) = self.perturbation_rounds {
            return v;
        }
        15 + (self.effort - 1) * 7
    }

    fn perturbation_strength_base(&self) -> usize {
        if let Some(v) = self.perturbation_strength {
            return v;
        }
        if self.effort >= 3 { 4 } else { 3 }
    }

    fn n_starts(&self, hard: bool, team_est: usize) -> usize {
        let mut base = if hard { 3 } else { 2 };
        if team_est >= 200 {
            base = (base + 1).min(4);
        }
        let bonus = if self.effort >= 5 { 2 } else if self.effort >= 3 { 1 } else { 0 };
        base + bonus
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
        for i in 0..n {
            contrib[i] = ch.values[i] as i32;
        }
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

fn set_windows_from_density(
    state: &mut State,
    by_density: &[usize],
    idx_first_rejected: usize,
    idx_last_inserted: usize,
) {
    let n = state.ch.num_items;
    let core_half = CORE_HALF;
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
        state.support[i] = unsafe { state.neigh.get_unchecked(i).len().min(u16::MAX as usize) as u16 };
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
                if score < worst_score {
                    worst_score = score;
                    worst_item = i;
                }
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
    const HUB_K: usize = 12;
    const SUP_BONUS: i64 = 70;

    let n = state.ch.num_items;
    let mut hubs: Vec<(i64, usize, u32)> = Vec::with_capacity(HUB_K);

    loop {
        let slack = state.slack();
        if slack == 0 { break; }

        let mut best_pos: Option<(usize, i64)> = None;
        let mut best_seed: Option<(usize, i64)> = None;

        hubs.clear();
        let mut hubs_min_score: i64 = i64::MAX;
        let mut hubs_min_pos: usize = 0;

        for i in 0..n {
            if state.selected_bit[i] { continue; }
            let w_u = state.ch.weights[i];
            if w_u == 0 || w_u > slack { continue; }
            let w = w_u as i64;
            let c = state.contrib[i] as i64;
            let tot_i = state.total_interactions[i];

            if c > 0 {
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot_i);
                let mut s = (adj * 1000) / w + (state.support[i] as i64) * SUP_BONUS;
                if noise_mask != 0 { s += (rng.next_u32() & noise_mask) as i64; }
                if best_pos.map_or(true, |(_, bs)| s > bs) { best_pos = Some((i, s)); }
            }

            let mut ss = (tot_i * 1000) / w;
            if noise_mask != 0 { ss += (rng.next_u32() & (noise_mask >> 1)) as i64; }
            if best_seed.map_or(true, |(_, bs)| ss > bs) { best_seed = Some((i, ss)); }

            if slack >= 2 {
                if hubs.len() < HUB_K {
                    hubs.push((ss, i, w_u));
                    if ss < hubs_min_score { hubs_min_score = ss; hubs_min_pos = hubs.len() - 1; }
                    if hubs.len() == 1 { hubs_min_score = ss; hubs_min_pos = 0; }
                } else if ss > hubs_min_score {
                    hubs[hubs_min_pos] = (ss, i, w_u);
                    hubs_min_score = hubs[0].0;
                    hubs_min_pos = 0;
                    for t in 1..hubs.len() {
                        if hubs[t].0 < hubs_min_score { hubs_min_score = hubs[t].0; hubs_min_pos = t; }
                    }
                }
            }
        }

        if let Some((i, _)) = best_pos {
            state.add_item(i);
            continue;
        }

        let mut best_pair: Option<(usize, usize, i64)> = None;
        if slack >= 2 && !hubs.is_empty() {
            for &(_hs, a, wa) in &hubs {
                if state.selected_bit[a] || wa == 0 || wa > slack { continue; }
                let row = unsafe { state.neigh.get_unchecked(a) };
                for &(bb, vv) in row.iter() {
                    let b = bb as usize;
                    if a == b || state.selected_bit[b] { continue; }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wa + wb > slack { continue; }
                    let v = vv as i64;
                    if v <= 0 { continue; }
                    let tot_a = state.total_interactions[a];
                    let tot_b = state.total_interactions[b];
                    let s = (v * 1_000_000) / ((wa + wb) as i64) + (tot_a + tot_b) / 2000;
                    if best_pair.map_or(true, |(_, _, bs)| s > bs) { best_pair = Some((a, b, s)); }
                }
            }
        }

        if let Some((a, b, _)) = best_pair {
            state.add_item(a);
            if state.slack() >= state.ch.weights[b] && !state.selected_bit[b] { state.add_item(b); }
            continue;
        }

        if allow_seed {
            if let Some((i, _)) = best_seed { state.add_item(i); } else { break; }
        } else {
            break;
        }
    }
}

fn construct_pair_seed_beta(state: &mut State, rng: &mut Rng) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    if state.total_weight == 0 {
        let cap = state.ch.max_weight;
        let mut best_pair: Option<(usize, usize, i64)> = None;
        for i in 0..n {
            let wi = state.ch.weights[i];
            if wi == 0 || wi > cap { continue; }
            let row = unsafe { state.neigh.get_unchecked(i) };
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

fn construct_forward_incremental(state: &mut State, mode: usize, rng: &mut Rng) {
    let n = state.ch.num_items;
    if n == 0 { return; }
    let cap = state.ch.max_weight;

    let mut seed_target: usize = 2;
    if mode >= 4 { seed_target += 1; }
    if mode >= 5 { seed_target += 1; }

    if state.total_weight == 0 {
        let mut best_h: Option<(i64, usize)> = None;
        let lim = state.hubs_static.len().min(192);
        for &i in state.hubs_static.iter().take(lim) {
            let wi = state.ch.weights[i];
            if wi == 0 || wi > cap { continue; }
            let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
            if best_h.map_or(true, |(bs, _)| s > bs) {
                best_h = Some((s, i));
            }
        }
        if best_h.is_none() {
            for i in 0..n {
                let wi = state.ch.weights[i];
                if wi == 0 || wi > cap { continue; }
                let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
                if best_h.map_or(true, |(bs, _)| s > bs) {
                    best_h = Some((s, i));
                }
            }
        }
        if let Some((_s, i)) = best_h {
            if !state.selected_bit[i] && state.total_weight + state.ch.weights[i] <= cap {
                state.add_item(i);
            }
        }

        if state.total_weight < cap && seed_target >= 2 && state.total_weight > 0 {
            let a = state.selected_items().get(0).copied().unwrap_or(usize::MAX);
            if a != usize::MAX {
                let slack = state.slack();
                let mut best_pair: Option<(i64, usize)> = None;
                let row = unsafe { state.neigh.get_unchecked(a) };
                for &(bb, vv) in row.iter() {
                    let b = bb as usize;
                    if state.selected_bit[b] { continue; }
                    let wb = state.ch.weights[b];
                    if wb == 0 || wb > slack { continue; }
                    let v = vv as i64;
                    if v <= 0 { continue; }
                    let wa = state.ch.weights[a];
                    let s = (v * 1_000_000) / ((wa + wb) as i64).max(1) + state.total_interactions[b] / 2000;
                    if best_pair.map_or(true, |(bs, _)| s > bs) {
                        best_pair = Some((s, b));
                    }
                }
                if let Some((_s, b)) = best_pair {
                    if !state.selected_bit[b] && state.total_weight + state.ch.weights[b] <= cap {
                        state.add_item(b);
                    }
                }
            }
        }

        let mut seeds_done = state.selected_items().len();
        if seeds_done < seed_target {
            let lim = state.hubs_static.len().min(192);
            for &i in state.hubs_static.iter().take(lim) {
                if seeds_done >= seed_target { break; }
                if state.selected_bit[i] { continue; }
                let wi = state.ch.weights[i];
                if wi == 0 { continue; }
                if state.total_weight + wi > cap { continue; }
                if state.contrib[i] <= 0 { continue; }
                state.add_item(i);
                seeds_done += 1;
            }
        }
    }

    let mut mark: Vec<u32> = vec![0u32; n];
    let mut epoch: u32 = 1;
    let mut active: Vec<usize> = state.selected_items();
    if active.is_empty() && n > 0 {
        let mut best_i: Option<(i64, usize)> = None;
        for i in 0..n {
            let wi = state.ch.weights[i];
            if wi == 0 || wi > cap { continue; }
            let s = (state.total_interactions[i] * 1000) / (wi as i64).max(1);
            if best_i.map_or(true, |(bs, _)| s > bs) { best_i = Some((s, i)); }
        }
        if let Some((_s, i)) = best_i {
            state.add_item(i);
            active.push(i);
        }
    }
    let mut act_idx: usize = 0;
    let mut cand: Vec<usize> = Vec::with_capacity(256);

    let mut global_best_add = |st: &mut State, rng: &mut Rng| -> Option<usize> {
        let slack = st.slack();
        if slack == 0 { return None; }
        let mut best: Option<(i64, usize)> = None;
        for i in 0..n {
            if st.selected_bit[i] { continue; }
            let wi = st.ch.weights[i];
            if wi > slack { continue; }
            let delta = st.contrib[i] as i64;
            if delta <= 0 { continue; }
            let w = (wi as i64).max(1);
            let mut s = (delta * 1000) / w;
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }
            if best.map_or(true, |(bs, _)| s > bs) { best = Some((s, i)); }
        }
        if let Some((_s, i)) = best {
            st.add_item(i);
            return Some(i);
        }
        None
    };

    loop {
        let slack = state.slack();
        if slack == 0 { break; }

        if cand.is_empty() {
            while act_idx < active.len() && cand.len() < 256 {
                let a = active[act_idx];
                act_idx += 1;
                let row = unsafe { state.neigh.get_unchecked(a) };
                let pref = row.len().min(MICRO_K * 2);
                for t in 0..pref {
                    let b = row[t].0 as usize;
                    if state.selected_bit[b] { continue; }
                    if unsafe { *mark.get_unchecked(b) } == epoch { continue; }
                    mark[b] = epoch;
                    cand.push(b);
                    if cand.len() >= 256 { break; }
                }
            }
            if cand.is_empty() {
                if let Some(i) = global_best_add(state, rng) {
                    active.push(i);
                    continue;
                } else {
                    break;
                }
            }
        }

        let mut best: Option<(i64, usize)> = None;
        for &i in &cand {
            if state.selected_bit[i] { continue; }
            let wi = state.ch.weights[i];
            if wi > slack { continue; }
            let delta = state.contrib[i] as i64;
            if delta <= 0 { continue; }
            let w = (wi as i64).max(1);
            let mut s = (delta * 1000) / w;
            if mode >= 4 {
                let mask = if mode >= 5 { 0x7F } else { 0x1F };
                s += (rng.next_u32() & mask) as i64;
            }
            if best.map_or(true, |(bs, _)| s > bs) { best = Some((s, i)); }
        }

        if let Some((_s, i)) = best {
            cand.clear();
            epoch = epoch.wrapping_add(1);
            if epoch == 0 { mark.fill(0); epoch = 1; }
            state.add_item(i);
            active.push(i);
            continue;
        } else {
            cand.clear();
            epoch = epoch.wrapping_add(1);
            if epoch == 0 { mark.fill(0); epoch = 1; }
            if act_idx >= active.len() {
                if let Some(i) = global_best_add(state, rng) {
                    active.push(i);
                    continue;
                } else {
                    break;
                }
            }
            continue;
        }
    }

    if state.slack() >= 2 {
        let noise = if mode >= 4 { 0x1F } else { 0 };
        greedy_fill_with_beta(state, rng, noise, true);
    }
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
    let mut w_star: usize = 0;
    let mut best_v: i64 = dp_cache[0];
    for w in 1..=myw {
        let v = dp_cache[w];
        if v > best_v {
            best_v = v;
            w_star = w;
        } else if v == best_v && w < w_star {
            w_star = w;
        }
    }
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

fn isqrt_u64(mut n: u64) -> u64 {
    let mut res: u64 = 0;
    let mut bit: u64 = 1u64 << (u64::BITS - 2);
    while bit > n { bit >>= 2; }
    while bit != 0 {
        let cand = res + bit;
        if n >= cand { n -= cand; res = (res >> 1) + bit; } else { res >>= 1; }
        bit >>= 2;
    }
    res
}

#[inline]
fn compress_core_val(v0: i64) -> i32 {
    let abs: u64 = if v0 >= 0 { v0 as u64 } else { (0u64).wrapping_sub(v0 as u64) };
    if abs == 0 { return 0; }
    let band_bits: u32 = (i32::BITS / 2) as u32;
    let band: u64 = 1u64 << band_bits;

    let mapped_mag: i64 = if abs <= band {
        abs as i64
    } else {
        let rem = abs - band;
        (band as i64).saturating_add(isqrt_u64(rem) as i64)
    };

    let v = if v0 >= 0 { mapped_mag } else { -mapped_mag };
    if v > i32::MAX as i64 { i32::MAX } else if v < i32::MIN as i64 { i32::MIN } else { v as i32 }
}

fn dp_refinement(state: &mut State) {
    let passes = if state.window_core.len() <= 160 { 2 } else { 1 };
    let n = state.ch.num_items;
    for _ in 0..passes {
        let mut core_val: Vec<i32> = Vec::with_capacity(state.window_core.len());
        if !state.window_core.is_empty() {
            let mut sel_core_bit = vec![false; n];
            let mut locked_not_selected_bit = vec![false; n];
            for &i in &state.window_core {
                if state.selected_bit[i] { sel_core_bit[i] = true; }
            }
            for &i in &state.window_locked {
                if !state.selected_bit[i] { locked_not_selected_bit[i] = true; }
            }
            for &it in &state.window_core {
                let mut sum_sel_core: i64 = 0;
                let mut sum_locked_ns: i64 = 0;
                let row = unsafe { state.neigh.get_unchecked(it) };
                for &(k, v) in row.iter() {
                    let j = k as usize;
                    if sel_core_bit[j] { sum_sel_core += v as i64; }
                    if locked_not_selected_bit[j] { sum_locked_ns += v as i64; }
                }
                let v0 = (state.contrib[it] as i64).saturating_sub(sum_sel_core).saturating_add(sum_locked_ns);
                core_val.push(compress_core_val(v0));
            }
        }
        let target = integer_core_target(
            state.ch, &state.window_locked, &state.window_core,
            &core_val, &mut state.dp_cache, &mut state.choose_cache,
        );
        apply_dp_target_via_ops(state, &target);
    }
}

fn micro_bb_dfs(
    idx: usize, cur_w: u32, cur_v: i64, mask: u64, rem_cap: u32, k: usize,
    w: &[u32], base: &[i64], inter: &[i64], rem_pos_base: &[i64], rem_pos_incident: &[i64],
    best_val: &mut i64, best_mask: &mut u64,
) {
    let ub = cur_v.saturating_add(rem_pos_base[idx]).saturating_add(rem_pos_incident[idx]);
    if ub <= *best_val { return; }
    if idx >= k {
        if cur_v > *best_val { *best_val = cur_v; *best_mask = mask; }
        return;
    }
    let wi = w[idx];
    if cur_w <= rem_cap && wi <= rem_cap.saturating_sub(cur_w) {
        let mut add_v = base[idx];
        let mut m = mask;
        while m != 0 {
            let lsb = m & m.wrapping_neg();
            let j = lsb.trailing_zeros() as usize;
            add_v = add_v.saturating_add(inter[idx * k + j]);
            m ^= lsb;
        }
        micro_bb_dfs(idx + 1, cur_w.saturating_add(wi), cur_v.saturating_add(add_v),
            mask | (1u64 << idx), rem_cap, k, w, base, inter, rem_pos_base, rem_pos_incident, best_val, best_mask);
    }
    micro_bb_dfs(idx + 1, cur_w, cur_v, mask, rem_cap, k, w, base, inter,
        rem_pos_base, rem_pos_incident, best_val, best_mask);
}

fn micro_qkp_refinement(state: &mut State) {
    let n = state.ch.num_items;
    if n == 0 || state.window_core.is_empty() { return; }
    let team_est = (state.ch.max_weight as usize) / 6;
    let micro_k: usize = if team_est <= 160 { 18 } else { MICRO_K };
    let rm_k: usize = if team_est <= 160 { 9 } else { MICRO_RM_K };
    let add_k: usize = if team_est <= 160 { 9 } else { MICRO_ADD_K };

    let mut sel: Vec<usize> = Vec::new();
    let mut unsel: Vec<usize> = Vec::new();
    for &i in &state.window_core {
        if state.selected_bit[i] { sel.push(i); } else { unsel.push(i); }
    }

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

    let g = guides.len().min(6);
    for t in 0..g {
        let vtx = guides[t];
        let row = unsafe { state.neigh.get_unchecked(vtx) };
        let pref = row.len().min(24);
        for u in 0..pref {
            let cand = row[u].0 as usize;
            if !state.selected_bit[cand] { push_unsel(&mut unsel, cand); }
        }
    }

    let hub_take: usize = if team_est <= 160 { 10 } else { 8 };
    let mut added_hubs: usize = 0;
    let lim = state.hubs_static.len().min(192);
    for &h in state.hubs_static.iter().take(lim) {
        if added_hubs >= hub_take { break; }
        if state.selected_bit[h] { continue; }
        push_unsel(&mut unsel, h);
        added_hubs += 1;
        let row = unsafe { state.neigh.get_unchecked(h) };
        let pref = row.len().min(16);
        for u in 0..pref {
            let cand = row[u].0 as usize;
            if !state.selected_bit[cand] { push_unsel(&mut unsel, cand); }
        }
    }

    let extra_r = state.window_rejected.len().min(24);
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] { unsel.push(i); }
    }
    let extra_l = state.window_locked.len().min(24);
    let start_l = state.window_locked.len().saturating_sub(extra_l);
    for &i in &state.window_locked[start_l..] {
        if state.selected_bit[i] { sel.push(i); }
    }

    if g >= 2 {
        let mut is_guide: Vec<bool> = vec![false; n];
        for t in 0..g {
            is_guide[guides[t]] = true;
        }

        let mut mark: Vec<bool> = vec![false; n];
        let mut pool: Vec<usize> = Vec::with_capacity(512);
        let mut push_pool = |idx: usize, mark: &mut [bool], pool: &mut Vec<usize>, st: &State| {
            if idx >= st.ch.num_items { return; }
            if st.selected_bit[idx] { return; }
            if !mark[idx] {
                mark[idx] = true;
                pool.push(idx);
            }
        };

        for &i in &state.window_rejected[..extra_r] {
            push_pool(i, &mut mark, &mut pool, state);
        }

        let lim_h = state.hubs_static.len().min(192);
        for &h in state.hubs_static.iter().take(lim_h) {
            push_pool(h, &mut mark, &mut pool, state);
        }

        for t in 0..g {
            let vtx = guides[t];
            let row = unsafe { state.neigh.get_unchecked(vtx) };
            let pref = row.len().min(24);
            for u in 0..pref {
                let cand = row[u].0 as usize;
                push_pool(cand, &mut mark, &mut pool, state);
            }
        }

        if !pool.is_empty() {
            let mut bridge: Vec<(usize, u16, i64)> = Vec::with_capacity(pool.len());
            for &a in &pool {
                if state.selected_bit[a] { continue; }
                let mut cnt: u16 = 0;
                let mut sum: i64 = 0;
                let row = unsafe { state.neigh.get_unchecked(a) };
                for &(k, v) in row.iter() {
                    if v <= 0 { continue; }
                    let j = k as usize;
                    if unsafe { *is_guide.get_unchecked(j) } {
                        cnt = cnt.saturating_add(1);
                        sum = sum.saturating_add(v as i64);
                    }
                }
                if cnt >= 2 && sum > 0 {
                    bridge.push((a, cnt, sum));
                }
            }
            if !bridge.is_empty() {
                bridge.sort_unstable_by(|a, b| {
                    b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)).then_with(|| a.0.cmp(&b.0))
                });
                let take = add_k.min(bridge.len());
                for t in 0..take {
                    push_unsel(&mut unsel, bridge[t].0);
                }
            }
        }
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
    for &i in sel.iter().take(rm_k) { push_u(&mut cand, i); if cand.len() >= micro_k { break; } }
    for &i in unsel.iter().take(add_k) { push_u(&mut cand, i); if cand.len() >= micro_k { break; } }
    if cand.len() < 2 { return; }

    let k = cand.len();
    if k >= (u64::BITS as usize) { return; }

    let mut sel_cand: Vec<usize> = Vec::new();
    let mut sel_cand_w: u32 = 0;
    for &it in &cand {
        if state.selected_bit[it] { sel_cand.push(it); sel_cand_w = sel_cand_w.saturating_add(state.ch.weights[it]); }
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

    let mut cur_mask_orig: u64 = 0;
    for t in 0..k { if state.selected_bit[cand[t]] { cur_mask_orig |= 1u64 << t; } }

    let mut cur_w_sum: u32 = 0;
    let mut cur_v: i64 = 0;
    for i in 0..k {
        if ((cur_mask_orig >> i) & 1) == 0 { continue; }
        cur_w_sum = cur_w_sum.saturating_add(w[i]);
        cur_v = cur_v.saturating_add(base[i]);
        for j in 0..i {
            if ((cur_mask_orig >> j) & 1) != 0 { cur_v = cur_v.saturating_add(inter[i * k + j]); }
        }
    }
    if cur_w_sum > rem_cap { return; }

    let mut pos_incident: Vec<i64> = vec![0; k];
    for i in 0..k {
        let mut s: i64 = 0;
        for j in 0..k { let v = inter[i * k + j]; if v > 0 { s = s.saturating_add(v); } }
        pos_incident[i] = s;
    }

    let mut ord: Vec<usize> = (0..k).collect();
    ord.sort_unstable_by(|&a, &b| {
        let sa = base[a].saturating_add(pos_incident[a]);
        let sb = base[b].saturating_add(pos_incident[b]);
        sb.cmp(&sa).then_with(|| b.cmp(&a))
    });

    let mut w_o: Vec<u32> = vec![0; k];
    let mut base_o: Vec<i64> = vec![0; k];
    let mut pos_incident_o: Vec<i64> = vec![0; k];
    for oi in 0..k { let a = ord[oi]; w_o[oi] = w[a]; base_o[oi] = base[a]; pos_incident_o[oi] = pos_incident[a]; }

    let mut inter_o: Vec<i64> = vec![0; k * k];
    for i in 0..k { let a = ord[i]; for j in 0..k { let b = ord[j]; inter_o[i * k + j] = inter[a * k + b]; } }

    let mut cur_mask: u64 = 0;
    for oi in 0..k { let orig = ord[oi]; if ((cur_mask_orig >> orig) & 1) != 0 { cur_mask |= 1u64 << oi; } }

    let mut rem_pos_base: Vec<i64> = vec![0; k + 1];
    let mut rem_pos_inc: Vec<i64> = vec![0; k + 1];
    for i in (0..k).rev() {
        rem_pos_base[i] = rem_pos_base[i + 1].saturating_add(base_o[i].max(0));
        rem_pos_inc[i] = rem_pos_inc[i + 1].saturating_add(pos_incident_o[i]);
    }

    let mut best_mask: u64 = cur_mask;
    let mut best_val: i64 = cur_v;
    micro_bb_dfs(0, 0, 0, 0, rem_cap, k, &w_o, &base_o, &inter_o, &rem_pos_base, &rem_pos_inc, &mut best_val, &mut best_mask);

    if best_mask == cur_mask { return; }

    let mut to_remove: Vec<usize> = Vec::new();
    let mut to_add: Vec<usize> = Vec::new();
    for oi in 0..k {
        let it = cand[ord[oi]];
        let want = ((best_mask >> oi) & 1) != 0;
        let have = state.selected_bit[it];
        if have && !want { to_remove.push(it); } else if !have && want { to_add.push(it); }
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
        for &cand in &state.window_rejected {
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
    let n = state.ch.num_items;
    if n == 0 { return false; }
    let edge_lim: usize = 9000;
    let node_lim: usize = 72;
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
            let row = unsafe { state.neigh.get_unchecked(idx) };
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
    if let Some((r, a, b, _)) = best { state.remove_item(r); state.add_item(a); state.add_item(b); true } else { false }
}

fn apply_best_replace21_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 { return false; }
    let n = state.ch.num_items;
    if n == 0 { return false; }

    let slack0 = state.slack();

    let mut rm: Vec<usize> = Vec::with_capacity(used.len());
    for &i in used {
        if state.selected_bit[i] && state.ch.weights[i] != 0 {
            rm.push(i);
        }
    }
    if rm.len() < 2 { return false; }
    rm.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64; let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64; let wb = state.ch.weights[b] as i64;
        (ca * wb).cmp(&(cb * wa))
    });
    if rm.len() > 14 { rm.truncate(14); }
    if rm.len() < 2 { return false; }

    let mut mark = vec![false; n];
    let mut pool: Vec<usize> = Vec::with_capacity(state.window_core.len() + 48 + 48);
    for &i in &state.window_core {
        if !state.selected_bit[i] && !mark[i] {
            mark[i] = true;
            pool.push(i);
        }
    }
    let extra_r = state.window_rejected.len().min(48);
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] && !mark[i] {
            mark[i] = true;
            pool.push(i);
        }
    }
    let extra_h = state.hubs_static.len().min(48);
    for &i in &state.hubs_static[..extra_h] {
        if !state.selected_bit[i] && !mark[i] {
            mark[i] = true;
            pool.push(i);
        }
    }

    let mut blocked_scored: Vec<(i32, u16, usize)> = Vec::new();
    blocked_scored.reserve(pool.len());
    for &c in &pool {
        if state.selected_bit[c] { continue; }
        let wc = state.ch.weights[c];
        if wc == 0 { continue; }
        if wc <= slack0 { continue; }
        let dc = state.contrib[c];
        if dc <= 0 { continue; }
        blocked_scored.push((dc, state.support[c], c));
    }
    if blocked_scored.is_empty() { return false; }
    blocked_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));
    if blocked_scored.len() > 28 { blocked_scored.truncate(28); }

    let mut best: Option<(usize, usize, usize, i64)> = None;

    for &(_dc, _sup, c) in &blocked_scored {
        let wc = state.ch.weights[c];
        let need = wc - slack0;

        let mut best_pair: Option<(usize, usize, i64, u32, u32)> = None;
        for x in 0..rm.len() {
            let a = rm[x];
            if !state.selected_bit[a] { continue; }
            let wa = state.ch.weights[a];
            for y in (x + 1)..rm.len() {
                let b = rm[y];
                if !state.selected_bit[b] { continue; }
                let wb = state.ch.weights[b];
                if wa.saturating_add(wb) < need { continue; }
                let loss = (state.contrib[a] as i64) + (state.contrib[b] as i64) - (state.ch.interaction_values[a][b] as i64);
                if best_pair.map_or(true, |(_, _, bl, _, _)| loss < bl) {
                    best_pair = Some((a, b, loss, wa, wb));
                }
            }
        }
        let Some((a, b, loss, wa, wb)) = best_pair else { continue; };

        let new_w = state.total_weight.saturating_sub(wa).saturating_sub(wb).saturating_add(wc);
        if new_w > cap { continue; }

        let delta = (state.contrib[c] as i64)
            - (state.ch.interaction_values[c][a] as i64)
            - (state.ch.interaction_values[c][b] as i64)
            - loss;

        if delta > 0 && best.map_or(true, |(_, _, _, bd)| delta > bd) {
            best = Some((c, a, b, delta));
        }
    }

    if let Some((c, a, b, _)) = best {
        let wc = state.ch.weights[c];
        let wa = state.ch.weights[a];
        let wb = state.ch.weights[b];
        let new_w = state.total_weight.saturating_sub(wa).saturating_sub(wb).saturating_add(wc);
        if new_w > cap { return false; }
        if state.selected_bit[a] { state.remove_item(a); }
        if state.selected_bit[b] { state.remove_item(b); }
        if !state.selected_bit[c] && state.total_weight + wc <= cap {
            state.add_item(c);
            true
        } else {
            false
        }
    } else {
        false
    }
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
                if delta > 0 && best.map_or(true, |(_, _, bd)| delta > bd) { best = Some((cand, rm, delta)); }
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
    let mut best: Option<(usize, usize, i64, i64)> = None;
    for &rm in used {
        if !state.selected_bit[rm] { continue; }
        let wrm = state.ch.weights[rm];
        let row = unsafe { state.neigh.get_unchecked(rm) };
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

fn apply_best_swap_frontier_global(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.is_empty() { return false; }
    let n = state.ch.num_items;
    if n == 0 { return false; }
    const BETA_NUM: i64 = 3;
    const BETA_DEN: i64 = 20;
    const TOP_CAND: usize = 48;
    const TOP_RM: usize = 32;
    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa + (state.support[a] as i64) * 140;
        let sb = (state.contrib[b] as i64 * 1000) / wb + (state.support[b] as i64) * 140;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm.len() > TOP_RM { rm.truncate(TOP_RM); }
    let mut max_rm_w: u32 = 0;
    for &i in used { let w = state.ch.weights[i]; if w > max_rm_w { max_rm_w = w; } }
    if max_rm_w == 0 { return false; }
    let slack0 = state.slack();
    let edge_lim: usize = 7000;
    let node_lim: usize = 64;
    let start = (((state.total_value as u64) as usize) ^ ((state.total_weight as usize).wrapping_mul(911))) % n;
    let mut step = (n / 97).max(1);
    step |= 1;
    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(TOP_CAND);
    let push_top_unique = |list: &mut Vec<(i64, usize)>, s: i64, idx: usize| {
        for &(_, j) in list.iter() { if j == idx { return; } }
        if list.len() < TOP_CAND { list.push((s, idx)); return; }
        let mut worst_pos = 0usize;
        let mut worst_s = list[0].0;
        for t in 1..list.len() { if list[t].0 < worst_s { worst_s = list[t].0; worst_pos = t; } }
        if s > worst_s { list[worst_pos] = (s, idx); }
    };
    let mut scanned_edges: usize = 0;
    let mut scanned_nodes: usize = 0;
    let mut idx = start;
    let mut tries: usize = 0;
    while tries < n && scanned_nodes < node_lim && scanned_edges < edge_lim {
        if state.selected_bit[idx] {
            scanned_nodes += 1;
            let row = unsafe { state.neigh.get_unchecked(idx) };
            for &(cj, _vv) in row.iter() {
                scanned_edges += 1;
                if scanned_edges > edge_lim { break; }
                let cand = cj as usize;
                if state.selected_bit[cand] { continue; }
                let wc = state.ch.weights[cand];
                if wc == 0 { continue; }
                if wc > slack0.saturating_add(max_rm_w) { continue; }
                let c = state.contrib[cand] as i64;
                if c <= 0 { continue; }
                let tot = state.total_interactions[cand];
                let adj = c * BETA_DEN + BETA_NUM * (2 * c - tot);
                let s = (adj * 1000) / (wc as i64).max(1) + (state.support[cand] as i64) * 60;
                push_top_unique(&mut cand_list, s, cand);
            }
        }
        idx += step;
        if idx >= n { idx -= n; }
        tries += 1;
    }
    if cand_list.is_empty() { return false; }
    let mut best: Option<(usize, usize, i64, i64)> = None;
    for &(_s0, cand) in &cand_list {
        let wc = state.ch.weights[cand];
        if wc == 0 { continue; }
        for &r in &rm {
            if !state.selected_bit[r] { continue; }
            let wr = state.ch.weights[r];
            if (state.total_weight as u64) + (wc as u64) > (cap as u64) + (wr as u64) { continue; }
            let inter = state.ch.interaction_values[cand][r] as i64;
            let delta = (state.contrib[cand] as i64) - (state.contrib[r] as i64) - inter;
            if delta <= 0 { continue; }
            let score: i64 = if wc == wr {
                delta * 1_000_000
            } else if wc < wr {
                delta * 1000 + (wr as i64 - wc as i64)
            } else {
                let dw = (wc - wr) as i64;
                (delta * 1000) / dw.max(1)
            };
            if best.map_or(true, |(_, _, bs, bd)| score > bs || (score == bs && delta > bd)) {
                best = Some((cand, r, score, delta));
            }
        }
    }
    if let Some((cand, r, _, _)) = best { state.replace_item(r, cand); true } else { false }
}

fn apply_best_exchange22_windowed(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 { return false; }
    if state.total_weight > cap { return false; }
    let n = state.ch.num_items;
    if n == 0 { return false; }

    let mut anchors: Vec<usize> = Vec::new();
    anchors.reserve(used.len().min(16));
    for &i in used {
        if state.selected_bit[i] {
            anchors.push(i);
        }
    }
    if anchors.is_empty() { return false; }
    anchors.sort_unstable_by(|&a, &b| {
        state.support[b].cmp(&state.support[a])
            .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            .then_with(|| b.cmp(&a))
    });
    if anchors.len() > 6 { anchors.truncate(6); }

    let mut groups: Vec<Vec<usize>> = Vec::with_capacity(anchors.len());
    for &a in &anchors {
        let row = unsafe { state.neigh.get_unchecked(a) };
        let pref = row.len().min(24);
        let mut list: Vec<usize> = Vec::with_capacity(MICRO_ADD_K);
        for t in 0..pref {
            let cand = row[t].0 as usize;
            if state.selected_bit[cand] { continue; }
            if state.contrib[cand] <= 0 { continue; }
            if list.iter().any(|&x| x == cand) { continue; }
            list.push(cand);
            if list.len() >= MICRO_ADD_K { break; }
        }
        groups.push(list);
    }

    let mut any = false;
    for g in &groups {
        if g.len() >= 2 { any = true; break; }
    }
    if !any {
        let mut tot: usize = 0;
        for g in &groups { tot += g.len(); }
        if tot < 2 { return false; }
    }

    let mut rm: Vec<usize> = used.to_vec();
    rm.sort_unstable_by(|&a, &b| {
        let ca = state.contrib[a] as i64; let cb = state.contrib[b] as i64;
        let wa = state.ch.weights[a] as i64; let wb = state.ch.weights[b] as i64;
        (ca * wb).cmp(&(cb * wa))
    });
    if rm.len() > 14 { rm.truncate(14); }
    if rm.len() < 2 { return false; }

    let mut best: Option<(i64, i64, usize, usize, usize, usize)> = None;

    for x in 0..rm.len() {
        let r1 = rm[x];
        if !state.selected_bit[r1] { continue; }
        let wr1 = state.ch.weights[r1] as u64;
        for y in (x + 1)..rm.len() {
            let r2 = rm[y];
            if !state.selected_bit[r2] { continue; }
            let wr2 = state.ch.weights[r2] as u64;
            let base_w = (state.total_weight as u64).saturating_sub(wr1).saturating_sub(wr2);
            if base_w > cap as u64 { continue; }
            let inter_r12 = state.ch.interaction_values[r1][r2] as i64;

            for gi in 0..groups.len() {
                let g = &groups[gi];
                if g.len() >= 2 {
                    let anchor = anchors[gi];
                    for ia in 0..g.len() {
                        let a = g[ia];
                        let wa = state.ch.weights[a] as u64;
                        for ib in (ia + 1)..g.len() {
                            let b = g[ib];
                            if a == b { continue; }
                            let wb = state.ch.weights[b] as u64;
                            let new_w = base_w.saturating_add(wa).saturating_add(wb);
                            if new_w > cap as u64 { continue; }

                            let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64)
                                - (state.contrib[r1] as i64) - (state.contrib[r2] as i64)
                                - (state.ch.interaction_values[a][r1] as i64)
                                - (state.ch.interaction_values[a][r2] as i64)
                                - (state.ch.interaction_values[b][r1] as i64)
                                - (state.ch.interaction_values[b][r2] as i64)
                                + (state.ch.interaction_values[a][b] as i64)
                                + inter_r12;
                            if delta <= 0 { continue; }

                            let iaa = state.ch.interaction_values[a][anchor] as i64;
                            let ibb = state.ch.interaction_values[b][anchor] as i64;
                            let bonus = iaa.max(0).saturating_add(ibb.max(0));
                            let score = delta.saturating_add(bonus);

                            if best.map_or(true, |(bs, bd, br1, br2, ba, bb)| {
                                score > bs || (score == bs && (delta > bd || (delta == bd && (r1, r2, a, b) < (br1, br2, ba, bb))))
                            }) {
                                best = Some((score, delta, r1, r2, a, b));
                            }
                        }
                    }
                }
            }

            for gi in 0..groups.len() {
                let g1 = &groups[gi];
                if g1.is_empty() { continue; }
                let anc1 = anchors[gi];
                for gj in (gi + 1)..groups.len() {
                    let g2 = &groups[gj];
                    if g2.is_empty() { continue; }
                    let anc2 = anchors[gj];
                    for &a in g1 {
                        let wa = state.ch.weights[a] as u64;
                        for &b in g2 {
                            if a == b { continue; }
                            let wb = state.ch.weights[b] as u64;
                            let new_w = base_w.saturating_add(wa).saturating_add(wb);
                            if new_w > cap as u64 { continue; }

                            let delta = (state.contrib[a] as i64) + (state.contrib[b] as i64)
                                - (state.contrib[r1] as i64) - (state.contrib[r2] as i64)
                                - (state.ch.interaction_values[a][r1] as i64)
                                - (state.ch.interaction_values[a][r2] as i64)
                                - (state.ch.interaction_values[b][r1] as i64)
                                - (state.ch.interaction_values[b][r2] as i64)
                                + (state.ch.interaction_values[a][b] as i64)
                                + inter_r12;
                            if delta <= 0 { continue; }

                            let bonus = (state.ch.interaction_values[a][anc1] as i64).max(0)
                                .saturating_add((state.ch.interaction_values[b][anc1] as i64).max(0))
                                .saturating_add((state.ch.interaction_values[a][anc2] as i64).max(0))
                                .saturating_add((state.ch.interaction_values[b][anc2] as i64).max(0));
                            let score = delta.saturating_add(bonus);

                            if best.map_or(true, |(bs, bd, br1, br2, ba, bb)| {
                                score > bs || (score == bs && (delta > bd || (delta == bd && (r1, r2, a, b) < (br1, br2, ba, bb))))
                            }) {
                                best = Some((score, delta, r1, r2, a, b));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((_score, _delta, r1, r2, a, b)) = best {
        if state.selected_bit[r1] { state.remove_item(r1); }
        if state.selected_bit[r2] { state.remove_item(r2); }
        if !state.selected_bit[a] && (state.total_weight as u64) + (state.ch.weights[a] as u64) <= cap as u64 {
            state.add_item(a);
        } else {
            return false;
        }
        if !state.selected_bit[b] && (state.total_weight as u64) + (state.ch.weights[b] as u64) <= cap as u64 {
            state.add_item(b);
        } else {
            if state.selected_bit[a] { state.remove_item(a); }
            if !state.selected_bit[r1] && (state.total_weight as u64) + (state.ch.weights[r1] as u64) <= cap as u64 { state.add_item(r1); }
            if !state.selected_bit[r2] && (state.total_weight as u64) + (state.ch.weights[r2] as u64) <= cap as u64 { state.add_item(r2); }
            return false;
        }
        true
    } else {
        false
    }
}

fn apply_best_remove2_add1_synergy_injection(state: &mut State, used: &[usize]) -> bool {
    let cap = state.ch.max_weight;
    if used.len() < 2 { return false; }
    if state.total_weight > cap { return false; }
    let n = state.ch.num_items;
    if n == 0 { return false; }

    let mut guides: Vec<usize> = Vec::new();
    for &i in used {
        if state.selected_bit[i] { guides.push(i); }
    }
    if guides.is_empty() { return false; }
    guides.sort_unstable_by(|&a, &b| {
        state.support[b].cmp(&state.support[a])
            .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            .then_with(|| b.cmp(&a))
    });
    if guides.len() > 6 { guides.truncate(6); }

    let mut mark: Vec<u32> = vec![0u32; n];
    let mut epoch: u32 = 1;
    let mut pool: Vec<usize> = Vec::with_capacity(512);
    let mut push = |idx: usize, pool: &mut Vec<usize>, mark: &mut [u32], epoch: u32| {
        if mark[idx] != epoch {
            mark[idx] = epoch;
            pool.push(idx);
        }
    };

    epoch = epoch.wrapping_add(1);
    if epoch == 0 { mark.fill(0); epoch = 1; }
    pool.clear();

    let extra_r = state.window_rejected.len().min(64);
    for &i in &state.window_rejected[..extra_r] {
        if !state.selected_bit[i] { push(i, &mut pool, &mut mark, epoch); }
    }
    for &i in state.hubs_static.iter().take(64) {
        if !state.selected_bit[i] { push(i, &mut pool, &mut mark, epoch); }
    }
    for &g in &guides {
        let row = unsafe { state.neigh.get_unchecked(g) };
        let pref = row.len().min(MICRO_K * 2);
        for t in 0..pref {
            let cand = row[t].0 as usize;
            if !state.selected_bit[cand] { push(cand, &mut pool, &mut mark, epoch); }
        }
    }
    if pool.is_empty() { return false; }

    let mut cand_list: Vec<(i64, usize)> = Vec::with_capacity(MICRO_K * 3);
    for &a in &pool {
        if state.selected_bit[a] { continue; }
        let wa = state.ch.weights[a];
        if wa == 0 { continue; }
        if wa <= state.slack() { continue; }
        let mut s = state.contrib[a] as i64;
        for &g in &guides {
            let v = state.ch.interaction_values[a][g] as i64;
            if v > 0 { s = s.saturating_add(v); }
        }
        let denom = (wa as i64).max(1);
        let sc = (s * 1000) / denom;
        cand_list.push((sc, a));
    }
    if cand_list.is_empty() { return false; }
    cand_list.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if cand_list.len() > MICRO_K * 2 { cand_list.truncate(MICRO_K * 2); }

    let mut rm_list: Vec<usize> = Vec::new();
    rm_list.reserve(used.len());
    for &i in used {
        if state.selected_bit[i] && state.ch.weights[i] != 0 {
            rm_list.push(i);
        }
    }
    if rm_list.len() < 2 { return false; }
    rm_list.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa;
        let sb = (state.contrib[b] as i64 * 1000) / wb;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm_list.len() > 32 { rm_list.truncate(32); }
    if rm_list.len() < 2 { return false; }

    let slack0 = state.slack();
    let mut best: Option<(i64, usize, usize, usize)> = None;

    for &(_sc, a) in &cand_list {
        if state.selected_bit[a] { continue; }
        let wa = state.ch.weights[a];
        if wa == 0 { continue; }
        if wa <= slack0 { continue; }
        let need = wa - slack0;
        for x in 0..rm_list.len() {
            let r1 = rm_list[x];
            if !state.selected_bit[r1] { continue; }
            let wr1 = state.ch.weights[r1];
            for y in (x + 1)..rm_list.len() {
                let r2 = rm_list[y];
                if !state.selected_bit[r2] { continue; }
                let wr2 = state.ch.weights[r2];
                if wr1.saturating_add(wr2) < need { continue; }
                let new_w = state.total_weight.saturating_sub(wr1).saturating_sub(wr2).saturating_add(wa);
                if new_w > cap { continue; }
                let ca = state.contrib[a] as i64;
                let cr1 = state.contrib[r1] as i64;
                let cr2 = state.contrib[r2] as i64;
                let ir12 = state.ch.interaction_values[r1][r2] as i64;
                let iar1 = state.ch.interaction_values[a][r1] as i64;
                let iar2 = state.ch.interaction_values[a][r2] as i64;
                let delta = ca - cr1 - cr2 + ir12 - iar1 - iar2;
                if delta > 0 && best.map_or(true, |(bd, _, _, _)| delta > bd) {
                    best = Some((delta, a, r1, r2));
                }
            }
        }
    }

    if let Some((_d, a, r1, r2)) = best {
        if state.selected_bit[r1] { state.remove_item(r1); }
        if state.selected_bit[r2] { state.remove_item(r2); }
        if !state.selected_bit[a] && state.total_weight + state.ch.weights[a] <= cap {
            state.add_item(a);
            return true;
        }
    }
    false
}

fn apply_reinsert10_greedy(state: &mut State, used: &[usize]) -> bool {
    const SUP_ADD_BONUS: i64 = 40;
    if used.is_empty() { return false; }
    let cap = state.ch.max_weight;
    if state.total_weight > cap { return false; }
    let n = state.ch.num_items;
    if n == 0 { return false; }

    let mut rm_list: Vec<usize> = Vec::with_capacity(used.len());
    for &i in used {
        if state.selected_bit[i] && state.ch.weights[i] != 0 { rm_list.push(i); }
    }
    if rm_list.is_empty() { return false; }

    rm_list.sort_unstable_by(|&a, &b| {
        let wa = (state.ch.weights[a] as i64).max(1);
        let wb = (state.ch.weights[b] as i64).max(1);
        let sa = (state.contrib[a] as i64 * 1000) / wa;
        let sb = (state.contrib[b] as i64 * 1000) / wb;
        sa.cmp(&sb).then_with(|| a.cmp(&b))
    });
    if rm_list.len() > 24 { rm_list.truncate(24); }

    let mut mark = vec![false; n];
    let mut cand: Vec<usize> = Vec::with_capacity(256);

    for &rm in &rm_list {
        let old_val = state.total_value;
        let old_w = state.total_weight;

        state.remove_item(rm);

        cand.clear();
        for v in mark.iter_mut() { *v = false; }

        let mut push = |x: usize, cand: &mut Vec<usize>, mark: &mut [bool]| {
            if !mark[x] {
                mark[x] = true;
                cand.push(x);
            }
        };

        for &i in &state.window_core {
            if !state.selected_bit[i] { push(i, &mut cand, &mut mark); }
        }
        let extra_r = state.window_rejected.len().min(64);
        for &i in &state.window_rejected[..extra_r] {
            if !state.selected_bit[i] { push(i, &mut cand, &mut mark); }
        }
        let row = unsafe { state.neigh.get_unchecked(rm) };
        for &(k, _v) in row.iter() {
            let i = k as usize;
            if !state.selected_bit[i] { push(i, &mut cand, &mut mark); }
        }

        let mut added: Vec<usize> = Vec::new();
        loop {
            let slack = cap - state.total_weight;
            if slack == 0 { break; }
            let mut best: Option<(usize, i64)> = None;
            for &i in &cand {
                if state.selected_bit[i] { continue; }
                let w = state.ch.weights[i];
                if w == 0 || w > slack { continue; }
                let delta = state.contrib[i];
                if delta <= 0 { continue; }
                let s = (delta as i64) + (state.support[i] as i64) * SUP_ADD_BONUS;
                if best.map_or(true, |(_, bs)| s > bs) { best = Some((i, s)); }
            }
            if let Some((i, _)) = best {
                state.add_item(i);
                added.push(i);
                continue;
            }
            break;
        }

        let improved = state.total_value > old_val;
        if improved {
            return true;
        }

        for &i in added.iter().rev() {
            if state.selected_bit[i] { state.remove_item(i); }
        }
        if !state.selected_bit[rm] {
            if state.total_weight + state.ch.weights[rm] <= cap {
                state.add_item(rm);
            } else {
                state.total_value = old_val;
                state.total_weight = old_w;
                return false;
            }
        }

        if state.total_value != old_val || state.total_weight != old_w {
            state.total_value = old_val;
            state.total_weight = old_w;
            return false;
        }
    }
    false
}

fn apply_negative_edge_cleanup(state: &mut State, used: &[usize]) -> bool {
    let n = state.ch.num_items;
    if n == 0 || used.is_empty() { return false; }
    let cap = state.ch.max_weight;
    if state.total_weight > cap { return false; }

    let mut rm_cand: Vec<(i64, u32, usize)> = Vec::new();
    rm_cand.reserve(used.len());
    for &i in used {
        if !state.selected_bit[i] { continue; }
        let inter = (state.contrib[i] - state.ch.values[i] as i32) as i64;
        rm_cand.push((inter, state.ch.weights[i], i));
    }
    if rm_cand.is_empty() { return false; }
    rm_cand.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));
    if rm_cand.len() > MICRO_K { rm_cand.truncate(MICRO_K); }

    let mut anchors: Vec<usize> = Vec::new();
    anchors.reserve(used.len().min(MICRO_ADD_K));
    let mut tmp: Vec<usize> = Vec::new();
    tmp.reserve(used.len());
    for &i in used {
        if state.selected_bit[i] { tmp.push(i); }
    }
    tmp.sort_unstable_by(|&a, &b| {
        state.support[b].cmp(&state.support[a])
            .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            .then_with(|| b.cmp(&a))
    });
    for &i in tmp.iter().take(MICRO_ADD_K) { anchors.push(i); }
    if anchors.is_empty() { return false; }

    let mut mark: Vec<bool> = vec![false; n];
    let mut pool: Vec<usize> = Vec::with_capacity(state.window_core.len().min(MICRO_K * MICRO_RM_K));

    let mut add_to_pool = |x: usize, mark: &mut [bool], pool: &mut Vec<usize>| {
        if !mark[x] {
            mark[x] = true;
            pool.push(x);
        }
    };

    for &(_inter, _w, rm) in &rm_cand {
        if !state.selected_bit[rm] { continue; }
        let old_val = state.total_value;
        let old_w = state.total_weight;

        state.remove_item(rm);

        pool.clear();
        mark.fill(false);

        for &i in &state.window_core {
            if !state.selected_bit[i] && i != rm { add_to_pool(i, &mut mark, &mut pool); }
        }
        let extra_r = state.window_rejected.len().min(MICRO_K);
        for &i in &state.window_rejected[..extra_r] {
            if !state.selected_bit[i] && i != rm { add_to_pool(i, &mut mark, &mut pool); }
        }
        let extra_h = state.hubs_static.len().min(MICRO_K);
        for &i in &state.hubs_static[..extra_h] {
            if !state.selected_bit[i] && i != rm { add_to_pool(i, &mut mark, &mut pool); }
        }

        for &a in &anchors {
            if !state.selected_bit[a] { continue; }
            let row = unsafe { state.neigh.get_unchecked(a) };
            let pref = row.len().min(MICRO_K);
            for t in 0..pref {
                let b = row[t].0 as usize;
                if !state.selected_bit[b] && b != rm { add_to_pool(b, &mut mark, &mut pool); }
            }
        }

        let mut added: Vec<usize> = Vec::new();
        loop {
            let slack = cap - state.total_weight;
            if slack == 0 { break; }
            let mut best: Option<(i64, u16, usize)> = None;
            for &i in &pool {
                if state.selected_bit[i] { continue; }
                if i == rm { continue; }
                let wi = state.ch.weights[i];
                if wi == 0 || wi > slack { continue; }
                let di = state.contrib[i];
                if di <= 0 { continue; }
                let s = di as i64;
                let sup = state.support[i];
                if best.map_or(true, |(bs, bsu, _)| s > bs || (s == bs && sup > bsu)) {
                    best = Some((s, sup, i));
                }
            }
            if let Some((_s, _sup, i)) = best {
                state.add_item(i);
                added.push(i);
                continue;
            }
            break;
        }

        if state.total_value > old_val {
            return true;
        }

        for &i in added.iter().rev() {
            if state.selected_bit[i] { state.remove_item(i); }
        }
        if !state.selected_bit[rm] {
            if state.total_weight + state.ch.weights[rm] <= cap {
                state.add_item(rm);
            } else {
                state.total_value = old_val;
                state.total_weight = old_w;
                return false;
            }
        }

        if state.total_value != old_val || state.total_weight != old_w {
            state.total_value = old_val;
            state.total_weight = old_w;
            return false;
        }
    }

    false
}

fn local_search_vnd(state: &mut State, params: &Params) {
    let mut iterations = 0;
    let max_iterations = 300usize;
    let mut used: Vec<usize> = Vec::new();
    let mut micro_used = false;
    let mut frontier_swap_tries: usize = 0;
    let max_frontier_swaps: usize = 2;
    let mut dirty_window = false;
    let mut n_rebuilds = 0usize;
    let max_rebuilds: usize = 2;

    loop {
        iterations += 1;
        if iterations > max_iterations { break; }
        if apply_best_add_windowed(state) { continue; }
        if apply_best_add_neigh_global(state) { dirty_window = true; continue; }
        used.clear();
        for &i in &state.window_core { if state.selected_bit[i] { used.push(i); } }
        let extra = state.window_locked.len().min(24);
        let start = state.window_locked.len().saturating_sub(extra);
        for &i in state.window_locked[start..].iter() { if state.selected_bit[i] { used.push(i); } }

        if apply_negative_edge_cleanup(state, &used) { dirty_window = true; continue; }

        if apply_best_swap_diff_reduce_windowed_cached(state, &used) { continue; }
        if apply_best_swap_diff_increase_windowed_cached(state, &used) { continue; }
        if apply_best_swap_neigh_any(state, &used) { dirty_window = true; continue; }
        if frontier_swap_tries < max_frontier_swaps {
            frontier_swap_tries += 1;
            if apply_best_swap_frontier_global(state, &used) { dirty_window = true; continue; }
        }
        if apply_best_remove2_add1_synergy_injection(state, &used) { dirty_window = true; continue; }
        if apply_reinsert10_greedy(state, &used) { dirty_window = true; continue; }
        if apply_best_exchange22_windowed(state, &used) { dirty_window = true; continue; }
        if apply_best_replace12_windowed(state, &used) { continue; }
        if apply_best_replace21_windowed(state, &used) { continue; }
        if dirty_window && n_rebuilds < max_rebuilds {
            n_rebuilds += 1; dirty_window = false; rebuild_windows(state); continue;
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
    let _ = params;
}

fn perturb_by_strategy(state: &mut State, rng: &mut Rng, strength: usize, stall_count: usize, strategy: usize) {
    let selected = state.selected_items();
    if selected.is_empty() { return; }
    let cap = state.ch.max_weight;
    let n = state.ch.num_items;

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
            let a = (rng.next_u32() as usize) % n;
            if state.selected_bit[a] { continue; }
            let wa = state.ch.weights[a];
            if wa == 0 || wa > cap { continue; }
            let mut s = (state.total_interactions[a] * 1000) / (wa as i64).max(1);
            s += (state.contrib[a] as i64) * 10;
            s += (rng.next_u32() & 0x3F) as i64;
            if best_a.map_or(true, |(bs, _, _)| s > bs) { best_a = Some((s, a, wa)); }
        }
        if let Some((_sa, a, wa)) = best_a {
            let row = unsafe { state.neigh.get_unchecked(a) };
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

    let (ta, tb, tw) = if let Some((a, b, w)) = target { (a, b, w) } else { (usize::MAX, usize::MAX, 0u32) };

    let mut need_w: u32 = 0;
    if ta != usize::MAX {
        let slack = state.slack();
        if tw > slack { need_w = tw - slack; }
        need_w = need_w.saturating_add(((strength as u32) + (stall_count as u32)).min(10));
    }

    let mut anchors: Vec<usize> = Vec::new();
    let mut is_anchor: Vec<bool> = Vec::new();
    if strategy == 5 {
        anchors = selected.clone();
        anchors.sort_unstable_by(|&a, &b| {
            state.support[b].cmp(&state.support[a])
                .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
                .then_with(|| b.cmp(&a))
        });
        if anchors.len() > MICRO_ADD_K { anchors.truncate(MICRO_ADD_K); }
        is_anchor = vec![false; n];
        for &a in &anchors {
            if a < n { is_anchor[a] = true; }
        }
    }

    let mut removal_candidates: Vec<(i64, usize, u32)> = Vec::with_capacity(selected.len());
    for &i in &selected {
        let w = state.ch.weights[i];
        if w == 0 { continue; }

        if strategy == 5 {
            if unsafe { *is_anchor.get_unchecked(i) } { continue; }
            let mut anchor_cnt: i64 = 0;
            let row = unsafe { state.neigh.get_unchecked(i) };
            for &(k, v) in row.iter() {
                if v <= 0 { continue; }
                let j = k as usize;
                if unsafe { *is_anchor.get_unchecked(j) } {
                    anchor_cnt += 1;
                }
            }
            let dens = (state.contrib[i] as i64 * 1000) / (w as i64).max(1);
            let s = dens
                - anchor_cnt.saturating_mul(1_000_000)
                - (state.support[i] as i64).saturating_mul(500);
            removal_candidates.push((s, i, w));
            continue;
        }

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
                let ca = state.contrib[a] as i64; let cb = state.contrib[b] as i64;
                let ta = state.total_interactions[a]; let tb = state.total_interactions[b];
                let adja = ca * BETA_DEN + BETA_NUM * (2 * ca - ta);
                let adjb = cb * BETA_DEN + BETA_NUM * (2 * cb - tb);
                let lhs = (adja as i128) * (wb as i128);
                let rhs = (adjb as i128) * (wa as i128);
                rhs.cmp(&lhs).then_with(|| state.support[b].cmp(&state.support[a]))
                    .then_with(|| tb.cmp(&ta)).then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        1 => {
            candidates.sort_unstable_by(|&a, &b| {
                state.ch.weights[a].cmp(&state.ch.weights[b]).then(state.contrib[b].cmp(&state.contrib[a]))
            });
        }
        2 => {
            candidates.sort_unstable_by_key(|&i| -(state.total_interactions[i] + (state.contrib[i] as i64) * 10));
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
                let sa = state.support[a] as i64; let sb = state.support[b] as i64;
                let wa = (state.ch.weights[a] as i64).max(1); let wb = (state.ch.weights[b] as i64).max(1);
                let ca = state.contrib[a] as i64; let cb = state.contrib[b] as i64;
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
    for _ in 0..2 {
        let mut added_any = false;
        for &i in &candidates {
            if state.selected_bit[i] { continue; }
            let w = state.ch.weights[i];
            if state.total_weight + w <= cap && (allow_zero || state.contrib[i] > 0) {
                state.add_item(i); added_any = true;
            }
        }
        if !added_any { break; }
    }

    let slack = state.slack();
    if slack >= 2 {
        let noise = if strategy == 0 { 0 } else { 0x0F };
        let allow_seed = slack >= 6;
        greedy_fill_with_beta(state, rng, noise, allow_seed);
    }
}

fn rebuild_from_solution_items(state: &mut State, items: &[usize]) {
    let n = state.ch.num_items;
    for i in 0..n {
        state.selected_bit[i] = false;
        state.contrib[i] = state.ch.values[i] as i32;
        state.support[i] = 0;
    }
    state.total_value = 0;
    state.total_weight = 0;
    for &i in items {
        state.add_item(i);
    }
}

fn elite_restart_perturb(state: &mut State, rng: &mut Rng, best_sel: &[usize], strength: usize, stall_count: usize, strategy: usize, effort: usize) {
    if best_sel.is_empty() { return; }
    rebuild_from_solution_items(state, best_sel);

    let base_remove = (best_sel.len() / 10).max(1);
    let adaptive_mult = 1 + (stall_count / 2);
    let strength_scaled = strength + (best_sel.len() / 40);
    let n_remove = (base_remove * adaptive_mult).min(strength_scaled).min(best_sel.len() / 3);
    if n_remove == 0 { return; }

    let mut guides: Vec<usize> = best_sel.to_vec();
    guides.sort_unstable_by(|&a, &b| {
        state.support[b].cmp(&state.support[a])
            .then_with(|| state.contrib[b].cmp(&state.contrib[a]))
            .then_with(|| b.cmp(&a))
    });
    if guides.len() > MICRO_ADD_K { guides.truncate(MICRO_ADD_K); }

    let n = state.ch.num_items;
    let mut is_guide = vec![false; n];
    for &g in &guides { is_guide[g] = true; }

    let mut rm_cand: Vec<(i64, i64, usize)> = Vec::with_capacity(best_sel.len());
    for &i in best_sel {
        let w = state.ch.weights[i];
        if w == 0 { continue; }
        if is_guide[i] { continue; }
        let mut neg_sum: i64 = 0;
        for &g in &guides {
            let v = state.ch.interaction_values[i][g] as i64;
            if v < 0 { neg_sum = neg_sum.saturating_sub(v); }
        }
        let sw = state.support[i] as i64;
        rm_cand.push((neg_sum, sw, i));
    }
    if rm_cand.is_empty() { return; }
    rm_cand.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));

    let mut removed = 0usize;
    for &(_ns, _sw, i) in &rm_cand {
        if removed >= n_remove { break; }
        if state.selected_bit[i] {
            state.remove_item(i);
            removed += 1;
        }
    }

    let recon_strategy = (strategy + effort) % 7;
    greedy_reconstruct(state, rng, recon_strategy);
}

fn run_one_instance(challenge: &Challenge, params: &Params) -> Solution {
    let n = challenge.num_items;
    let mut rng = Rng::from_seed(&challenge.seed);
    let (neigh, total_pre) = build_sparse_neighbors_and_totals(challenge);
    let team_est = (challenge.max_weight as usize) / 6;

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
    let hubs_static: Vec<usize> = hubs_all.into_iter().take(192usize.min(n)).map(|(_, i)| i).collect();

    let n_starts = params.n_starts(hard, team_est);

    let mut best: Option<State> = None;
    let mut second: Option<State> = None;

    for sid in 0..n_starts {
        let mut st = State::new_empty(challenge, &total_pre, &hubs_static, &neigh);
        match sid {
            0 => build_initial_solution(&mut st),
            1 => { construct_pair_seed_beta(&mut st, &mut rng); rebuild_windows(&mut st); }
            2 => { construct_forward_incremental(&mut st, 1, &mut rng); rebuild_windows(&mut st); }
            _ => {
                let m = if hard { 5 } else { 4 };
                construct_forward_incremental(&mut st, m, &mut rng);
                rebuild_windows(&mut st);
            }
        }
        dp_refinement(&mut st);
        rebuild_windows(&mut st);
        micro_qkp_refinement(&mut st);
        local_search_vnd(&mut st, params);

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
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, &neigh);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] && b2.selected_bit[i]
                        && hyb.total_weight + challenge.weights[i] <= challenge.max_weight
                    {
                        hyb.add_item(i);
                    }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);
            if hyb.total_value > best_new_val { best_new_val = hyb.total_value; best_new = Some(hyb); }
        }

        let (inter_cnt, union_cnt) = {
            let b1 = best.as_ref().unwrap();
            let b2 = second.as_ref().unwrap();
            let mut inter_cnt = 0usize;
            let mut union_cnt = 0usize;
            for i in 0..n {
                let a = b1.selected_bit[i]; let b = b2.selected_bit[i];
                if a || b { union_cnt += 1; }
                if a && b { inter_cnt += 1; }
            }
            (inter_cnt, union_cnt)
        };

        if union_cnt > 0 && (inter_cnt * 100) / union_cnt <= 85 {
            let mut hyb = State::new_empty(challenge, &total_pre, &hubs_static, &neigh);
            {
                let b1 = best.as_ref().unwrap();
                let b2 = second.as_ref().unwrap();
                for i in 0..n {
                    if b1.selected_bit[i] || b2.selected_bit[i] { hyb.add_item(i); }
                }
            }
            if hyb.total_weight > challenge.max_weight {
                while hyb.total_weight > challenge.max_weight {
                    let mut worst_item: Option<usize> = None;
                    let mut worst_score: i64 = i64::MAX;
                    for i in 0..n {
                        if !hyb.selected_bit[i] { continue; }
                        let c = hyb.contrib[i] as i64;
                        let w = challenge.weights[i] as i64;
                        let s = if w > 0 { (c * 1000) / w } else { c * 1000 };
                        if s < worst_score { worst_score = s; worst_item = Some(i); }
                    }
                    if let Some(wi) = worst_item { hyb.remove_item(wi); } else { break; }
                }
            }
            greedy_fill_with_beta(&mut hyb, &mut rng, 0, true);
            rebuild_windows(&mut hyb);
            dp_refinement(&mut hyb);
            rebuild_windows(&mut hyb);
            micro_qkp_refinement(&mut hyb);
            local_search_vnd(&mut hyb, params);
            if hyb.total_value > best_new_val { best_new = Some(hyb); }
        }

        if let Some(s) = best_new { best = Some(s); }
    }

    let mut state = best.unwrap();
    let mut best_sel: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n { if state.selected_bit[i] { best_sel.push(i); } }
    let mut best_val = state.total_value;
    let max_rounds = params.n_perturbation_rounds();
    let stall_limit = params.stall_limit_effective();
    let mut stall_count = 0;
    let mut elite_restart_used = false;

    for perturbation_round in 0..max_rounds {
        let is_last_round = perturbation_round >= max_rounds - 1;
        state.snap_bits.clone_from(&state.selected_bit);
        state.snap_contrib.clone_from(&state.contrib);
        state.snap_support.clone_from(&state.support);
        let prev_val = state.total_value;
        let prev_weight = state.total_weight;

        let apply_dp = !is_last_round && stall_count < 5;
        if apply_dp {
            rebuild_windows(&mut state);
            dp_refinement(&mut state);
            rebuild_windows(&mut state);
            micro_qkp_refinement(&mut state);
        }
        local_search_vnd(&mut state, params);

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
            if perturbation_round >= 7 && stall_count >= stall_limit {
                if elite_restart_used {
                    break;
                }
                elite_restart_used = true;
                let strategy = perturbation_round % 7;
                let strength = params.perturbation_strength_base() + (perturbation_round as usize) / 2;
                elite_restart_perturb(&mut state, &mut rng, &best_sel, strength, stall_count, strategy, params.effort);
                rebuild_windows(&mut state);
                dp_refinement(&mut state);
                rebuild_windows(&mut state);
                micro_qkp_refinement(&mut state);
                local_search_vnd(&mut state, params);
                if state.total_value > best_val {
                    best_val = state.total_value;
                    best_sel.clear();
                    for i in 0..n {
                        if state.selected_bit[i] {
                            if state.usage[i] < u16::MAX { state.usage[i] += 1; }
                            best_sel.push(i);
                        }
                    }
                }
                stall_count = 0;
                continue;
            }
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
            local_search_vnd(&mut state, params);
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

pub fn solve(challenge: &Challenge, hyperparameters: &Option<Map<String, Value>>) -> Solution {
    let params = Params::initialize(hyperparameters);
    run_one_instance(challenge, &params)
}
