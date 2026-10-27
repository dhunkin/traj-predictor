fn main() {
    use predictor::particle::predict;
    use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
    // Read simple.txt exactly like run_scenario
    let contents = std::fs::read_to_string("scenarios/simple.txt").unwrap();
    let mut observations: Vec<Observation> = Vec::new();
    let mut oois: Vec<Ooi> = Vec::new();
    let mut horizon_s = 10.0; let mut n_samples = 10000u32; let mut dt_s = 1.0;
    let mut p_turn = 0.3; let mut azimuth_bins = 72u32; let mut speed_noise = 0.5;
    let mut seed = 42u64;
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut it = line.split_whitespace();
        let d = it.next().unwrap_or("");
        match d {
            "ttc" => { let _vmin = it.next(); let _vmax = it.next(); let _wmax = it.next(); let _ = (it.next(), it.next(), it.next()); }
            "obs" => { let t = it.next().unwrap().parse::<f64>().unwrap(); let x = it.next().unwrap().parse::<f64>().unwrap(); let y = it.next().unwrap().parse::<f64>().unwrap();
                let cxx = it.next().unwrap().parse::<f64>().unwrap(); let cxy = it.next().unwrap().parse::<f64>().unwrap(); let cyy = it.next().unwrap().parse::<f64>().unwrap();
                observations.push(Observation { t, pos: [x, y], v: 0.0, psi: 0.0, cov: [cxx, cxy, cyy] }); }
            "ooi" => { let x = it.next().unwrap().parse::<f64>().unwrap(); let y = it.next().unwrap().parse::<f64>().unwrap(); oois.push(Ooi { pos: [x, y] }); }
            "config" => {
                horizon_s = it.next().unwrap().parse::<f64>().unwrap();
                n_samples = it.next().unwrap().parse::<u32>().unwrap();
                dt_s = it.next().unwrap().parse::<f64>().unwrap();
                p_turn = it.next().unwrap().parse::<f64>().unwrap();
                azimuth_bins = it.next().unwrap().parse::<u32>().unwrap();
                speed_noise = it.next().unwrap().parse::<f64>().unwrap();
                seed = it.next().unwrap().parse::<u64>().unwrap();
            }
            _ => {}
        }
    }
    let n = observations.len();
    let (px, py, pt) = (observations[n-2].pos[0], observations[n-2].pos[1], observations[n-2].t);
    let last = &mut observations[n-1];
    let dx = last.pos[0]-px; let dy = last.pos[1]-py; let dt = (last.t-pt).abs();
    last.psi = dy.atan2(dx); last.v = if dt>0.0 { (dx*dx+dy*dy).sqrt()/dt } else { 0.0 };
    let t = Ttc { v_min: 2.0, v_max: 8.0, a_lat: 9.0, a_long: 3.0, omega_cap: 1.0 };
    let c = PredictConfig { horizon_s, n_samples, dt_s, p_turn, azimuth_bins, speed_noise_mps: speed_noise, intent_hypothesis_prob: 0.8, seed };
    let p = predict(&observations, &t, &oois, &c);
    eprintln!("PROBE nz-ish s50={} s90={}", p.sector50, p.sector90);
}
