//! server.rs — client/server HTTP API for the predictor core.
//!
//! Moves the prediction + simulation computation from the browser (WASM)
//! into a Rust backend served over HTTP with JSON, so the frontend talks
//! to the server via `fetch`. The core library (`predictor`) stays
//! dependency-free: this binary is the only thing that pulls in axum,
//! tokio, serde and serde_json, and it converts its JSON DTOs to/from the
//! core types before calling `mc::predict` and `simulator::generate_curved_track`.
//!
//! Endpoints:
//!   GET  /health     -> "ok"
//!   POST /api/predict   -> PredictResponseDto
//!   POST /api/simulate  -> SimulateResponseDto
//!
//! Binds to 0.0.0.0:8081 (all interfaces) so the web demo on the LAN can reach
//! it. CORS is added by hand (no tower-http).
//!
//! This binary is only compiled when the `server` feature is enabled
//! (`required-features = ["server"]` in Cargo.toml), so it may use axum,
//! tokio, serde and serde_json without affecting the core lib or the wasm
//! build.
#![cfg(feature = "server")]

use axum::body::Body;
use axum::extract::Json;
use axum::http::{Request, Response, StatusCode};
use axum::middleware::{self, Next};
use axum::routing::{get, post};
use axum::Router;

use serde::{Deserialize, Serialize};

use predictor::filter::filter_track;
use predictor::mc::predict;
use predictor::simulator::{generate_curved_track, Trajectory};
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};

// ---- JSON DTOs -------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TtcDto {
    v_min: f64,
    v_max: f64,
    omega_max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OoiDto {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
struct ObservationDto {
    t: f64,
    x: f64,
    y: f64,
    v: f64,
    psi: f64,
    cov_xx: f64,
    cov_xy: f64,
    cov_yy: f64,
}

impl From<Observation> for ObservationDto {
    fn from(o: Observation) -> Self {
        ObservationDto {
            t: o.t,
            x: o.pos[0],
            y: o.pos[1],
            v: o.v,
            psi: o.psi,
            cov_xx: o.cov[0],
            cov_xy: o.cov[1],
            cov_yy: o.cov[2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictConfigDto {
    horizon_s: f64,
    n_samples: u32,
    dt_s: f64,
    p_turn: f64,
    azimuth_bins: u32,
    speed_noise_mps: f64,
    seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictRequestDto {
    ttc: TtcDto,
    // Stage 2: send the full track. The server runs filter::filter_track over
    // it and feeds the smoothed state to mc::predict(). Kept last_observation
    // optional for backward compatibility; if track is empty, last_observation
    // is used as a single-element track.
    #[serde(default)]
    track: Vec<ObservationDto>,
    #[serde(default)]
    last_observation: ObservationDto,
    oois: Vec<OoiDto>,
    config: PredictConfigDto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictResponseDto {
    sector50: f64,
    sector90: f64,
    heading_mean: f64,
    bin_centers: Vec<f64>,
    bin_probs: Vec<f64>,
    point_cloud: Vec<f64>, // flattened [x0, y0, x1, y1, ...]
    n_bins_nonzero: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulateRequestDto {
    trajectory_type: String,
    params: Vec<f64>,
    speed: f64,
    duration_s: f64,
    dt_obs_s: f64,
    cov_xx: f64,
    cov_xy: f64,
    cov_yy: f64,
    seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulateResponseDto {
    observations: Vec<ObservationDto>,
}

// ---- Handlers --------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn predict_handler(Json(req): Json<PredictRequestDto>) -> Result<Json<PredictResponseDto>, (StatusCode, String)> {
    // Map DTOs -> core types.
    let ttc = Ttc {
        v_min: req.ttc.v_min,
        v_max: req.ttc.v_max,
        omega_max: req.ttc.omega_max,
    };

    let last = Observation {
        t: req.last_observation.t,
        pos: [req.last_observation.x, req.last_observation.y],
        v: req.last_observation.v,
        psi: req.last_observation.psi,
        cov: [
            req.last_observation.cov_xx,
            req.last_observation.cov_xy,
            req.last_observation.cov_yy,
        ],
    };

    let oois: Vec<Ooi> = req
        .oois
        .iter()
        .map(|o| Ooi {
            pos: [o.x, o.y],
        })
        .collect();

    // Stage 2: build the full track from the request. If the client sends a
    // non-empty track, filter it and feed the smoothed state to predict();
    // otherwise fall back to the single last_observation.
    let track: Vec<Observation> = req
        .track
        .iter()
        .map(|o| Observation {
            t: o.t,
            pos: [o.x, o.y],
            v: o.v,
            psi: o.psi,
            cov: [o.cov_xx, o.cov_xy, o.cov_yy],
        })
        .collect();

    let smoothed = if track.len() > 1 {
        filter_track(&track)
    } else {
        last
    };

    let cfg = PredictConfig {
        horizon_s: req.config.horizon_s,
        n_samples: req.config.n_samples,
        dt_s: req.config.dt_s,
        p_turn: req.config.p_turn,
        azimuth_bins: req.config.azimuth_bins,
        speed_noise_mps: req.config.speed_noise_mps,
        seed: req.config.seed,
    };

    // Call the core Monte-Carlo engine with the smoothed state estimate.
    let prediction = predict(&smoothed, &ttc, &oois, &cfg);

    // Map result back to the response DTO.
    let point_cloud: Vec<f64> = prediction
        .point_cloud
        .iter()
        .flat_map(|p| [p[0], p[1]])
        .collect();
    let n_bins_nonzero = prediction.bin_probs.iter().filter(|&&p| p > 0.0).count();

    let resp = PredictResponseDto {
        sector50: prediction.sector50,
        sector90: prediction.sector90,
        heading_mean: prediction.final_heading_mean,
        bin_centers: prediction.bin_centers,
        bin_probs: prediction.bin_probs,
        point_cloud,
        n_bins_nonzero,
    };

    Ok(Json(resp))
}

async fn simulate_handler(Json(req): Json<SimulateRequestDto>) -> Result<Json<SimulateResponseDto>, (StatusCode, String)> {
    // Match the trajectory_type string to the core Trajectory enum.
    let p = &req.params;
    let get = |i: usize, name: &str| -> Result<f64, (StatusCode, String)> {
        p.get(i)
            .copied()
            .ok_or_else(|| (StatusCode::BAD_REQUEST, format!("missing params[{}] ({})", i, name)))
    };

    let traj = match req.trajectory_type.as_str() {
        "straight" => Trajectory::Straight {
            psi0: get(0, "psi0")?,
        },
        "spiral" => Trajectory::Spiral {
            r0: get(0, "r0")?,
            r_growth: get(1, "r_growth")?,
            turns: get(2, "turns")?,
        },
        "sine" => Trajectory::Sine {
            wavelength_m: get(0, "wavelength_m")?,
            amplitude_m: get(1, "amplitude_m")?,
            direction: get(2, "direction")?,
        },
        "zigzag" => Trajectory::Zigzag {
            segment_m: get(0, "segment_m")?,
            angle_rad: get(1, "angle_rad")?,
            baseline: get(2, "baseline")?,
        },
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("unknown trajectory_type: {}", other),
            ))
        }
    };

    // Call the core track generator.
    let observations = generate_curved_track(
        &traj,
        req.speed,
        req.duration_s,
        req.dt_obs_s,
        [req.cov_xx, req.cov_xy, req.cov_yy],
        req.seed,
    )
    .into_iter()
    .map(ObservationDto::from)
    .collect();

    Ok(Json(SimulateResponseDto { observations }))
}

// ---- Manual CORS middleware ------------------------------------------------

// Handle CORS preflight (OPTIONS) requests from the browser.
async fn preflight() -> Response<Body> {
    let mut resp = Response::new(Body::empty());
    resp.headers_mut().insert(
        "Access-Control-Allow-Origin",
        "*".parse().unwrap(),
    );
    resp.headers_mut().insert(
        "Access-Control-Allow-Methods",
        "GET, POST, OPTIONS".parse().unwrap(),
    );
    resp.headers_mut().insert(
        "Access-Control-Allow-Headers",
        "Content-Type".parse().unwrap(),
    );
    resp
}

async fn cors(req: Request<Body>, next: Next) -> Response<Body> {
    let mut resp = next.run(req).await;
    resp.headers_mut()
        .insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    resp
}

// ---- main ------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/health", get(health).options(preflight))
        .route("/api/predict", post(predict_handler).options(preflight))
        .route("/api/simulate", post(simulate_handler).options(preflight))
        .layer(middleware::from_fn(cors));

    // Bind to 0.0.0.0:8081 by default (all interfaces) so the web demo on the LAN
    // can reach it; override with the PORT env var if a different port is needed.
    let port: u16 = std::env::var("PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8081);
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind {} failed: {}", addr, e));
    println!("predictor server listening on http://{}", addr);
    axum::serve(listener, app).await.expect("axum::serve");
}
