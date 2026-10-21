//! prng.rs — xoshiro256**, splitmix64, gaussian.
//!
//! Deterministic pseudo-random generation with no OS randomness, no
//! `unsafe`, and no external crates. All RNG state lives in the struct.
//!
//! TODO(Stage2): (reserved)

/// splitmix64: expand a single u64 seed into a stream of u64 words.
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** — 4-word state, standard rotation-based transition.
#[derive(Debug, Clone)]
pub struct Xoshiro256ss {
    state: [u64; 4],
    /// Cached second Marsaglia polar sample (raw standard normal `v*factor`).
    gauss_cache: Option<f64>,
}

impl Xoshiro256ss {
    /// Build an RNG from a single u64 seed, expanded via splitmix64.
    pub fn from_seed(seed: u64) -> Self {
        let mut s = seed;
        let mut state = [0u64; 4];
        for word in &mut state {
            *word = splitmix64(&mut s);
        }
        Xoshiro256ss {
            state,
            gauss_cache: None,
        }
    }

    /// Next raw u64 from the standard xoshiro256** transition.
    pub fn next_u64(&mut self) -> u64 {
        let rotl = |x: u64, k: u32| (x << k) | (x >> (64 - k));
        let result = rotl(self.state[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = rotl(self.state[3], 45);
        result
    }

    /// Uniform f64 in [0, 1): top 53 bits of next_u64() / 2^53.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9007_1992_5474_0992.0
    }

    /// Gaussian sample via the Marsaglia polar method.
    ///
    /// The second of the two polar samples is cached in `Option<f64>` as a
    /// raw standard normal (`v * factor`); the caller's `mean`/`std` are
    /// applied (`mean + std * z`) on every return path, never baked into the
    /// cached value.
    pub fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        let z = if let Some(z) = self.gauss_cache.take() {
            z
        } else {
            let mut u;
            let mut v;
            let mut s;
            loop {
                u = 2.0 * self.next_f64() - 1.0;
                v = 2.0 * self.next_f64() - 1.0;
                s = u * u + v * v;
                if s > 0.0 && s < 1.0 {
                    break;
                }
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.gauss_cache = Some(v * factor);
            u * factor
        };
        mean + std * z
    }
}
