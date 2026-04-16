use rand::{make_rng, rngs::StdRng};

pub mod misc;

pub fn default_random_state() -> StdRng {
    make_rng()
}
