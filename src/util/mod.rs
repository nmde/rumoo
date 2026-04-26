use rand::{make_rng, rngs::StdRng};

pub mod display;
pub mod dominator;
pub mod misc;
pub mod nds;
pub mod optimum;
pub mod sliding_window;

pub fn default_random_state() -> StdRng {
    make_rng()
}
