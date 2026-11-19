//! Randomness the rules ask for: who starts, and shuffles (the train cards at the start and at every
//! reshuffle of the discard pile, the destination tickets at the start).
//!
//! The rules call the generator at fixed points in a fixed order, so a seed and a list of moves
//! replay a game card for card.

pub trait GameRng {
    /// A number in `0..n`.
    fn below(&mut self, n: u32) -> u32;
    /// Shuffle `xs` in place.
    fn shuffle(&mut self, xs: &mut [u8]);
}

/// SplitMix64, used to seed and as a quick mixer.
#[inline]
pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ (Blackman and Vigna): fast, 256-bit state, more than enough for shuffling cards.
#[derive(Clone, Debug)]
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let s = [
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
            splitmix64(&mut x),
        ];
        Xoshiro { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// An independent generator (for a worker, a sampled world).
    pub fn fork(&mut self) -> Xoshiro {
        Xoshiro::new(self.next_u64())
    }
}

impl GameRng for Xoshiro {
    /// Lemire's multiply-and-reject: exactly uniform.
    #[inline]
    fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        let n64 = n as u64;
        loop {
            let m = (self.next_u64() >> 32) * n64;
            let low = m as u32;
            if low >= n.wrapping_neg() % n {
                return (m >> 32) as u32;
            }
        }
    }

    /// Fisher-Yates.
    fn shuffle(&mut self, xs: &mut [u8]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_is_in_range_and_covers_it() {
        let mut r = Xoshiro::new(7);
        let mut seen = [0u32; 5];
        for _ in 0..10_000 {
            seen[r.below(5) as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n > 1_800 && n < 2_200), "{seen:?}");
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Xoshiro::new(11);
        let mut xs: Vec<u8> = (0..110).map(|i| (i % 9) as u8).collect();
        let mut before = xs.clone();
        r.shuffle(&mut xs);
        let mut after = xs.clone();
        before.sort();
        after.sort();
        assert_eq!(before, after);
    }

    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let (mut a, mut b) = (Xoshiro::new(42), Xoshiro::new(42));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(Xoshiro::new(1).next_u64(), Xoshiro::new(2).next_u64());
    }
}
