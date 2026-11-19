//! ASCII 13a class-1 byte scan: the tightest inner loop of the tokeniser.
//! Four implementations, all counting the same bytes:
//! scalar (portable Rust, left to LLVM), SSE2 and AVX2 intrinsics, and the
//! shared hand-written AVX2 routine in `../../asm/scan_avx2.s`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Shared with the C side: the same assembly file is linked there.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
std::arch::global_asm!(include_str!("../../asm/scan_avx2.s"), options(att_syntax));

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
extern "C" {
    fn swadesh_scan_avx2(p: *const u8, nblocks: usize) -> u64;
}

#[inline]
pub fn class1(b: u8) -> bool {
    matches!(b, b'{'..=b'~' | b'['..=b'`' | b' '..=b'&' | b'('..=b'+' | b':'..=b'@' | b'/')
}

pub fn scan_scalar(p: &[u8]) -> u64 {
    p.iter().map(|&b| class1(b) as u64).sum()
}

const fn build_cls() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        t[i] = matches!(b, b'{'..=b'~' | b'['..=b'`' | b' '..=b'&' | b'('..=b'+' | b':'..=b'@' | b'/') as u8;
        i += 1;
    }
    t
}
static CLS: [u8; 256] = build_cls();

/// Scalar with a 256-entry class table (the strong portable scalar baseline).
pub fn scan_lut(p: &[u8]) -> u64 {
    p.iter().map(|&b| CLS[b as usize] as u64).sum()
}

// Bounds for signed compares: lo-1 and hi+1 per range.
const LO1: [i8; 6] = [0x7A, 0x5A, 0x1F, 0x27, 0x39, 0x2E];
const HI1: [i8; 6] = [0x7F, 0x61, 0x27, 0x2C, 0x41, 0x30];

#[cfg(target_arch = "x86_64")]
pub fn scan_sse2(p: &[u8]) -> u64 {
    let blocks = p.len() / 16;
    // SAFETY: SSE2 is part of the x86_64 baseline; all loads stay in bounds.
    unsafe {
        let lo: [__m128i; 6] = std::array::from_fn(|r| _mm_set1_epi8(LO1[r]));
        let hi: [__m128i; 6] = std::array::from_fn(|r| _mm_set1_epi8(HI1[r]));
        let mut c = 0u64;
        for b in 0..blocks {
            let x = _mm_loadu_si128(p.as_ptr().add(b * 16) as *const __m128i);
            let mut m = _mm_setzero_si128();
            for r in 0..6 {
                m = _mm_or_si128(m, _mm_and_si128(_mm_cmpgt_epi8(x, lo[r]), _mm_cmpgt_epi8(hi[r], x)));
            }
            c += _mm_movemask_epi8(m).count_ones() as u64;
        }
        c + scan_scalar(&p[blocks * 16..])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_avx2_inner(p: &[u8]) -> u64 {
    let blocks = p.len() / 32;
    let lo: [__m256i; 6] = std::array::from_fn(|r| _mm256_set1_epi8(LO1[r]));
    let hi: [__m256i; 6] = std::array::from_fn(|r| _mm256_set1_epi8(HI1[r]));
    let mut c = 0u64;
    for b in 0..blocks {
        let x = _mm256_loadu_si256(p.as_ptr().add(b * 32) as *const __m256i);
        let mut m = _mm256_setzero_si256();
        for r in 0..6 {
            m = _mm256_or_si256(m, _mm256_and_si256(_mm256_cmpgt_epi8(x, lo[r]), _mm256_cmpgt_epi8(hi[r], x)));
        }
        c += _mm256_movemask_epi8(m).count_ones() as u64;
    }
    c + scan_scalar(&p[blocks * 32..])
}

#[cfg(target_arch = "x86_64")]
pub fn scan_avx2(p: &[u8]) -> u64 {
    assert!(is_x86_feature_detected!("avx2"));
    // SAFETY: AVX2 availability checked above.
    unsafe { scan_avx2_inner(p) }
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
pub fn scan_asm(p: &[u8]) -> u64 {
    assert!(is_x86_feature_detected!("avx2"));
    let nb = p.len() / 32;
    // SAFETY: the routine reads exactly nb * 32 bytes, all inside `p`.
    unsafe { swadesh_scan_avx2(p.as_ptr(), nb) + scan_scalar(&p[nb * 32..]) }
}
