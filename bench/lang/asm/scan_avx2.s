# swadesh_scan_avx2(const uint8_t *p, size_t nblocks) -> uint64_t
#
# Counts bytes that are 13a class-1 characters over nblocks * 32 bytes, i.e.
# the byte ranges 0x7B-0x7E, 0x5B-0x60, 0x20-0x26, 0x28-0x2B, 0x3A-0x40, 0x2F.
# Bytes are compared as signed int8 with lo-1 < x < hi+1. Bytes >= 0x80 are
# negative, so they never match. Register plan (all caller-saved in SysV):
#   ymm0 data, ymm1/ymm2 temporaries, ymm3 accumulator,
#   ymm4..ymm9 = lo-1 per range, ymm10..ymm15 = hi+1 per range.
# System V AMD64: rdi = p, rsi = nblocks, return value in rax.
# Assemble with GNU as (AT&T syntax). Requires AVX2 and POPCNT.

        .text
        .globl  swadesh_scan_avx2
        .type   swadesh_scan_avx2, @function
swadesh_scan_avx2:
        xorl    %eax, %eax              # rax = running total
        testq   %rsi, %rsi
        jz      1f

        # Broadcast the twelve range bounds.
        movl    $0x7A, %r8d;  vmovd %r8d, %xmm4;   vpbroadcastb %xmm4, %ymm4
        movl    $0x5A, %r8d;  vmovd %r8d, %xmm5;   vpbroadcastb %xmm5, %ymm5
        movl    $0x1F, %r8d;  vmovd %r8d, %xmm6;   vpbroadcastb %xmm6, %ymm6
        movl    $0x27, %r8d;  vmovd %r8d, %xmm7;   vpbroadcastb %xmm7, %ymm7
        movl    $0x39, %r8d;  vmovd %r8d, %xmm8;   vpbroadcastb %xmm8, %ymm8
        movl    $0x2E, %r8d;  vmovd %r8d, %xmm9;   vpbroadcastb %xmm9, %ymm9
        movl    $0x7F, %r8d;  vmovd %r8d, %xmm10;  vpbroadcastb %xmm10, %ymm10
        movl    $0x61, %r8d;  vmovd %r8d, %xmm11;  vpbroadcastb %xmm11, %ymm11
        movl    $0x27, %r8d;  vmovd %r8d, %xmm12;  vpbroadcastb %xmm12, %ymm12
        movl    $0x2C, %r8d;  vmovd %r8d, %xmm13;  vpbroadcastb %xmm13, %ymm13
        movl    $0x41, %r8d;  vmovd %r8d, %xmm14;  vpbroadcastb %xmm14, %ymm14
        movl    $0x30, %r8d;  vmovd %r8d, %xmm15;  vpbroadcastb %xmm15, %ymm15

        .p2align 4
2:      vmovdqu (%rdi), %ymm0
        vpxor   %ymm3, %ymm3, %ymm3

        # range r: acc |= (x > lo-1) & (hi+1 > x)
        vpcmpgtb %ymm4,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm10, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpcmpgtb %ymm5,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm11, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpcmpgtb %ymm6,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm12, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpcmpgtb %ymm7,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm13, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpcmpgtb %ymm8,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm14, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpcmpgtb %ymm9,  %ymm0, %ymm1
        vpcmpgtb %ymm0,  %ymm15, %ymm2
        vpand   %ymm2, %ymm1, %ymm1
        vpor    %ymm1, %ymm3, %ymm3

        vpmovmskb %ymm3, %r8d
        popcntl %r8d, %r8d
        addq    %r8, %rax
        addq    $32, %rdi
        decq    %rsi
        jnz     2b
        vzeroupper
1:      ret
        .size   swadesh_scan_avx2, .-swadesh_scan_avx2
        .section .note.GNU-stack,"",@progbits
