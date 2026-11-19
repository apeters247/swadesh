#!/usr/bin/env bash
# Build and run the Rust-vs-C-vs-asm micro-benchmarks.
#
#   bench/lang/run_bench.sh INPUTS_DIR RESULTS_TSV
#
# INPUTS_DIR holds en-de.{hyp,ref}.txt and zh-en.{hyp,ref}.txt from gen_inputs.py.
# Each measurement is appended to RESULTS_TSV with the 1-minute load average read
# just before it starts. Afterwards every (langpair, kernel) group must have one
# checksum across all implementations and variants, or the script fails.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
IN=${1:?inputs dir}
OUT=${2:?results tsv}
BIN=${BIN_DIR:-$(mktemp -d)}   # binaries go to a temp dir, not the repo
THREADS_MT=${THREADS_MT:-4}
REPS=${REPS:-5}

gcc -std=c11 -O3 -march=native -fopenmp -Wall -Wextra -o "$BIN/langbench_c" \
    "$HERE/c/langbench.c" "$HERE/asm/scan_avx2.s"
(cd "$HERE/rust" && CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2} cargo build --release --offline -q)
cp "$HERE/rust/target/release/langbench" "$BIN/langbench_rs"
C="$BIN/langbench_c"
RS="$BIN/langbench_rs"

printf "langpair\tload1\timpl\tkernel\tcount\tscan\tthreads\tsegments\tns_per_seg\tchecksum\treps_ns\n" > "$OUT"

run() {  # run LANGPAIR IMPL-BINARY ARGS...  -> one TSV row
    local lp=$1; shift
    local load; load=$(cut -d' ' -f1 /proc/loadavg)
    local line; line=$("$@")
    # parse key=value fields of the binary's output line
    local impl kernel count scan threads segs med chk reps
    impl=$(echo "$line" | grep -o 'impl=[^ ]*' | cut -d= -f2)
    kernel=$(echo "$line" | grep -o 'kernel=[^ ]*' | cut -d= -f2)
    count=$(echo "$line" | grep -o 'count=[^ ]*' | cut -d= -f2)
    scan=$(echo "$line" | grep -o 'scan=[^ ]*' | cut -d= -f2)
    threads=$(echo "$line" | grep -o 'threads=[^ ]*' | cut -d= -f2)
    segs=$(echo "$line" | grep -o 'segments=[^ ]*' | cut -d= -f2)
    med=$(echo "$line" | grep -o 'ns_per_seg=[^ ]*' | cut -d= -f2)
    chk=$(echo "$line" | grep -o 'checksum=[^ ]*' | cut -d= -f2)
    reps=$(echo "$line" | grep -o 'reps=[^ ]*' | cut -d= -f2)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$lp" "$load" "$impl" "$kernel" "$count" "$scan" "$threads" "$segs" "$med" "$chk" "$reps" >> "$OUT"
    echo "$lp $impl $kernel count=$count scan=$scan T=$threads -> ${med} ns/seg (load $load)" >&2
}

for LP in en-de zh-en; do
    H="$IN/$LP.hyp.txt"; R="$IN/$LP.ref.txt"
    # single-threaded
    for B in "$C" "$RS"; do
        run "$LP" "$B" --kernel tok --reps $REPS --hyp "$H" --ref "$R"
        for CNT in sort hash; do
            run "$LP" "$B" --kernel chr --count $CNT --reps $REPS --hyp "$H" --ref "$R"
            run "$LP" "$B" --kernel wrd --count $CNT --reps $REPS --hyp "$H" --ref "$R"
        done
        for S in scalar sse2 avx2 asm; do
            run "$LP" "$B" --kernel scan --scan $S --reps $REPS --hyp "$H" --ref "$R"
        done
    done
    # Rust ablation: allocation pattern of the current swadesh-core code
    run "$LP" "$RS" --kernel tok --count alloc --reps $REPS --hyp "$H" --ref "$R"
    run "$LP" "$RS" --kernel chr --count alloc --reps $REPS --hyp "$H" --ref "$R"
    run "$LP" "$RS" --kernel wrd --count alloc --reps $REPS --hyp "$H" --ref "$R"
    # multithreaded (THREADS_MT workers, both languages)
    for B in "$C" "$RS"; do
        run "$LP" "$B" --kernel tok --threads $THREADS_MT --reps $REPS --hyp "$H" --ref "$R"
        run "$LP" "$B" --kernel chr --count sort --threads $THREADS_MT --reps $REPS --hyp "$H" --ref "$R"
        run "$LP" "$B" --kernel wrd --count sort --threads $THREADS_MT --reps $REPS --hyp "$H" --ref "$R"
        run "$LP" "$B" --kernel scan --scan avx2 --threads $THREADS_MT --reps $REPS --hyp "$H" --ref "$R"
        run "$LP" "$B" --kernel scan --scan asm --threads $THREADS_MT --reps $REPS --hyp "$H" --ref "$R"
    done
done

# Parity gate: one checksum per (langpair, kernel, threads) across every implementation
# and variant (sort/hash/alloc counting, the four scan kinds, C and Rust).
awk -F'\t' 'NR>1 {k=$1"|"$4"|"$7; if(!(k in s)) s[k]=$10; else if (s[k]!=$10) {bad=1; print "MISMATCH " k " " s[k] " vs " $10 > "/dev/stderr"}} END{exit bad}' "$OUT"
echo "parity: all checksums agree per (langpair, kernel, threads)" >&2
