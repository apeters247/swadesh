#!/usr/bin/env bash
# Interleaved focused rerun (single-threaded, en-de): 3 rounds, each binary/variant
# measured once per round, so load drift affects all variants alike.
#   run_focused.sh INPUTS_DIR OUT_TSV C_BIN RS_BIN
set -euo pipefail
IN=$1; OUT=$2; C=$3; RS=$4
H=$IN/en-de.hyp.txt; R=$IN/en-de.ref.txt
printf "round\tload1\timpl\tkernel\tcount\tscan\tns_per_seg\tchecksum\n" > "$OUT"
one() {  # round impl-args...
    local rd=$1; shift
    local load; load=$(cut -d' ' -f1 /proc/loadavg)
    local line; line=$("$@")
    local ns chk impl kernel count scan
    ns=$(echo "$line" | grep -o 'ns_per_seg=[^ ]*' | cut -d= -f2)
    chk=$(echo "$line" | grep -o 'checksum=[^ ]*' | cut -d= -f2)
    impl=$(echo "$line" | grep -o 'impl=[^ ]*' | cut -d= -f2)
    kernel=$(echo "$line" | grep -o 'kernel=[^ ]*' | cut -d= -f2)
    count=$(echo "$line" | grep -o 'count=[^ ]*' | cut -d= -f2)
    scan=$(echo "$line" | grep -o 'scan=[^ ]*' | cut -d= -f2)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$rd" "$load" "$impl" "$kernel" "$count" "$scan" "$ns" "$chk" >> "$OUT"
    echo "round $rd $impl $kernel $count $scan -> $ns ns/seg (load $load)" >&2
}
for rd in 1 2 3; do
    one $rd "$C" --kernel tok --reps 3 --hyp "$H" --ref "$R"
    one $rd "$RS" --kernel tok --reps 3 --hyp "$H" --ref "$R"
    one $rd "$RS" --kernel tok --count ascii --reps 3 --hyp "$H" --ref "$R"
    for cnt in sort hash; do
        one $rd "$C" --kernel chr --count $cnt --reps 3 --hyp "$H" --ref "$R"
        one $rd "$RS" --kernel chr --count $cnt --reps 3 --hyp "$H" --ref "$R"
    done
    one $rd "$C" --kernel wrd --count sort --reps 3 --hyp "$H" --ref "$R"
    one $rd "$RS" --kernel wrd --count sort --reps 3 --hyp "$H" --ref "$R"
    for s in scalar lut sse2 avx2 asm; do
        one $rd "$C" --kernel scan --scan $s --reps 5 --hyp "$H" --ref "$R"
        one $rd "$RS" --kernel scan --scan $s --reps 5 --hyp "$H" --ref "$R"
    done
done
echo done >&2
