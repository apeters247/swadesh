# swadesh cross-machine benchmark (SUMMARY)

Hypotheses are synthetic perturbations of WMT22 en-de references (word deletions, swaps, and reversals, fixed seed 1234), not real MT output. Each swadesh result is checked for `==` equality with sacreBLEU 2.6.0 on the same machine. Times are best of 3 (corpus cases) or 1 (sentence and bootstrap cases). Speedup = sacrebleu s / swadesh s, truncated (not rounded) to one decimal place.

## troy-1 (x86_64, Linux)

```
machine: troy1
host: kpeters-X870-AORUS-ELITE-WIFI7  os: Linux 7.0.0-38-generic x86_64
CPU(s):                                  24
Model name:                              AMD Ryzen 9 7900X 12-Core Processor
CPU(s) scaling MHz:                      71%
python: 3.11.18 (main, Nov 11 2026, 00:16:37) [Clang 23.1.3 ]  sacrebleu 2.6.0  numpy 2.4.6
free disk before: /dev/nvme0n1p2  1.8T  1.5T  261G  86% /

== 20k
uptime:  23:49:47 up 28 days,  6:02,  5 users,  load average: 1.34, 2.46, 2.49
start: 2026-11-19T04:49:47Z
end: 2026-11-19T04:50:24Z
uptime after:  23:50:24 up 28 days,  6:02,  5 users,  load average: 1.83, 2.46, 2.48

== 100k
uptime:  23:50:24 up 28 days,  6:02,  5 users,  load average: 1.83, 2.46, 2.48
start: 2026-11-19T04:50:24Z
end: 2026-11-19T04:53:54Z
uptime after:  23:53:54 up 28 days,  6:06,  5 users,  load average: 3.84, 2.91, 2.64

== 20k-1thread
uptime:  23:53:54 up 28 days,  6:06,  5 users,  load average: 3.84, 2.91, 2.64
start: 2026-11-19T04:53:54Z
end: 2026-11-19T04:54:35Z
uptime after:  23:54:35 up 28 days,  6:07,  5 users,  load average: 6.65, 3.76, 2.94
DONE
```

### troy1 20k (20000 lines, cpus=24)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.735 | 0.0176 | 98.5x | yes |
| chrF corpus | 3.695 | 0.0141 | 261.2x | yes |
| chrF++ corpus | 4.303 | 0.0139 | 309.3x | yes |
| sentence BLEU x20000 | 1.013 | 0.0095 | 106.8x | yes |
| sentence chrF x20000 | 2.828 | 0.0095 | 297.8x | yes |
| BLEU + bootstrap(1000) x5000 | 0.870 | 0.0711 | 12.2x | yes |

### troy1 100k (100000 lines, cpus=24)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 8.439 | 0.0709 | 119.0x | yes |
| chrF corpus | 27.961 | 0.0825 | 338.9x | yes |
| chrF++ corpus | 29.840 | 0.0845 | 353.0x | yes |
| sentence BLEU x20000 | 1.122 | 0.0112 | 100.3x | yes |
| sentence chrF x20000 | 3.071 | 0.0144 | 213.4x | yes |

### troy1 20k-1thread (20000 lines, cpus=24)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.472 | 0.0664 | 22.1x | yes |
| chrF corpus | 4.605 | 0.0845 | 54.4x | yes |
| chrF++ corpus | 5.186 | 0.1244 | 41.6x | yes |
| sentence BLEU x20000 | 1.052 | 0.0697 | 15.0x | yes |
| sentence chrF x20000 | 2.862 | 0.0853 | 33.5x | yes |
| BLEU + bootstrap(1000) x5000 | 0.715 | 0.0631 | 11.3x | yes |

## GX10 (aarch64, Linux)

```
machine: gx10
host: gx10-b61b  os: Linux 6.17.0-1021-nvidia aarch64
CPU(s):                                  20
Model name:                              Cortex-X925
CPU(s) scaling MHz:                      100%
Model name:                              Cortex-A725
CPU(s) scaling MHz:                      100%
python: 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]  sacrebleu 2.6.0  numpy 2.5.5
free disk before: /dev/nvme0n1p2  916G  351G  519G  41% /

== 20k
uptime:  23:49:48 up 10 days,  4:03,  2 users,  load average: 2.03, 2.23, 2.08
start: 2026-11-19T04:49:48Z
end: 2026-11-19T04:50:35Z
uptime after:  23:50:35 up 10 days,  4:04,  2 users,  load average: 2.10, 2.22, 2.08

== 100k
uptime:  23:50:35 up 10 days,  4:04,  2 users,  load average: 2.10, 2.22, 2.08
start: 2026-11-19T04:50:35Z
end: 2026-11-19T04:54:33Z
uptime after:  23:54:33 up 10 days,  4:08,  2 users,  load average: 4.05, 2.88, 2.36

== 20k-1thread
uptime:  23:54:33 up 10 days,  4:08,  2 users,  load average: 4.05, 2.88, 2.36
start: 2026-11-19T04:54:33Z
end: 2026-11-19T04:55:18Z
uptime after:  23:55:18 up 10 days,  4:09,  2 users,  load average: 3.54, 2.91, 2.39
DONE
```

### gx10 20k (20000 lines, cpus=20)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.793 | 0.0332 | 54.0x | yes |
| chrF corpus | 5.493 | 0.0147 | 373.7x | yes |
| chrF++ corpus | 6.127 | 0.0490 | 125.1x | yes |
| sentence BLEU x20000 | 0.897 | 0.0341 | 26.3x | yes |
| sentence chrF x20000 | 2.872 | 0.0121 | 237.1x | yes |
| BLEU + bootstrap(1000) x5000 | 0.684 | 0.0467 | 14.6x | yes |

### gx10 100k (100000 lines, cpus=20)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 9.743 | 0.1668 | 58.3x | yes |
| chrF corpus | 29.881 | 0.0788 | 379.0x | yes |
| chrF++ corpus | 35.054 | 0.2381 | 147.1x | yes |
| sentence BLEU x20000 | 0.888 | 0.0353 | 25.1x | yes |
| sentence chrF x20000 | 2.963 | 0.0138 | 214.5x | yes |

### gx10 20k-1thread (20000 lines, cpus=20)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.348 | 0.0819 | 16.4x | yes |
| chrF corpus | 4.657 | 0.0711 | 65.5x | yes |
| chrF++ corpus | 6.369 | 0.1099 | 57.9x | yes |
| sentence BLEU x20000 | 0.899 | 0.0832 | 10.7x | yes |
| sentence chrF x20000 | 2.985 | 0.0753 | 39.6x | yes |
| BLEU + bootstrap(1000) x5000 | 0.751 | 0.0762 | 9.8x | yes |

## Mac mini (Apple M4, macOS)

```
machine: macmini
host: Andrews-Mac-mini.local  os: Darwin 25.5.0 arm64
Apple M4
10
python: 3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]  sacrebleu 2.6.0  numpy 2.4.6
free disk before: /dev/disk3s5   228Gi    96Gi    93Gi    51%    910k  978M    0%   /System/Volumes/Data

== 20k
uptime: 23:49  up 5 days,  7:24, 2 users, load averages: 3.10 3.51 3.93
start: 2026-11-19T04:49:48Z
end: 2026-11-19T04:50:39Z
uptime after: 23:50  up 5 days,  7:25, 2 users, load averages: 3.04 3.43 3.87

== 100k
uptime: 23:50  up 5 days,  7:25, 2 users, load averages: 3.04 3.43 3.87
start: 2026-11-19T04:50:39Z
end: 2026-11-19T04:54:44Z
uptime after: 23:54  up 5 days,  7:29, 2 users, load averages: 3.90 3.64 3.84

== 20k-1thread
uptime: 23:54  up 5 days,  7:29, 2 users, load averages: 3.90 3.64 3.84
start: 2026-11-19T04:54:44Z
end: 2026-11-19T04:55:35Z
uptime after: 23:55  up 5 days,  7:30, 2 users, load averages: 4.39 3.83 3.90
DONE
```

### macmini 20k (20000 lines, cpus=10)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.845 | 0.0233 | 79.1x | yes |
| chrF corpus | 5.109 | 0.0245 | 208.8x | yes |
| chrF++ corpus | 6.293 | 0.0373 | 168.6x | yes |
| sentence BLEU x20000 | 1.639 | 0.0275 | 59.6x | yes |
| sentence chrF x20000 | 4.733 | 0.0211 | 224.6x | yes |
| BLEU + bootstrap(1000) x5000 | 0.735 | 0.0462 | 15.9x | yes |

### macmini 100k (100000 lines, cpus=10)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 8.980 | 0.1622 | 55.3x | yes |
| chrF corpus | 27.261 | 0.1772 | 153.8x | yes |
| chrF++ corpus | 36.305 | 0.2438 | 148.9x | yes |
| sentence BLEU x20000 | 1.908 | 0.0236 | 80.9x | yes |
| sentence chrF x20000 | 4.216 | 0.0174 | 242.1x | yes |

### macmini 20k-1thread (20000 lines, cpus=10)

| case | sacrebleu s | swadesh s | speedup | exact |
|---|---|---|---|---|
| BLEU corpus | 1.585 | 0.1086 | 14.5x | yes |
| chrF corpus | 4.496 | 0.1483 | 30.3x | yes |
| chrF++ corpus | 5.255 | 0.1818 | 28.9x | yes |
| sentence BLEU x20000 | 1.507 | 0.1074 | 14.0x | yes |
| sentence chrF x20000 | 4.251 | 0.1139 | 37.3x | yes |
| BLEU + bootstrap(1000) x5000 | 0.619 | 0.0907 | 6.8x | yes |

