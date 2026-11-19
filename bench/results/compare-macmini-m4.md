# swadesh vs sacreBLEU: comparison report

2026-11-19 07:50:39 · Andrews-Mac-mini.local (arm64, 10 CPUs, threads=all) · Python 3.11.15 · numpy 2.4.6 · sacreBLEU 2.6.0 · swadesh 0.1.0 · load (2.11, 2.1, 1.85)

Times are the best of up to 5 repetitions (1-2 for slow cases). "Identical" means the full results (score objects, strings, CIs, p-values, matrices, CLI stdout) compare equal.


## corpus

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU tok=13a | 20,000 | 0.8805 | 0.0111 | **79.4×** | yes |
| BLEU tok=intl | 20,000 | 0.7897 | 0.0221 | **35.7×** | yes |
| BLEU tok=zh | 20,000 | 1.3755 | 0.0110 | **125.1×** | yes |
| BLEU tok=char | 20,000 | 1.2927 | 0.0119 | **108.7×** | yes |
| BLEU tok=none | 20,000 | 0.5162 | 0.00788 | **65.5×** | yes |
| BLEU lowercase (13a) | 20,000 | 0.8961 | 0.0179 | **50.1×** | yes |
| BLEU 2 references | 20,000 | 1.5931 | 0.0165 | **96.5×** | yes |
| chrF | 20,000 | 2.6267 | 0.0130 | **201.3×** | yes |
| chrF++ | 20,000 | 2.9477 | 0.0189 | **156.2×** | yes |
| chrF whitespace | 20,000 | 2.9169 | 0.0134 | **217.5×** | yes |
| chrF eps_smoothing | 20,000 | 2.5922 | 0.0128 | **202.1×** | yes |
| chrF lowercase | 20,000 | 2.5813 | 0.0203 | **127.0×** | yes |
| chrF zh | 20,000 | 1.1118 | 0.00665 | **167.2×** | yes |
| TER | 5,000 | 2.6913 | 0.0192 | **140.1×** | yes |
| TER normalized | 5,000 | 5.1757 | 0.0334 | **155.0×** | yes |
| TER case_sensitive | 5,000 | 2.6965 | 0.0151 | **178.3×** | yes |
| TER no_punct | 5,000 | 2.8286 | 0.0195 | **145.3×** | yes |
| TER normalized+asian_support zh | 5,000 | 1.8267 | 0.0230 | **79.5×** | yes |
| corpus_bleu() compat function | 20,000 | 1.0245 | 0.0103 | **99.9×** | yes |

## sentence

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU batch (sentence_scores vs loop) | 5,000 | 0.1350 | 0.00266 | **50.8×** | yes |
| BLEU sentence_score() in a loop | 5,000 | 0.1343 | 0.0268 | **5.0×** | yes |
| chrF batch (sentence_scores vs loop) | 5,000 | 0.5578 | 0.00267 | **209.3×** | yes |
| chrF sentence_score() in a loop | 5,000 | 0.5612 | 0.0267 | **21.0×** | yes |
| chrF++ batch (sentence_scores vs loop) | 5,000 | 0.6366 | 0.00430 | **147.9×** | yes |
| chrF++ sentence_score() in a loop | 5,000 | 0.6407 | 0.0378 | **17.0×** | yes |
| TER sentence_score() in a loop | 2,000 | 1.0580 | 0.0355 | **29.8×** | yes |
| sentence_bleu() compat function x1000 | 1,000 | 0.0571 | 0.00742 | **7.7×** | yes |

## significance

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU bootstrap CI (1000 resamples) | 5,000 | 1.2926 | 0.00852 | **151.6×** | yes |
| chrF bootstrap CI (1000 resamples) | 5,000 | 0.6739 | 0.0139 | **48.5×** | yes |
| TER bootstrap CI (200 resamples) | 2,000 | 1.1883 | 0.00809 | **146.8×** | yes |
| paired bootstrap, 3 systems, BLEU+chrF | 5,000 | 2.6821 | 0.1083 | **24.8×** | yes |
| paired approx. randomization (2000) | 5,000 | 3.7796 | 0.1539 | **24.6×** | yes |

## mbr

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| chrF 32x32 matrix | 1,024 | 0.0924 | 0.00099 | **93.2×** | yes |
| BLEU 32x32 matrix | 1,024 | 0.0206 | 0.00031 | **65.9×** | yes |
| chrF 128x128 matrix | 16,384 | 1.4672 | 0.00821 | **178.7×** | yes |
| BLEU 128x128 matrix | 16,384 | 0.3117 | 0.00175 | **178.4×** | yes |

## tokenize

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| tokenize 13a (cold cache) | 20,000 | 0.0254 | 0.00537 | **4.7×** | yes |
| tokenize intl (cold cache) | 20,000 | 0.0164 | 0.00918 | **1.8×** | yes |
| tokenize zh (cold cache) | 20,000 | 0.0454 | 0.00578 | **7.9×** | yes |
| tokenize char (cold cache) | 20,000 | 0.00350 | 0.00408 | **0.9×** | yes |

## cli

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| python CLI: -m bleu chrf | 20,000 | 3.8003 | 0.1166 | **32.6×** | yes |
| rust binary: -m bleu chrf | 20,000 | 3.5735 | 0.0416 | **85.8×** | yes |
| python CLI: -m ter | 20,000 | 9.9422 | 0.1692 | **58.8×** | yes |
| rust binary: -m ter | 20,000 | 9.9683 | 0.0907 | **109.9×** | yes |
| python CLI: -m bleu chrf ter -f text | 20,000 | 14.7536 | 0.1952 | **75.6×** | yes |
| rust binary: -m bleu chrf ter -f text | 20,000 | 14.6693 | 0.1105 | **132.7×** | yes |
| python CLI: --confidence (BLEU) | 20,000 | 1.3359 | 0.1516 | **8.8×** | yes |
| python CLI: --paired-bs, 2 systems | 20,000 | 7.7207 | 0.3193 | **24.2×** | yes |

## process

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| interpreter + import | 1 | 0.0513 | 0.0733 | **0.7×** | yes |

## memory

| case | n | sacreBLEU (MB) | swadesh (MB) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| peak RSS MB: BLEU | 20,000 | 224 | 54 | **4.2×** | yes |
| peak RSS MB: CHRF | 20,000 | 866 | 55 | **15.6×** | yes |
| peak RSS MB: BLEU,n_bootstrap=1000 | 5,000 | 303 | 82 | **3.7×** | yes |

**Summary:** 48 timed comparisons; speedup min 0.9×, median 85.8×, max 217.5×; 0 non-identical results.

