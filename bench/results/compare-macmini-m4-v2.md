# swadesh vs sacreBLEU: comparison report

2026-11-19 10:26:07 · Andrews-Mac-mini.local (arm64, 10 CPUs, threads=all) · Python 3.11.15 · numpy 2.4.6 · sacreBLEU 2.6.0 · swadesh 0.1.0 · load (2.23, 1.52, 1.35)

Times are the best of up to 5 repetitions (1-2 for slow cases). "Identical" means the full results (score objects, strings, CIs, p-values, matrices, CLI stdout) compare equal. The corpora cycle the WMT22 test set, so sacreBLEU's tokenizer memo cache hits often: speedups are conservative.


## corpus

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU tok=13a | 20,000 | 0.8750 | 0.0113 | **77.4×** | yes |
| BLEU tok=intl | 20,000 | 0.7699 | 0.0137 | **56.0×** | yes |
| BLEU tok=zh | 20,000 | 1.3705 | 0.0111 | **123.3×** | yes |
| BLEU tok=char | 20,000 | 1.2196 | 0.0122 | **100.1×** | yes |
| BLEU tok=none | 20,000 | 0.5034 | 0.00795 | **63.3×** | yes |
| BLEU lowercase (13a) | 20,000 | 0.8911 | 0.0181 | **49.1×** | yes |
| BLEU 2 references | 20,000 | 1.5775 | 0.0159 | **99.3×** | yes |
| chrF | 20,000 | 2.5852 | 0.0131 | **197.8×** | yes |
| chrF++ | 20,000 | 2.9332 | 0.0191 | **153.3×** | yes |
| chrF whitespace | 20,000 | 2.8781 | 0.0135 | **212.5×** | yes |
| chrF eps_smoothing | 20,000 | 2.5656 | 0.0129 | **199.0×** | yes |
| chrF lowercase | 20,000 | 2.5665 | 0.0202 | **127.3×** | yes |
| chrF zh | 20,000 | 1.0822 | 0.00681 | **158.8×** | yes |
| TER | 5,000 | 2.6898 | 0.0197 | **136.3×** | yes |
| TER normalized | 5,000 | 5.1297 | 0.0343 | **149.6×** | yes |
| TER case_sensitive | 5,000 | 2.6937 | 0.0151 | **178.3×** | yes |
| TER no_punct | 5,000 | 2.7431 | 0.0195 | **140.9×** | yes |
| TER normalized+asian_support zh | 5,000 | 1.9064 | 0.0239 | **79.7×** | yes |
| corpus_bleu() compat function | 20,000 | 0.9395 | 0.0107 | **88.2×** | yes |

## sentence

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU batch (sentence_scores vs loop) | 5,000 | 0.1367 | 0.00273 | **50.1×** | yes |
| BLEU sentence_score() in a loop | 5,000 | 0.1325 | 0.0272 | **4.9×** | yes |
| chrF batch (sentence_scores vs loop) | 5,000 | 0.5604 | 0.00278 | **201.8×** | yes |
| chrF sentence_score() in a loop | 5,000 | 0.5621 | 0.0268 | **21.0×** | yes |
| chrF++ batch (sentence_scores vs loop) | 5,000 | 0.6382 | 0.00426 | **149.7×** | yes |
| chrF++ sentence_score() in a loop | 5,000 | 0.6409 | 0.0381 | **16.8×** | yes |
| TER sentence_score() in a loop | 2,000 | 1.0596 | 0.0590 | **18.0×** | yes |
| sentence_bleu() compat function x1000 | 1,000 | 0.0574 | 0.00722 | **8.0×** | yes |

## significance

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| BLEU bootstrap CI (1000 resamples) | 5,000 | 0.3304 | 0.00787 | **42.0×** | yes |
| chrF bootstrap CI (1000 resamples) | 5,000 | 0.6962 | 0.00878 | **79.3×** | yes |
| TER bootstrap CI (200 resamples) | 2,000 | 1.0634 | 0.00810 | **131.3×** | yes |
| paired bootstrap, 3 systems, BLEU+chrF | 5,000 | 2.6663 | 0.1270 | **21.0×** | yes |
| paired approx. randomization (2000) | 5,000 | 3.7359 | 0.1327 | **28.2×** | yes |

## mbr

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| chrF 32x32 matrix | 1,024 | 0.0942 | 0.00104 | **90.8×** | yes |
| BLEU 32x32 matrix | 1,024 | 0.0205 | 0.00035 | **59.1×** | yes |
| chrF 128x128 matrix | 16,384 | 1.4569 | 0.00814 | **179.0×** | yes |
| BLEU 128x128 matrix | 16,384 | 0.3065 | 0.00158 | **193.8×** | yes |

## tokenize

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| tokenize 13a (cold cache, unique lines) | 1,896 | 0.0248 | 0.00065 | **38.1×** | yes |
| tokenize intl (cold cache, unique lines) | 1,978 | 0.0158 | 0.00061 | **25.8×** | yes |
| tokenize zh (cold cache, unique lines) | 1,895 | 0.0449 | 0.00062 | **72.7×** | yes |
| tokenize char (cold cache, unique lines) | 1,982 | 0.00270 | 0.00047 | **5.7×** | yes |

## cli

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| python CLI: -m bleu chrf | 20,000 | 3.5286 | 0.1141 | **30.9×** | yes |
| rust binary: -m bleu chrf | 20,000 | 3.5286 | 0.0388 | **90.9×** | yes |
| python CLI: -m ter | 20,000 | 9.8182 | 0.1671 | **58.8×** | yes |
| rust binary: -m ter | 20,000 | 9.7992 | 0.0837 | **117.1×** | yes |
| python CLI: -m bleu chrf ter -f text | 20,000 | 14.4199 | 0.1941 | **74.3×** | yes |
| rust binary: -m bleu chrf ter -f text | 20,000 | 14.4207 | 0.1037 | **139.0×** | yes |
| python CLI: --confidence (BLEU) | 20,000 | 1.2935 | 0.1242 | **10.4×** | yes |
| python CLI: --paired-bs, 2 systems | 20,000 | 7.3972 | 0.3691 | **20.0×** | yes |

## process

| case | n | sacreBLEU (s) | swadesh (s) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| interpreter + import | 1 | 0.0518 | 0.0719 | **0.7×** | yes |

## memory

| case | n | sacreBLEU (MB) | swadesh (MB) | ratio | identical |
|---|---:|---:|---:|---:|:---:|
| peak RSS MB: BLEU | 20,000 | 224 | 53 | **4.2×** | yes |
| peak RSS MB: CHRF | 20,000 | 866 | 56 | **15.6×** | yes |
| peak RSS MB: BLEU,n_bootstrap=1000 | 5,000 | 303 | 82 | **3.7×** | yes |

**Summary:** 48 timed comparisons; speedup min 4.9×, median 79.7×, max 212.5×; 0 non-identical results.

