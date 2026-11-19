/*
 * langbench.c - C11 implementation of the swadesh hot kernels, for the
 * Rust-vs-C-vs-asm comparison in docs/language-choice.md.
 *
 *   tok   sacreBLEU 13a tokenizer (pre-steps, class substitution, three
 *         fixed two-character rewrite passes, whitespace normalisation)
 *   chr   chrF character n-gram statistics, orders 1..6 (hyp vs one ref)
 *   wrd   BLEU word n-gram statistics, orders 1..4 (hyp vs one ref)
 *   scan  ASCII 13a class-1 byte scan (the tightest inner loop of tok)
 *
 * Counting is either sort + run-length (--count sort) or open addressing
 * (--count hash). Keys are packed exactly in unsigned __int128.
 *
 * Build: gcc -std=c11 -O3 -march=native -fopenmp langbench.c ../asm/scan_avx2.s
 * Run:   langbench --kernel tok|chr|wrd|scan --count sort|hash
 *                  --scan scalar|sse2|avx2|asm --threads T --reps R
 *                  --hyp HYP --ref REF
 *
 * Prints one line: impl, kernel, config, median ns per segment, checksum.
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef uint32_t u32;
typedef uint64_t u64;

/* Assembly kernel (asm/scan_avx2.s): counts class-1 bytes over full 32-byte blocks. */
extern u64 swadesh_scan_avx2(const unsigned char *p, size_t nblocks);

#define ENSURE(ptr, cap, need)                                  \
    do {                                                        \
        if ((size_t)(need) > (cap)) {                           \
            free(ptr);                                          \
            (cap) = (size_t)(need) * 2 + 16;                    \
            (ptr) = malloc((cap) * sizeof *(ptr));              \
            if (!(ptr)) { perror("malloc"); exit(2); }         \
        }                                                       \
    } while (0)

/* ------------------------------------------------------------- timing / input */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    char *buf;
    size_t total;
    size_t *off, *len;
    size_t n;
} Lines;

static Lines load_lines(const char *path) {
    Lines L = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    L.buf = malloc(sz + 1);
    if (fread(L.buf, 1, sz, f) != (size_t)sz) { perror("read"); exit(2); }
    fclose(f);
    L.buf[sz] = 0;
    L.total = sz;
    size_t n = 0;
    for (long i = 0; i < sz; i++) n += L.buf[i] == '\n';
    if (sz && L.buf[sz - 1] != '\n') n++;
    L.off = malloc(n * sizeof *L.off);
    L.len = malloc(n * sizeof *L.len);
    size_t k = 0, start = 0;
    for (long i = 0; i < sz; i++) {
        if (L.buf[i] == '\n') {
            L.off[k] = start;
            L.len[k++] = i - start;
            start = i + 1;
        }
    }
    if (start < (size_t)sz) {
        L.off[k] = start;
        L.len[k++] = sz - start;
    }
    L.n = k;
    return L;
}

static inline const char *line_ptr(const Lines *L, size_t i) { return L->buf + L->off[i]; }

/* ------------------------------------------------------------- UTF-8 / Unicode */

static size_t utf8_decode(const unsigned char *s, size_t n, u32 *o) {
    size_t k = 0, i = 0;
    while (i < n) {
        unsigned c = s[i];
        if (c < 0x80) { o[k++] = c; i += 1; }
        else if ((c & 0xE0) == 0xC0) { o[k++] = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); i += 2; }
        else if ((c & 0xF0) == 0xE0) {
            o[k++] = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            i += 3;
        } else {
            o[k++] = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) |
                     (s[i + 3] & 0x3F);
            i += 4;
        }
    }
    return k;
}

static inline size_t utf8_put(u32 c, unsigned char *o) {
    if (c < 0x80) { o[0] = (unsigned char)c; return 1; }
    if (c < 0x800) { o[0] = 0xC0 | (c >> 6); o[1] = 0x80 | (c & 0x3F); return 2; }
    if (c < 0x10000) {
        o[0] = 0xE0 | (c >> 12); o[1] = 0x80 | ((c >> 6) & 0x3F); o[2] = 0x80 | (c & 0x3F);
        return 3;
    }
    o[0] = 0xF0 | (c >> 18); o[1] = 0x80 | ((c >> 12) & 0x3F);
    o[2] = 0x80 | ((c >> 6) & 0x3F); o[3] = 0x80 | (c & 0x3F);
    return 4;
}

/* Python str.isspace() (what str.split()/strip() use). */
static inline int is_py_space(u32 c) {
    if (c < 128) return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f);
    return c == 0x85 || c == 0xa0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200a) ||
           c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

/* ------------------------------------------------------------- 13a tokenizer */

typedef struct {
    u32 *a, *b, *c;          size_t acap, bcap, ccap;  /* code point work buffers */
    unsigned char *ob;       size_t obcap;             /* UTF-8 output */
    char *s1, *s2;           size_t s1cap, s2cap;      /* string pre-step buffers */
    u32 *cph, *cpr, *cpf;    size_t cphcap, cprcap, cpfcap;  /* chrF code points */
    u32 *hid, *rid;          size_t hidcap, ridcap;    /* interned word ids */
    u128 *kh, *kr;           size_t khcap, krcap;      /* packed keys (sort) */
    u128 *rh_k, *rr_k;       u32 *rh_c, *rr_c;         /* run-length (sort) */
    size_t rhcap, rrcap;
    u128 *hk; u32 *hv; size_t hcap; size_t hsz;     /* open-addressing (hash) */
    u128 *rk; u32 *rv; size_t rcap; size_t rsz;
    struct { const char *p; u32 len, id; } *it; size_t itcap, itsz; /* word intern */
} Scratch;

static void scratch_free(Scratch *S) {
    free(S->a); free(S->b); free(S->c); free(S->ob); free(S->s1); free(S->s2);
    free(S->cph); free(S->cpr); free(S->cpf); free(S->hid); free(S->rid);
    free(S->kh); free(S->kr); free(S->rh_k); free(S->rr_k); free(S->rh_c); free(S->rr_c);
    free(S->hk); free(S->hv); free(S->rk); free(S->rv); free(S->it);
    memset(S, 0, sizeof *S);
}

static inline int isdig(u32 c) { return c - '0' < 10u; }

static inline int is_13a_class1(u32 c) {
    return (c >= '{' && c <= '~') || (c >= '[' && c <= '`') || (c >= ' ' && c <= '&') ||
           (c >= '(' && c <= '+') || (c >= ':' && c <= '@') || c == '/';
}

/* s[0..n] with every occurrence of pat replaced by rep; returns new length in d. */
static size_t replace_all(const char *s, size_t n, const char *pat, size_t pl, const char *rep,
                          size_t rl, char *d) {
    size_t k = 0;
    const char *p = s, *end = s + n;
    for (;;) {
        const char *m = memmem(p, (size_t)(end - p), pat, pl);
        if (!m) break;
        memcpy(d + k, p, (size_t)(m - p)); k += (size_t)(m - p);
        memcpy(d + k, rep, rl); k += rl;
        p = m + pl;
    }
    memcpy(d + k, p, (size_t)(end - p)); k += (size_t)(end - p);
    return k;
}

/* sacreBLEU 13a pre-steps. Each rule runs only when its pattern occurs. */
static size_t prep13a(const char *in, size_t n, Scratch *S, const char **outp) {
    ENSURE(S->s1, S->s1cap, n + 1);
    ENSURE(S->s2, S->s2cap, n + 1);
    static const struct { const char *p; size_t pl; const char *r; size_t rl; } rules[] = {
        {"<skipped>", 9, "", 0}, {"-\n", 2, "", 0}, {"\n", 1, " ", 1},
        {"&quot;", 6, "\"", 1}, {"&amp;", 5, "&", 1}, {"&lt;", 4, "<", 1}, {"&gt;", 4, ">", 1},
    };
    const char *cur = in;
    size_t len = n;
    char *bufs[2] = {S->s1, S->s2};
    int w = 0;
    for (size_t r = 0; r < sizeof rules / sizeof rules[0]; r++) {
        if (r == 3 && !memchr(cur, '&', len)) continue;
        if (!memmem(cur, len, rules[r].p, rules[r].pl)) continue;
        len = replace_all(cur, len, rules[r].p, rules[r].pl, rules[r].r, rules[r].rl, bufs[w]);
        cur = bufs[w];
        w ^= 1;
    }
    *outp = cur;
    return len;
}

/* Tokenise one segment. Output UTF-8 in S->ob, returns its length. */
static size_t tok13a(const char *line, size_t n, Scratch *S) {
    const char *txt;
    size_t len = prep13a(line, n, S, &txt);
    size_t m = len + 2;                       /* ' ' + text + ' ' */
    ENSURE(S->a, S->acap, m + 1);
    /* sub1 triples, then each two-character pass at most doubles: 3m -> 6m -> 12m; c <= 24m. */
    ENSURE(S->b, S->bcap, 24 * m + 16);
    ENSURE(S->c, S->ccap, 24 * m + 16);
    u32 *a = S->a, *b = S->b, *c = S->c;
    a[0] = ' ';
    size_t k = utf8_decode((const unsigned char *)txt, len, a + 1);
    a[k + 1] = ' ';
    m = k + 2;

    /* (\s-free class substitution) X -> ' X ' */
    size_t nb = 0;
    for (size_t i = 0; i < m; i++) {
        if (is_13a_class1(a[i])) { b[nb++] = ' '; b[nb++] = a[i]; b[nb++] = ' '; }
        else b[nb++] = a[i];
    }
    /* ([^0-9])([\.,]) -> \1 \2 */
    size_t nc = 0;
    for (size_t i = 0; i < nb;) {
        if (i + 1 < nb && !isdig(b[i]) && (b[i + 1] == '.' || b[i + 1] == ',')) {
            c[nc++] = b[i]; c[nc++] = ' '; c[nc++] = b[i + 1]; c[nc++] = ' '; i += 2;
        } else c[nc++] = b[i++];
    }
    /* ([\.,])([^0-9]) -> _\1 \2 */
    nb = 0;
    for (size_t i = 0; i < nc;) {
        if (i + 1 < nc && (c[i] == '.' || c[i] == ',') && !isdig(c[i + 1])) {
            b[nb++] = ' '; b[nb++] = c[i]; b[nb++] = ' '; b[nb++] = c[i + 1]; i += 2;
        } else b[nb++] = c[i++];
    }
    /* ([0-9])(-) -> \1 _\2 */
    nc = 0;
    for (size_t i = 0; i < nb;) {
        if (i + 1 < nb && isdig(b[i]) && b[i + 1] == '-') {
            c[nc++] = b[i]; c[nc++] = ' '; c[nc++] = '-'; c[nc++] = ' '; i += 2;
        } else c[nc++] = b[i++];
    }
    /* ' '.join(text.split()) */
    ENSURE(S->ob, S->obcap, 4 * nc + 4);
    size_t o = 0;
    int pend = 0;
    for (size_t i = 0; i < nc; i++) {
        if (is_py_space(c[i])) { if (o) pend = 1; }
        else {
            if (pend) { S->ob[o++] = ' '; pend = 0; }
            o += utf8_put(c[i], S->ob + o);
        }
    }
    return o;
}

/* ------------------------------------------------------------- counting */

/* Sort u128 keys ascending (Hoare quicksort, insertion sort for short runs). */
static void sort_u128(u128 *a, size_t n) {
    while (n > 16) {
        u128 piv = a[n / 2];
        ptrdiff_t i = -1, j = (ptrdiff_t)n;
        for (;;) {
            do i++; while (a[i] < piv);
            do j--; while (a[j] > piv);
            if (i >= j) break;
            u128 t = a[i]; a[i] = a[j]; a[j] = t;
        }
        size_t left = (size_t)j + 1;   /* [0, j] and [j+1, n) */
        if (left < n - left) { sort_u128(a, left); a += left; n -= left; }
        else { sort_u128(a + left, n - left); n = left; }
    }
    for (size_t i = 1; i < n; i++) {
        u128 v = a[i];
        size_t j = i;
        while (j > 0 && a[j - 1] > v) { a[j] = a[j - 1]; j--; }
        a[j] = v;
    }
}

static inline u128 pack(const u32 *p, int n, int bits) {
    u128 k = 0;
    for (int j = 0; j < n; j++) k = (k << bits) | p[j];
    return k;
}

static inline u64 mix128(u128 k) {
    u64 x = (u64)k ^ ((u64)(k >> 64) * 0x9E3779B97F4A7C15ULL);
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 29;
    return x;
}

static void ht_reset(u128 **k, u32 **v, size_t *cap, size_t *sz, size_t need) {
    size_t s = 16;
    while (s < 2 * need) s <<= 1;
    if (s > *cap) {
        free(*k); free(*v);
        *k = malloc(s * sizeof **k);
        *v = malloc(s * sizeof **v);
        *cap = s;
    }
    *sz = s;
    memset(*v, 0, s * sizeof **v);
}

static inline void ht_add(u128 *k, u32 *v, size_t sz, u128 key) {
    size_t m = sz - 1, i = mix128(key) & m;
    while (v[i] && k[i] != key) i = (i + 1) & m;
    k[i] = key;
    v[i]++;
}

static inline u32 ht_get(const u128 *k, const u32 *v, size_t sz, u128 key) {
    size_t m = sz - 1, i = mix128(key) & m;
    while (v[i]) {
        if (k[i] == key) return v[i];
        i = (i + 1) & m;
    }
    return 0;
}

/* Run-length encode sorted keys into (key, count) arrays. Returns #runs. */
static size_t rle_sorted(const u128 *k, size_t n, u128 *rk, u32 *rc) {
    size_t r = 0;
    for (size_t i = 0; i < n; i++) {
        if (r && rk[r - 1] == k[i]) rc[r - 1]++;
        else { rk[r] = k[i]; rc[r] = 1; r++; }
    }
    return r;
}

/*
 * Statistics for orders 1..maxn between hypothesis h and reference r, as
 * checksum contribution. Every (n) contributes n * (m*C1 + #hyp_ngrams*97 + #ref_ngrams),
 * m = sum over shared n-grams of min(hyp count, ref count).
 */
static u64 ngram_stats(const u32 *h, size_t hn, const u32 *r, size_t rn, int maxn, int bits,
                       int use_hash, Scratch *S) {
    u64 chk = 0;
    for (int n = 1; n <= maxn; n++) {
        size_t nh = hn >= (size_t)n ? hn - n + 1 : 0;
        size_t nr = rn >= (size_t)n ? rn - n + 1 : 0;
        u64 m = 0;
        if (use_hash) {
            ht_reset(&S->hk, &S->hv, &S->hcap, &S->hsz, nh);
            ht_reset(&S->rk, &S->rv, &S->rcap, &S->rsz, nr);
            for (size_t i = 0; i < nh; i++) ht_add(S->hk, S->hv, S->hsz, pack(h + i, n, bits));
            for (size_t i = 0; i < nr; i++) ht_add(S->rk, S->rv, S->rsz, pack(r + i, n, bits));
            for (size_t j = 0; j < S->rsz; j++) {
                if (!S->rv[j]) continue;
                u32 hv = ht_get(S->hk, S->hv, S->hsz, S->rk[j]);
                m += hv < S->rv[j] ? hv : S->rv[j];
            }
        } else {
            ENSURE(S->kh, S->khcap, nh + 1);
            ENSURE(S->kr, S->krcap, nr + 1);
            if (nh + 1 > S->rhcap) {
                free(S->rh_k); free(S->rh_c);
                S->rhcap = 2 * (nh + 1) + 16;
                S->rh_k = malloc(S->rhcap * sizeof *S->rh_k);
                S->rh_c = malloc(S->rhcap * sizeof *S->rh_c);
            }
            if (nr + 1 > S->rrcap) {
                free(S->rr_k); free(S->rr_c);
                S->rrcap = 2 * (nr + 1) + 16;
                S->rr_k = malloc(S->rrcap * sizeof *S->rr_k);
                S->rr_c = malloc(S->rrcap * sizeof *S->rr_c);
            }
            for (size_t i = 0; i < nh; i++) S->kh[i] = pack(h + i, n, bits);
            for (size_t i = 0; i < nr; i++) S->kr[i] = pack(r + i, n, bits);
            sort_u128(S->kh, nh);
            sort_u128(S->kr, nr);
            size_t a = rle_sorted(S->kh, nh, S->rh_k, S->rh_c);
            size_t b = rle_sorted(S->kr, nr, S->rr_k, S->rr_c);
            size_t i = 0, j = 0;
            while (i < a && j < b) {
                if (S->rh_k[i] < S->rr_k[j]) i++;
                else if (S->rh_k[i] > S->rr_k[j]) j++;
                else {
                    u32 x = S->rh_c[i], y = S->rr_c[j];
                    m += x < y ? x : y;
                    i++; j++;
                }
            }
        }
        chk += (u64)n * (m * 1000003ULL + (u64)nh * 97ULL + (u64)nr);
    }
    return chk;
}

/* chrF: code points with Python whitespace removed (chrF whitespace=False). */
static u64 chr_pair(const char *hs, size_t hl, const char *rs, size_t rl, int use_hash, Scratch *S) {
    ENSURE(S->cph, S->cphcap, hl + 1);
    ENSURE(S->cpr, S->cprcap, rl + 1);
    ENSURE(S->cpf, S->cpfcap, (hl > rl ? hl : rl) + 1);
    size_t nh = utf8_decode((const unsigned char *)hs, hl, S->cph), k = 0;
    for (size_t i = 0; i < nh; i++)
        if (!is_py_space(S->cph[i])) S->cpf[k++] = S->cph[i];
    /* hyp in cpf[0..k); copy into cph so cpf can hold ref */
    memcpy(S->cph, S->cpf, k * sizeof(u32));
    size_t hn = k;
    size_t nr = utf8_decode((const unsigned char *)rs, rl, S->cpr);
    k = 0;
    for (size_t i = 0; i < nr; i++)
        if (!is_py_space(S->cpr[i])) S->cpf[k++] = S->cpr[i];
    memcpy(S->cpr, S->cpf, k * sizeof(u32));
    return ngram_stats(S->cph, hn, S->cpr, k, 6, 21, use_hash, S);
}

/* Word interning across hyp and ref of one segment (open addressing, FNV-1a). */
static u32 intern_one(const char *p, u32 len, u32 *next, Scratch *S) {
    size_t m = S->itsz - 1;
    u64 h = 1469598103934665603ULL;
    for (u32 i = 0; i < len; i++) { h ^= (unsigned char)p[i]; h *= 1099511628211ULL; }
    size_t i = h & m;
    while (S->it[i].len) {
        if (S->it[i].len == len && !memcmp(S->it[i].p, p, len)) return S->it[i].id;
        i = (i + 1) & m;
    }
    S->it[i].p = p; S->it[i].len = len; S->it[i].id = (*next)++;
    return S->it[i].id;
}

/* Split tokenised text on ' ' (tokeniser output has no other whitespace) and intern. */
static size_t words_to_ids(const char *s, size_t n, u32 *ids, u32 *next, Scratch *S) {
    size_t k = 0, i = 0;
    while (i < n) {
        while (i < n && s[i] == ' ') i++;
        size_t st = i;
        while (i < n && s[i] != ' ') i++;
        if (i > st) ids[k++] = intern_one(s + st, (u32)(i - st), next, S);
    }
    return k;
}

static u64 wrd_pair(const char *hs, size_t hl, const char *rs, size_t rl, int use_hash, Scratch *S) {
    size_t cap = hl + rl + 2;
    ENSURE(S->hid, S->hidcap, cap);
    ENSURE(S->rid, S->ridcap, cap);
    S->itsz = 16;
    while (S->itsz < 2 * cap) S->itsz <<= 1;
    if (S->itcap < S->itsz) {
        free(S->it);
        S->itcap = S->itsz;
        S->it = malloc(S->itcap * sizeof *S->it);
    }
    memset(S->it, 0, S->itsz * sizeof *S->it);
    u32 next = 0;
    size_t hn = words_to_ids(hs, hl, S->hid, &next, S);
    size_t rn = words_to_ids(rs, rl, S->rid, &next, S);
    return ngram_stats(S->hid, hn, S->rid, rn, 4, 32, use_hash, S);
}

/* ------------------------------------------------------------- ASCII class scan */

/* Scalar: branch-free range tests on bytes. */
static size_t scan_scalar(const unsigned char *p, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) {
        u32 b = p[i];
        c += (b >= '{' && b <= '~') | (b >= '[' && b <= '`') | (b >= ' ' && b <= '&') |
             (b >= '(' && b <= '+') | (b >= ':' && b <= '@') | (b == '/');
    }
    return c;
}

/* Scalar with a 256-entry class table (the strong portable scalar baseline). */
static unsigned char CLS[256];
static void init_cls(void) {
    for (u32 b = 0; b < 256; b++)
        CLS[b] = (unsigned char)((b >= '{' && b <= '~') || (b >= '[' && b <= '`') || (b >= ' ' && b <= '&') ||
                                 (b >= '(' && b <= '+') || (b >= ':' && b <= '@') || b == '/');
}

static size_t scan_lut(const unsigned char *p, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) c += CLS[p[i]];
    return c;
}

/* Vector range test: signed compares, bytes >= 0x80 are negative and never match. */
static const signed char LO1[6] = {0x7A, 0x5A, 0x1F, 0x27, 0x39, 0x2E}; /* lo - 1 */
static const signed char HI1[6] = {0x7F, 0x61, 0x27, 0x2C, 0x41, 0x30}; /* hi + 1 */

static size_t scan_sse2(const unsigned char *p, size_t n) {
    __m128i lo[6], hi[6];
    for (int r = 0; r < 6; r++) {
        lo[r] = _mm_set1_epi8(LO1[r]);
        hi[r] = _mm_set1_epi8(HI1[r]);
    }
    size_t c = 0, i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i x = _mm_loadu_si128((const __m128i *)(p + i));
        __m128i m = _mm_setzero_si128();
        for (int r = 0; r < 6; r++)
            m = _mm_or_si128(m, _mm_and_si128(_mm_cmpgt_epi8(x, lo[r]), _mm_cmpgt_epi8(hi[r], x)));
        c += (size_t)__builtin_popcount((unsigned)_mm_movemask_epi8(m));
    }
    return c + scan_scalar(p + i, n - i);
}

__attribute__((target("avx2"))) static size_t scan_avx2_intr(const unsigned char *p, size_t n) {
    __m256i lo[6], hi[6];
    for (int r = 0; r < 6; r++) {
        lo[r] = _mm256_set1_epi8(LO1[r]);
        hi[r] = _mm256_set1_epi8(HI1[r]);
    }
    size_t c = 0, i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i x = _mm256_loadu_si256((const __m256i *)(p + i));
        __m256i m = _mm256_setzero_si256();
        for (int r = 0; r < 6; r++)
            m = _mm256_or_si256(m, _mm256_and_si256(_mm256_cmpgt_epi8(x, lo[r]),
                                                    _mm256_cmpgt_epi8(hi[r], x)));
        c += (size_t)__builtin_popcount((unsigned)_mm256_movemask_epi8(m));
    }
    return c + scan_scalar(p + i, n - i);
}

static size_t scan_asm(const unsigned char *p, size_t n) {
    size_t nb = n / 32;
    return (size_t)swadesh_scan_avx2(p, nb) + scan_scalar(p + nb * 32, n - nb * 32);
}

typedef size_t (*scan_fn)(const unsigned char *, size_t);

/* ------------------------------------------------------------- drivers */

static int cmp_u64(const void *a, const void *b) {
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return x < y ? -1 : x > y;
}

typedef struct {
    const char *kernel, *count, *scan;
    int threads, reps;
    const char *hyp_path, *ref_path;
} Opts;

/* Runs one kernel over all segments; returns the checksum. Timed by the caller. */
typedef struct {
    Lines hyp, ref;
    Lines tokh, tokr;  /* tokenised copies, for wrd */
    char *tokh_buf, *tokr_buf;
} Data;

static u64 run_tok(const Data *D, int T) {
    u64 chk = 0;
    long N = (long)D->hyp.n;
#pragma omp parallel num_threads(T) reduction(+ : chk)
    {
        Scratch S = {0};
#pragma omp for schedule(static)
        for (long i = 0; i < N; i++) {
            for (int side = 0; side < 2; side++) {
                const Lines *L = side ? &D->ref : &D->hyp;
                size_t n = tok13a(line_ptr(L, i), L->len[i], &S);
                u64 h = 1469598103934665603ULL;
                for (size_t j = 0; j < n; j++) { h ^= S.ob[j]; h *= 1099511628211ULL; }
                chk += h + n;
            }
        }
        scratch_free(&S);
    }
    return chk;
}

static u64 run_chr(const Data *D, int T, int use_hash) {
    u64 chk = 0;
    long N = (long)D->hyp.n;
#pragma omp parallel num_threads(T) reduction(+ : chk)
    {
        Scratch S = {0};
#pragma omp for schedule(static)
        for (long i = 0; i < N; i++)
            chk += chr_pair(line_ptr(&D->hyp, i), D->hyp.len[i], line_ptr(&D->ref, i), D->ref.len[i],
                            use_hash, &S);
        scratch_free(&S);
    }
    return chk;
}

static u64 run_wrd(const Data *D, int T, int use_hash) {
    u64 chk = 0;
    long N = (long)D->hyp.n;
#pragma omp parallel num_threads(T) reduction(+ : chk)
    {
        Scratch S = {0};
#pragma omp for schedule(static)
        for (long i = 0; i < N; i++)
            chk += wrd_pair(line_ptr(&D->tokh, i), D->tokh.len[i], line_ptr(&D->tokr, i),
                            D->tokr.len[i], use_hash, &S);
        scratch_free(&S);
    }
    return chk;
}

static u64 run_scan(const Data *D, int T, scan_fn f) {
    u64 chk = 0;
    size_t tot = D->hyp.total;
    const unsigned char *p = (const unsigned char *)D->hyp.buf;
    size_t parts = (size_t)T * 4;
#pragma omp parallel for num_threads(T) reduction(+ : chk) schedule(static)
    for (long k = 0; k < (long)parts; k++) {
        size_t s = tot * k / parts, e = tot * (k + 1) / parts;
        chk += f(p + s, e - s);
    }
    return chk;
}

static void tokenise_all(const Lines *L, Lines *out, char **buf, Scratch *S) {
    /* Tokenised text is longer than the input, so grow the buffer as needed. */
    size_t cap = 2 * L->total + 2 * L->n + 64, used = 0;
    *buf = malloc(cap);
    out->off = malloc(L->n * sizeof *out->off);
    out->len = malloc(L->n * sizeof *out->len);
    out->n = L->n;
    for (size_t i = 0; i < L->n; i++) {
        size_t n = tok13a(line_ptr(L, i), L->len[i], S);
        if (used + n + 2 > cap) {
            cap = 2 * (used + n) + 64;
            *buf = realloc(*buf, cap);
            if (!*buf) { perror("realloc"); exit(2); }
        }
        memcpy(*buf + used, S->ob, n);
        out->off[i] = used;
        out->len[i] = n;
        used += n;
        (*buf)[used] = 0;
        used += 1;
    }
    out->buf = *buf;
    out->total = used;
}

static int parse(int argc, char **argv, Opts *o) {
    o->kernel = "tok"; o->count = "sort"; o->scan = "avx2"; o->threads = 1; o->reps = 5;
    o->hyp_path = o->ref_path = NULL;
    for (int i = 1; i + 1 < argc; i += 2) {
        const char *k = argv[i], *v = argv[i + 1];
        if (!strcmp(k, "--kernel")) o->kernel = v;
        else if (!strcmp(k, "--count")) o->count = v;
        else if (!strcmp(k, "--scan")) o->scan = v;
        else if (!strcmp(k, "--threads")) o->threads = atoi(v);
        else if (!strcmp(k, "--reps")) o->reps = atoi(v);
        else if (!strcmp(k, "--hyp")) o->hyp_path = v;
        else if (!strcmp(k, "--ref")) o->ref_path = v;
        else return 0;
    }
    return o->hyp_path && o->ref_path;
}

int main(int argc, char **argv) {
    Opts o;
    if (!parse(argc, argv, &o)) {
        fprintf(stderr, "usage: %s --kernel tok|chr|wrd|scan --count sort|hash --scan scalar|sse2|avx2|asm "
                        "--threads T --reps R --hyp HYP --ref REF\n", argv[0]);
        return 1;
    }
    Data D = {0};
    D.hyp = load_lines(o.hyp_path);
    D.ref = load_lines(o.ref_path);
    if (D.hyp.n != D.ref.n) { fprintf(stderr, "line count mismatch\n"); return 2; }
    int use_hash = !strcmp(o.count, "hash");
    init_cls();
    scan_fn sf = !strcmp(o.scan, "scalar") ? scan_scalar : !strcmp(o.scan, "lut") ? scan_lut
               : !strcmp(o.scan, "sse2") ? scan_sse2 : !strcmp(o.scan, "avx2") ? scan_avx2_intr : scan_asm;

    if (!strcmp(o.kernel, "wrd")) {
        Scratch S = {0};
        tokenise_all(&D.hyp, &D.tokh, &D.tokh_buf, &S);
        tokenise_all(&D.ref, &D.tokr, &D.tokr_buf, &S);
        scratch_free(&S);
    }

    u64 chk = 0;
    double *times = malloc(o.reps * sizeof *times);
    /* one untimed warm-up, then reps timed runs */
    for (int r = -1; r < o.reps; r++) {
        double t0 = now_s();
        if (!strcmp(o.kernel, "tok")) chk = run_tok(&D, o.threads);
        else if (!strcmp(o.kernel, "chr")) chk = run_chr(&D, o.threads, use_hash);
        else if (!strcmp(o.kernel, "wrd")) chk = run_wrd(&D, o.threads, use_hash);
        else if (!strcmp(o.kernel, "scan")) chk = run_scan(&D, o.threads, sf);
        else { fprintf(stderr, "bad kernel\n"); return 1; }
        double dt = now_s() - t0;
        if (r >= 0) times[r] = dt;
    }
    u64 *ns = malloc(o.reps * sizeof *ns);
    for (int r = 0; r < o.reps; r++) ns[r] = (u64)(times[r] * 1e9 / (double)D.hyp.n);
    qsort(ns, o.reps, sizeof *ns, cmp_u64);
    printf("impl=c kernel=%s count=%s scan=%s threads=%d segments=%zu ns_per_seg=%llu checksum=%llu reps=",
           o.kernel, o.count, o.scan, o.threads, D.hyp.n, (unsigned long long)ns[o.reps / 2],
           (unsigned long long)chk);
    for (int r = 0; r < o.reps; r++) printf("%s%.0f", r ? "," : "", times[r] * 1e9 / (double)D.hyp.n);
    printf("\n");
    return 0;
}
