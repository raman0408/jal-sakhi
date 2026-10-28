/*
 * Jal Sakhi — anomaly model trainer (C).
 *
 *   ./train_model --data dataset.csv [--target-fp 0.03] [--seed 11]
 *
 * Fits two one-class detectors on the NORMAL rows only. We never assume the
 * four contamination types we can reproduce on a bench are the four that occur
 * in a village, so neither detector is ever shown a labelled event during
 * fitting. The event rows exist purely to measure recall afterwards.
 *
 *   Mahalanobis      deployed on the ESP32. A 4x4 inverse Cholesky factor is
 *                    96 bytes of constants and about 200 float operations.
 *   Isolation Forest  runs on the village gateway as a second opinion. Handles
 *                    non-elliptical anomalies better, far too heavy for the MCU.
 *
 * Writes:
 *   model_params.h               constants the reference node compiles against
 *   metrics.json                  evaluation report
 *
 * Build: cc -O2 -o train_model train_model.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "scoring.h"

#define N_FEATURES 4
#define MAX_ROWS   200000
#define MAX_LINE   512

static const char *FEATURE_NAMES[N_FEATURES] = { "tds", "turbidity", "ph", "temp_c" };

/* Probe calibration constants, from the two-point turbidity and three-point pH
   bench calibration. Exported alongside the model so the firmware and the
   trainer can never silently disagree about what a volt means. */
#define TURB_SLOPE      -1120.4
#define TURB_INTERCEPT   3245.1
#define PH_SLOPE           -5.86
#define PH_INTERCEPT       21.34

#define NODE_ID "VLR-TK-01"

/* ------------------------------------------------------------------ RNG */
static unsigned long long rng_state = 11ULL;

static void rng_seed(unsigned long long s) { rng_state = s ? s : 11ULL; }

static double rng_uniform(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ULL) >> 11)
         / (double)(1ULL << 53);
}

static int rng_int(int lo, int hi) { return lo + (int)(rng_uniform() * (hi - lo + 1)); }

/* ------------------------------------------------------------- dataset */
typedef struct {
    double (*x)[N_FEATURES];
    int     *y;
    int      n;
} Dataset;

static int column_index(char *header, const char *name)
{
    char buf[MAX_LINE];
    char *tok;
    int idx = 0;

    snprintf(buf, sizeof buf, "%s", header);
    for (tok = strtok(buf, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), idx++) {
        if (!strcmp(tok, name)) return idx;
    }
    return -1;
}

static int load(const char *path, Dataset *ds)
{
    FILE *f = fopen(path, "r");
    char line[MAX_LINE];
    int col[N_FEATURES], col_label, i;

    if (!f) { perror(path); return 0; }
    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }

    for (i = 0; i < N_FEATURES; i++) {
        col[i] = column_index(line, FEATURE_NAMES[i]);
        if (col[i] < 0) {
            fprintf(stderr, "column '%s' missing from %s\n", FEATURE_NAMES[i], path);
            fclose(f); return 0;
        }
    }
    col_label = column_index(line, "label");
    if (col_label < 0) { fprintf(stderr, "column 'label' missing\n"); fclose(f); return 0; }

    ds->x = malloc(sizeof(*ds->x) * MAX_ROWS);
    ds->y = malloc(sizeof(*ds->y) * MAX_ROWS);
    ds->n = 0;

    while (fgets(line, sizeof line, f) && ds->n < MAX_ROWS) {
        char *tok;
        int idx = 0, got = 0;

        for (tok = strtok(line, ",\r\n"); tok; tok = strtok(NULL, ",\r\n"), idx++) {
            for (i = 0; i < N_FEATURES; i++) {
                if (idx == col[i]) { ds->x[ds->n][i] = atof(tok); got++; }
            }
            if (idx == col_label) { ds->y[ds->n] = atoi(tok); got++; }
        }
        if (got >= N_FEATURES + 1) ds->n++;
    }

    fclose(f);
    return ds->n > 0;
}

/* --------------------------------------------------------- linear algebra */
static void mean_of(double (*x)[N_FEATURES], int n, double *mu)
{
    int i, j;
    for (j = 0; j < N_FEATURES; j++) mu[j] = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < N_FEATURES; j++) mu[j] += x[i][j];
    for (j = 0; j < N_FEATURES; j++) mu[j] /= n;
}

static void covariance(double (*x)[N_FEATURES], int n, const double *mu,
                       double cov[N_FEATURES][N_FEATURES])
{
    int i, j, k;

    for (j = 0; j < N_FEATURES; j++)
        for (k = 0; k < N_FEATURES; k++) cov[j][k] = 0.0;

    for (i = 0; i < n; i++)
        for (j = 0; j < N_FEATURES; j++)
            for (k = 0; k < N_FEATURES; k++)
                cov[j][k] += (x[i][j] - mu[j]) * (x[i][k] - mu[k]);

    for (j = 0; j < N_FEATURES; j++)
        for (k = 0; k < N_FEATURES; k++) cov[j][k] /= (n - 1);   /* unbiased */

    /* Ridge term. Turbidity barely moves in clean water, so its variance on a
       short log is small enough to make the raw covariance near-singular and
       the Cholesky factorisation fail. */
    for (j = 0; j < N_FEATURES; j++) cov[j][j] += 1e-6;
}

/* cov = L * L^T, L lower triangular. Returns 0 if cov is not positive definite. */
static int cholesky(double cov[N_FEATURES][N_FEATURES], double L[N_FEATURES][N_FEATURES])
{
    int i, j, k;

    for (i = 0; i < N_FEATURES; i++)
        for (j = 0; j < N_FEATURES; j++) L[i][j] = 0.0;

    for (i = 0; i < N_FEATURES; i++) {
        for (j = 0; j <= i; j++) {
            double sum = cov[i][j];
            for (k = 0; k < j; k++) sum -= L[i][k] * L[j][k];

            if (i == j) {
                if (sum <= 0.0) {
                    fprintf(stderr, "covariance not positive definite at %d\n", i);
                    return 0;
                }
                L[i][i] = sqrt(sum);
            } else {
                L[i][j] = sum / L[j][j];
            }
        }
    }
    return 1;
}

/* Inverse of a lower-triangular matrix by forward substitution. */
static void invert_lower(double L[N_FEATURES][N_FEATURES], double Li[N_FEATURES][N_FEATURES])
{
    int i, j, k;

    for (i = 0; i < N_FEATURES; i++)
        for (j = 0; j < N_FEATURES; j++) Li[i][j] = 0.0;

    for (i = 0; i < N_FEATURES; i++) {
        Li[i][i] = 1.0 / L[i][i];
        for (j = 0; j < i; j++) {
            double sum = 0.0;
            for (k = j; k < i; k++) sum += L[i][k] * Li[k][j];
            Li[i][j] = -sum / L[i][i];
        }
    }
}

/* Calls the same routine the firmware calls, at float precision, so the
   threshold we pick here is the threshold the node will actually apply. */
static double distance(const double *xd, const double *mu,
                       double Li[N_FEATURES][N_FEATURES])
{
    float xf[N_FEATURES], muf[N_FEATURES], lif[N_FEATURES * N_FEATURES];
    int i, j;

    for (i = 0; i < N_FEATURES; i++) {
        xf[i]  = (float)xd[i];
        muf[i] = (float)mu[i];
        for (j = 0; j < N_FEATURES; j++)
            lif[i * N_FEATURES + j] = (float)Li[i][j];
    }

    return (double)jal_mahalanobis(xf, muf, lif, N_FEATURES);
}

/* ------------------------------------------------------- Isolation Forest */
#define N_TREES     120
#define SUBSAMPLE   256

typedef struct Node {
    int    feature;         /* -1 marks a leaf */
    double split;
    int    size;            /* leaf only: points that reached it */
    struct Node *left, *right;
} Node;

static double harmonic(double n)
{
    return log(n) + 0.5772156649015329;   /* Euler-Mascheroni */
}

/* Expected path length of an unsuccessful BST search over n points. Used both
   to terminate trees and to normalise scores. */
static double c_factor(double n)
{
    if (n <= 1.0) return 0.0;
    if (n == 2.0) return 1.0;
    return 2.0 * harmonic(n - 1.0) - 2.0 * (n - 1.0) / n;
}

static Node *grow(double (*x)[N_FEATURES], int *idx, int n, int depth, int max_depth)
{
    Node *node = malloc(sizeof *node);
    double lo, hi;
    int f, i, left_n = 0, *left_idx, *right_idx, right_n = 0;

    node->left = node->right = NULL;

    if (depth >= max_depth || n <= 1) {
        node->feature = -1;
        node->size = n;
        return node;
    }

    /* Random feature, random split inside its observed range. */
    f  = rng_int(0, N_FEATURES - 1);
    lo = hi = x[idx[0]][f];
    for (i = 1; i < n; i++) {
        double v = x[idx[i]][f];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }

    if (hi - lo < 1e-12) {           /* degenerate: nothing to split on */
        node->feature = -1;
        node->size = n;
        return node;
    }

    node->feature = f;
    node->split   = lo + (hi - lo) * rng_uniform();

    left_idx  = malloc(sizeof(int) * n);
    right_idx = malloc(sizeof(int) * n);

    for (i = 0; i < n; i++) {
        if (x[idx[i]][f] < node->split) left_idx[left_n++]   = idx[i];
        else                            right_idx[right_n++] = idx[i];
    }

    node->left  = grow(x, left_idx,  left_n,  depth + 1, max_depth);
    node->right = grow(x, right_idx, right_n, depth + 1, max_depth);

    free(left_idx);
    free(right_idx);
    return node;
}

static void free_tree(Node *n)
{
    if (!n) return;
    free_tree(n->left);
    free_tree(n->right);
    free(n);
}

static double path_length(const Node *n, const double *x, int depth)
{
    if (n->feature < 0) return depth + c_factor((double)n->size);
    return (x[n->feature] < n->split)
         ? path_length(n->left,  x, depth + 1)
         : path_length(n->right, x, depth + 1);
}

typedef struct { Node *trees[N_TREES]; double c_psi; } Forest;

static void forest_fit(Forest *fo, double (*x)[N_FEATURES], int n)
{
    int psi = (n < SUBSAMPLE) ? n : SUBSAMPLE;
    int max_depth = (int)ceil(log((double)psi) / log(2.0));
    int t, i;
    int *idx  = malloc(sizeof(int) * psi);
    int *pool = malloc(sizeof(int) * n);

    fo->c_psi = c_factor((double)psi);

    for (t = 0; t < N_TREES; t++) {
        /* Subsample WITHOUT replacement, per Liu et al. Drawing with
           replacement duplicates points, which shrinks the effective spread
           each tree sees and measurably flattens recall — we lost about
           fifteen points of it before fixing this. Partial Fisher-Yates
           gives psi distinct indices without shuffling all n. */
        for (i = 0; i < n; i++) pool[i] = i;
        for (i = 0; i < psi; i++) {
            int k = rng_int(i, n - 1);
            int tmp = pool[i]; pool[i] = pool[k]; pool[k] = tmp;
            idx[i] = pool[i];
        }
        fo->trees[t] = grow(x, idx, psi, 0, max_depth);
    }

    free(pool);
    free(idx);
}

/* Higher means more anomalous, matching the Mahalanobis convention so the two
   detectors can be thresholded and compared the same way. */
static double forest_score(const Forest *fo, const double *x)
{
    double sum = 0.0;
    int t;
    for (t = 0; t < N_TREES; t++) sum += path_length(fo->trees[t], x, 0);
    return pow(2.0, -(sum / N_TREES) / fo->c_psi);
}

static void forest_free(Forest *fo)
{
    int t;
    for (t = 0; t < N_TREES; t++) free_tree(fo->trees[t]);
}

/* ---------------------------------------------------------- statistics */
static int cmp_double(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return (d > 0) - (d < 0);
}

/* Linear-interpolated quantile, matching numpy's default so the C and Python
   trainers pick the same threshold from the same distances. */
static double quantile(double *v, int n, double q)
{
    double pos, frac;
    int lo;

    qsort(v, n, sizeof(double), cmp_double);
    pos  = q * (n - 1);
    lo   = (int)floor(pos);
    frac = pos - lo;

    if (lo + 1 >= n) return v[n - 1];
    return v[lo] + frac * (v[lo + 1] - v[lo]);
}

static double rate_above(const double *v, int n, double threshold)
{
    int i, hits = 0;
    for (i = 0; i < n; i++) if (v[i] > threshold) hits++;
    return n ? (double)hits / n : 0.0;
}

static void report(const char *name, double recall, double fp, double threshold)
{
    printf("  %-16s threshold %8.4f   recall %5.1f %%   false positives %4.1f %%\n",
           name, threshold, recall * 100.0, fp * 100.0);
}

/* ---------------------------------------------------------------- export */
static void export_header(const char *path, const double *mu,
                          double Li[N_FEATURES][N_FEATURES], double threshold)
{
    FILE *f = fopen(path, "w");
    int i, j;

    if (!f) { perror(path); return; }

    fprintf(f, "// GENERATED by ml/train_model.c — do not hand-edit.\n");
    fprintf(f, "// Regenerate after any recalibration or dataset extension.\n");
    fprintf(f, "#pragma once\n\n");
    fprintf(f, "#define NODE_ID \"%s\"\n", NODE_ID);
    fprintf(f, "#define N_FEATURES %d\n", N_FEATURES);
    fprintf(f, "#define SCORE_THRESHOLD %.4ff\n\n", threshold);
    fprintf(f, "#define TURB_SLOPE %.4ff\n",     (double)TURB_SLOPE);
    fprintf(f, "#define TURB_INTERCEPT %.4ff\n", (double)TURB_INTERCEPT);
    fprintf(f, "#define PH_SLOPE %.4ff\n",       (double)PH_SLOPE);
    fprintf(f, "#define PH_INTERCEPT %.4ff\n\n", (double)PH_INTERCEPT);

    fprintf(f, "// order: ");
    for (i = 0; i < N_FEATURES; i++)
        fprintf(f, "%s%s", FEATURE_NAMES[i], i < N_FEATURES - 1 ? ", " : "\n");

    fprintf(f, "static const float MU[N_FEATURES] = {");
    for (i = 0; i < N_FEATURES; i++)
        fprintf(f, "%.6ff%s", mu[i], i < N_FEATURES - 1 ? ", " : "");
    fprintf(f, "};\n\n");

    fprintf(f, "// inverse Cholesky factor of the covariance, lower triangular\n");
    fprintf(f, "static const float LINV[N_FEATURES][N_FEATURES] = {\n");
    for (i = 0; i < N_FEATURES; i++) {
        fprintf(f, "  {");
        for (j = 0; j < N_FEATURES; j++)
            fprintf(f, "%.8ff%s", Li[i][j], j < N_FEATURES - 1 ? ", " : "");
        fprintf(f, "},\n");
    }
    fprintf(f, "};\n");

    fclose(f);
    printf("wrote %s\n", path);
}

static void export_metrics(const char *path, int n, int n_norm, int n_evt,
                           double m_thr, double m_rec, double m_fp,
                           double f_thr, double f_rec, double f_fp)
{
    FILE *f;
    int i;

    f = fopen(path, "w");
    if (!f) { perror(path); return; }

    fprintf(f, "{\n");
    fprintf(f, "  \"n_samples\": %d,\n", n);
    fprintf(f, "  \"n_normal\": %d,\n", n_norm);
    fprintf(f, "  \"n_events\": %d,\n", n_evt);
    fprintf(f, "  \"features\": [");
    for (i = 0; i < N_FEATURES; i++)
        fprintf(f, "\"%s\"%s", FEATURE_NAMES[i], i < N_FEATURES - 1 ? ", " : "");
    fprintf(f, "],\n");
    fprintf(f, "  \"mahalanobis\": {\"threshold\": %.4f, \"recall\": %.4f, "
               "\"false_positive_rate\": %.4f},\n", m_thr, m_rec, m_fp);
    fprintf(f, "  \"isolation_forest\": {\"threshold\": %.4f, \"recall\": %.4f, "
               "\"false_positive_rate\": %.4f},\n", f_thr, f_rec, f_fp);
    fprintf(f, "  \"deployed_on_node\": \"mahalanobis\",\n");
    fprintf(f, "  \"note\": \"Forest runs on the gateway only; node carries Mahalanobis.\"\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("wrote %s\n", path);
}

/* ----------------------------------------------------------------- main */
int main(int argc, char **argv)
{
    const char *data_path   = "dataset.csv";
    const char *header_path = "model_params.h";
    double target_fp = 0.03;
    unsigned long long seed = 11ULL;

    Dataset ds;
    double (*norm)[N_FEATURES], (*evt)[N_FEATURES];
    int n_norm = 0, n_evt = 0, n_fit, n_hold, i, j;

    double mu[N_FEATURES];
    double cov[N_FEATURES][N_FEATURES], L[N_FEATURES][N_FEATURES], Li[N_FEATURES][N_FEATURES];
    double *d_fit, *d_hold, *d_evt;
    double *f_fit, *f_hold, *f_evt;
    double m_thr, m_rec, m_fp, f_thr, f_rec, f_fp;
    Forest forest;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc)            data_path   = argv[++i];
        else if (!strcmp(argv[i], "--header") && i + 1 < argc)     header_path = argv[++i];
        else if (!strcmp(argv[i], "--target-fp") && i + 1 < argc)  target_fp   = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)       seed = strtoull(argv[++i], NULL, 10);
    }

    rng_seed(seed);

    if (target_fp <= 0.0 || target_fp >= 1.0) {
        fprintf(stderr, "--target-fp must be between 0 and 1 (exclusive)\n");
        return 1;
    }

    if (!load(data_path, &ds)) return 1;

    norm = malloc(sizeof(*norm) * ds.n);
    evt  = malloc(sizeof(*evt)  * ds.n);

    for (i = 0; i < ds.n; i++) {
        double (*dst)[N_FEATURES] = ds.y[i] ? evt : norm;
        int k = ds.y[i] ? n_evt++ : n_norm++;
        for (j = 0; j < N_FEATURES; j++) dst[k][j] = ds.x[i][j];
    }

    if (n_norm < 3 || n_evt < 1) {
        fprintf(stderr, "dataset needs at least 3 normal rows and 1 event row\n");
        free(norm); free(evt); free(ds.x); free(ds.y);
        return 1;
    }

    printf("loaded %d samples — %d normal, %d events\n", ds.n, n_norm, n_evt);

    /* Fisher-Yates over the normal rows, then a 70/30 fit/holdout cut. The
       holdout is what the false-positive rate is measured on; measuring it on
       the fit set would flatter the threshold. */
    for (i = n_norm - 1; i > 0; i--) {
        int k = rng_int(0, i);
        for (j = 0; j < N_FEATURES; j++) {
            double t = norm[i][j]; norm[i][j] = norm[k][j]; norm[k][j] = t;
        }
    }
    n_fit  = (int)(n_norm * 0.7);
    n_hold = n_norm - n_fit;

    /* ---- Mahalanobis ------------------------------------------------ */
    mean_of(norm, n_fit, mu);
    covariance(norm, n_fit, mu, cov);
    if (!cholesky(cov, L)) return 1;
    invert_lower(L, Li);

    d_fit  = malloc(sizeof(double) * n_fit);
    d_hold = malloc(sizeof(double) * n_hold);
    d_evt  = malloc(sizeof(double) * n_evt);

    for (i = 0; i < n_fit;  i++) d_fit[i]  = distance(norm[i],         mu, Li);
    for (i = 0; i < n_hold; i++) d_hold[i] = distance(norm[n_fit + i], mu, Li);
    for (i = 0; i < n_evt;  i++) d_evt[i]  = distance(evt[i],          mu, Li);

    m_thr = quantile(d_fit, n_fit, 1.0 - target_fp);
    m_fp  = rate_above(d_hold, n_hold, m_thr);
    m_rec = rate_above(d_evt,  n_evt,  m_thr);

    /* ---- Isolation Forest ------------------------------------------- */
    forest_fit(&forest, norm, n_fit);

    f_fit  = malloc(sizeof(double) * n_fit);
    f_hold = malloc(sizeof(double) * n_hold);
    f_evt  = malloc(sizeof(double) * n_evt);

    for (i = 0; i < n_fit;  i++) f_fit[i]  = forest_score(&forest, norm[i]);
    for (i = 0; i < n_hold; i++) f_hold[i] = forest_score(&forest, norm[n_fit + i]);
    for (i = 0; i < n_evt;  i++) f_evt[i]  = forest_score(&forest, evt[i]);

    f_thr = quantile(f_fit, n_fit, 1.0 - target_fp);
    f_fp  = rate_above(f_hold, n_hold, f_thr);
    f_rec = rate_above(f_evt,  n_evt,  f_thr);

    report("Mahalanobis",     m_rec, m_fp, m_thr);
    report("IsolationForest", f_rec, f_fp, f_thr);

    export_header(header_path, mu, Li, m_thr);
    export_metrics("metrics.json", ds.n, n_norm, n_evt,
                   m_thr, m_rec, m_fp, f_thr, f_rec, f_fp);

    forest_free(&forest);
    free(d_fit); free(d_hold); free(d_evt);
    free(f_fit); free(f_hold); free(f_evt);
    free(norm); free(evt); free(ds.x); free(ds.y);
    return 0;
}
