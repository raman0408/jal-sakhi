/*
 * Jal Sakhi — shared scoring core.
 *
 * Header-only, pure C, depends on nothing but math.h. This same file is
 * compiled into the desktop trainer and into the ESP32 firmware, so the
 * distance the model was thresholded against and the distance the node
 * computes in the field cannot drift apart. If you change the maths here,
 * both sides change together or neither does.
 */

#ifndef JAL_SCORING_H
#define JAL_SCORING_H

#include <math.h>

#ifndef JAL_MAX_FEATURES
#define JAL_MAX_FEATURES 8
#endif

/*
 * Mahalanobis distance of x from the fitted baseline.
 *
 * linv is the inverse of the lower-triangular Cholesky factor of the
 * covariance, stored row-major and n wide. Since cov = L * L^T, the distance
 * sqrt((x-mu)^T cov^-1 (x-mu)) reduces to the plain Euclidean norm of
 * linv * (x - mu) — which is why the node only needs n*(n+1)/2 useful
 * constants and about 200 float operations, with no matrix inversion at run
 * time and no division.
 */
static inline float jal_mahalanobis(const float *x, const float *mu,
                             const float *linv, int n)
{
    float d[JAL_MAX_FEATURES];
    float acc = 0.0f;
    int i, j;

    for (i = 0; i < n; i++) {
        d[i] = x[i] - mu[i];
    }

    for (i = 0; i < n; i++) {
        float row = 0.0f;
        /* linv is lower triangular: everything past the diagonal is zero,
           so the inner loop stops at i instead of n. */
        for (j = 0; j <= i; j++) {
            row += linv[i * n + j] * d[j];
        }
        acc += row * row;
    }

    return sqrtf(acc);
}

/*
 * Linear temperature compensation of a conductivity-derived reading back to a
 * 25 C reference. Solution conductivity climbs roughly 2 % per degree, so an
 * uncompensated probe reports a warm afternoon sample as if it were saltier
 * than the same water at dawn.
 */
static inline float jal_temp_compensate(float raw, float temp_c)
{
    return raw / (1.0f + 0.02f * (temp_c - 25.0f));
}

/* ---------------------------------------------------------------- IS 10500 */
/* Bit flags. Must stay in sync with REASONS in backend/app.py. */
#define JAL_REASON_MODEL     0x01
#define JAL_REASON_PH        0x02
#define JAL_REASON_TURBIDITY 0x04
#define JAL_REASON_TDS       0x08
#define JAL_REASON_LEVEL     0x10

#define JAL_PH_MIN        6.5f
#define JAL_PH_MAX        8.5f
#define JAL_TURBIDITY_MAX 5.0f    /* NTU, permissible */
#define JAL_TDS_MAX    2000.0f    /* mg/L, permissible */
#define JAL_LEVEL_MIN    10.0f    /* percent */

/*
 * IS 10500:2012 permissible limits for drinking water where no alternate
 * source exists. The statistical model may add an alert of its own, but it can
 * never clear a sample that breaches these — a fitted detector has no business
 * voting on a legal standard.
 */
static inline unsigned char jal_hard_limits(float tds, float turbidity,
                                     float ph, float level_pct)
{
    unsigned char f = 0;

    if (ph < JAL_PH_MIN || ph > JAL_PH_MAX) f |= JAL_REASON_PH;
    if (turbidity > JAL_TURBIDITY_MAX)      f |= JAL_REASON_TURBIDITY;
    if (tds > JAL_TDS_MAX)                  f |= JAL_REASON_TDS;
    if (level_pct < JAL_LEVEL_MIN)          f |= JAL_REASON_LEVEL;

    return f;
}

#endif /* JAL_SCORING_H */
