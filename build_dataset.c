/*
 * Jal Sakhi — dataset builder (C).
 *
 *   ./build_dataset --synthetic 3200 --out dataset.csv
 *   ./build_dataset --from-logs raw/log1.csv raw/log2.csv
 *
 * Synthetic mode produces a physically plausible stand-in with the same schema
 * as the bench rig, so the model pipeline can be developed before the logging
 * campaign is finished. Those rows are tagged source=synthetic:* and must be
 * excluded from anything reported as a bench result.
 *
 * Log mode merges SD-card logs and labels them from spike_log.csv, which
 * records the windows during which a contaminant was deliberately introduced.
 *
 * Schema: ts,tds,turbidity,ph,temp_c,level_pct,label,source
 *         label 0 = normal, 1 = contamination event
 *
 * Build: cc -O2 -o build_dataset build_dataset.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE   512
#define MAX_SPIKES 128

/* ------------------------------------------------------------------ RNG */
/* xorshift64*: reproducible across compilers and platforms, unlike rand(),
   which we need because the exported model constants must be regenerable. */
static unsigned long long rng_state = 11ULL;

static void rng_seed(unsigned long long s)
{
    rng_state = s ? s : 11ULL;
}

static double rng_uniform(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 2685821657736338717ULL) >> 11)
         / (double)(1ULL << 53);
}

static double rng_range(double lo, double hi)
{
    return lo + (hi - lo) * rng_uniform();
}

static int rng_int(int lo, int hi)   /* inclusive */
{
    return lo + (int)(rng_uniform() * (double)(hi - lo + 1));
}

/* Box-Muller, cached second deviate. */
static double rng_gauss(double mu, double sd)
{
    static int have = 0;
    static double spare;
    double u, v, s;

    if (have) { have = 0; return mu + sd * spare; }

    do {
        u = 2.0 * rng_uniform() - 1.0;
        v = 2.0 * rng_uniform() - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    have = 1;
    return mu + sd * (u * s);
}

/* ------------------------------------------------------------ baselines */
/* Measured from the hostel supply tap over nine days. Any change to these
   belongs in docs/PROBLEM_LOG.md with a reason. */
typedef struct { double mean, sd; } Stat;

static const Stat BASE_TDS  = { 212.0, 18.0 };   /* mg/L */
static const Stat BASE_TURB = {   0.62, 0.21 };  /* NTU  */
static const Stat BASE_PH   = {   7.24, 0.11 };
static const Stat BASE_TEMP = {  27.8,  2.4  };  /* C    */

/* The four contamination classes we can reproduce safely on a bench, as
   additive shifts on the baseline at full severity. */
typedef struct {
    const char *name;
    double d_tds, d_turb, d_ph;
} Spike;

static const Spike SPIKES[] = {
    { "sewage_ingress",   260.0, 11.0, -0.55 },
    { "silt_monsoon",      45.0, 18.0, -0.10 },
    { "saline_intrusion", 900.0,  0.4,  0.15 },
    { "alkaline_leach",   120.0,  1.2,  1.40 },
};
#define N_SPIKES ((int)(sizeof(SPIKES) / sizeof(SPIKES[0])))

/* ----------------------------------------------------------- timestamps */
/* Minutes since 2026-08-24 06:00, formatted without pulling in time.h
   locale behaviour — the trainer only ever compares these as strings. */
static void stamp(long minutes, char *out, size_t n)
{
    static const int mdays[] = { 31,28,31,30,31,30,31,31,30,31,30,31 };
    long total_min = minutes + 6L * 60L;
    long day       = total_min / 1440L;
    int  hh        = (int)((total_min % 1440L) / 60L);
    int  mm        = (int)(total_min % 60L);
    int  month = 8, dom = 24;

    while (day > 0) {
        int dim = mdays[month - 1];
        if (dom < dim) { dom++; }
        else           { dom = 1; month = month % 12 + 1; }
        day--;
    }

    snprintf(out, n, "2026-%02d-%02dT%02d:%02d:00", month, dom, hh, mm);
}

/* ------------------------------------------------------------ synthetic */
static void write_row(FILE *f, const char *ts, double tds, double turb,
                      double ph, double temp, double level,
                      int label, const char *source)
{
    if (turb < 0.0) turb = 0.0;
    fprintf(f, "%s,%.1f,%.2f,%.2f,%.1f,%.1f,%d,%s\n",
            ts, tds, turb, ph, temp, level, label, source);
}

static int synthesise(FILE *f, int n, unsigned long long seed)
{
    char ts[32], src[64];
    double level = 78.0;
    long minute = 0;
    int i = 0, events = 0;

    rng_seed(seed);

    while (i < n) {
        int hour;

        stamp(minute, ts, sizeof ts);
        hour = (ts[11] - '0') * 10 + (ts[12] - '0');

        /* Tank drains through the day and refills on the pump schedule. */
        level -= (hour >= 6 && hour <= 21) ? 0.9 : 0.15;
        if (level < 22.0) {
            level += 55.0;
            if (level > 95.0) level = 95.0;
        }

        if (rng_uniform() < 0.035) {
            const Spike *sp = &SPIKES[rng_int(0, N_SPIKES - 1)];
            int duration = rng_int(4, 14);
            /* Severity matters. A pipe joint weeping sewage looks almost like
               clean water for the first hour. If every simulated event were a
               full-strength spike the detector would score far better here
               than it will in a village, so a third are deliberately mild. */
            double severity = (rng_uniform() < 0.34)
                            ? rng_range(0.12, 0.45)
                            : rng_range(0.50, 1.00);
            int k;

            snprintf(src, sizeof src, "synthetic:%s", sp->name);

            for (k = 0; k < duration && i < n; k++, i++) {
                double ramp = (k + 1) / 3.0;
                if (ramp > 1.0) ramp = 1.0;
                ramp *= severity;

                stamp(minute, ts, sizeof ts);
                write_row(f, ts,
                          rng_gauss(BASE_TDS.mean,  BASE_TDS.sd)  + sp->d_tds  * ramp,
                          rng_gauss(BASE_TURB.mean, BASE_TURB.sd) + sp->d_turb * ramp,
                          rng_gauss(BASE_PH.mean,   BASE_PH.sd)   + sp->d_ph   * ramp,
                          rng_gauss(BASE_TEMP.mean, BASE_TEMP.sd),
                          level, 1, src);
                minute += 15;
                events++;
            }
            continue;
        }

        write_row(f, ts,
                  rng_gauss(BASE_TDS.mean,  BASE_TDS.sd),
                  rng_gauss(BASE_TURB.mean, BASE_TURB.sd),
                  rng_gauss(BASE_PH.mean,   BASE_PH.sd),
                  rng_gauss(BASE_TEMP.mean, BASE_TEMP.sd),
                  level, 0, "synthetic:baseline");
        minute += 15;
        i++;
    }

    return events;
}

/* ------------------------------------------------------------- log mode */
typedef struct {
    char start[32], end[32], agent[48];
} SpikeWindow;

static int load_spike_log(const char *path, SpikeWindow *out, int max)
{
    FILE *f = fopen(path, "r");
    char line[MAX_LINE];
    int n = 0;

    if (!f) {
        fprintf(stderr, "warning: %s not found — every row labelled 0\n", path);
        return 0;
    }

    if (!fgets(line, sizeof line, f)) { fclose(f); return 0; }  /* header */

    while (n < max && fgets(line, sizeof line, f)) {
        char *s = strtok(line, ",");
        char *e = strtok(NULL, ",");
        char *a = strtok(NULL, ",\r\n");
        if (!s || !e || !a) continue;
        snprintf(out[n].start, sizeof out[n].start, "%s", s);
        snprintf(out[n].end,   sizeof out[n].end,   "%s", e);
        snprintf(out[n].agent, sizeof out[n].agent, "%s", a);
        n++;
    }

    fclose(f);
    return n;
}

static int merge_logs(FILE *out, char **paths, int npaths)
{
    SpikeWindow win[MAX_SPIKES];
    int nwin = load_spike_log("spike_log.csv", win, MAX_SPIKES);
    int events = 0, p;

    for (p = 0; p < npaths; p++) {
        FILE *f = fopen(paths[p], "r");
        char line[MAX_LINE];

        if (!f) { fprintf(stderr, "skipping %s: cannot open\n", paths[p]); continue; }
        if (!fgets(line, sizeof line, f)) { fclose(f); continue; }

        while (fgets(line, sizeof line, f)) {
            char ts[32], src[64] = "bench:baseline";
            double tds, turb, ph, temp, level = 0.0;
            int label = 0, w;

            if (sscanf(line, "%31[^,],%lf,%lf,%lf,%lf,%lf",
                       ts, &tds, &turb, &ph, &temp, &level) != 6) continue;

            for (w = 0; w < nwin; w++) {
                if (strcmp(ts, win[w].start) >= 0 && strcmp(ts, win[w].end) <= 0) {
                    label = 1;
                    snprintf(src, sizeof src, "bench:%s", win[w].agent);
                    events++;
                    break;
                }
            }

            write_row(out, ts, tds, turb, ph, temp, level, label, src);
        }
        fclose(f);
    }

    return events;
}

/* ----------------------------------------------------------------- main */
int main(int argc, char **argv)
{
    const char *out_path = "dataset.csv";
    int synthetic = 0, first_log = -1, nlogs = 0, i;
    unsigned long long seed = 11ULL;
    FILE *out;
    int events, total;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--synthetic") && i + 1 < argc) {
            synthetic = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (!strcmp(argv[i], "--from-logs")) {
            first_log = i + 1;
            while (i + 1 < argc && argv[i + 1][0] != '-') { i++; nlogs++; }
        }
    }

    if (!synthetic && nlogs == 0) {
        fprintf(stderr, "usage: %s --synthetic N [--seed S] [--out PATH]\n"
                        "       %s --from-logs FILE... [--out PATH]\n",
                argv[0], argv[0]);
        return 1;
    }

    out = fopen(out_path, "w");
    if (!out) { perror(out_path); return 1; }

    fprintf(out, "ts,tds,turbidity,ph,temp_c,level_pct,label,source\n");

    if (nlogs > 0) {
        events = merge_logs(out, &argv[first_log], nlogs);
        total  = -1;
    } else {
        events = synthesise(out, synthetic, seed);
        total  = synthetic;
    }

    fclose(out);

    if (total >= 0) {
        printf("wrote %d rows to %s (%d contamination, %d normal)\n",
               total, out_path, events, total - events);
    } else {
        printf("wrote merged logs to %s (%d contamination rows labelled)\n",
               out_path, events);
    }

    return 0;
}
