/*
 * genpeg.c — Regular PEG LDPC matrix generator (FIXED)
 *
 * Implements the Progressive Edge Growth algorithm from:
 *   Xiao-Yu Hu, E. Eleftheriou, D.-M. Arnold,
 *   "Regular and irregular progressive edge-growth Tanner graphs",
 *   IEEE Trans. Inf. Theory, 2005.
 *
 * Output: genU-compatible sparse column format
 *
 * Usage: genpeg <m> <n> <c> <seed> <output>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int type; int idx; int dist; } QItem;
typedef struct {
    QItem *buf;
    int head, tail, cap;
} Queue;

static Queue *q_new(int cap) {
    Queue *q = (Queue *)malloc(sizeof(Queue));
    q->buf = (QItem *)malloc(cap * sizeof(QItem));
    q->head = q->tail = 0;
    q->cap = cap;
    return q;
}

static void q_push(Queue *q, int type, int idx, int dist) {
    if (q->tail >= q->cap) {
        q->cap *= 2;
        q->buf = (QItem *)realloc(q->buf, q->cap * sizeof(QItem));
    }
    q->buf[q->tail].type = type;
    q->buf[q->tail].idx = idx;
    q->buf[q->tail].dist = dist;
    q->tail++;
}

static QItem q_pop(Queue *q) { return q->buf[q->head++]; }
static int q_empty(Queue *q) { return q->head >= q->tail; }
static void q_free(Queue *q) { free(q->buf); free(q); }

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <m> <n> <c> <seed> <output>\n", argv[0]);
        return 1;
    }
    int m = atoi(argv[1]), n = atoi(argv[2]), c = atoi(argv[3]);
    int seed = atoi(argv[4]);
    const char *outfile = argv[5];
    srand(seed);

    int *ck_deg = (int *)calloc(m, sizeof(int));
    int **var_edge = (int **)malloc(n * sizeof(int *));
    for (int j = 0; j < n; j++) var_edge[j] = (int *)malloc(c * sizeof(int));

    int **ck_edge = (int **)calloc(m, sizeof(int *));
    int *ck_cap = (int *)calloc(m, sizeof(int));
    for (int r = 0; r < m; r++) {
        ck_cap[r] = 8;
        ck_edge[r] = (int *)malloc(ck_cap[r] * sizeof(int));
    }

    int total_edges = 0;

    for (int j = 0; j < n; j++) {
        if ((j + 1) % 1000 == 0 || j == 0)
            fprintf(stderr, "\rVar %d/%d  edges=%d", j + 1, n, total_edges);

        for (int k = 0; k < c; k++) {
            int chosen = -1;

            if (k == 0) {
                /* First edge: min-degree check node */
                int min_deg = m + 1;
                for (int r = 0; r < m; r++)
                    if (ck_deg[r] < min_deg) min_deg = ck_deg[r];
                int n_cand = 0;
                for (int r = 0; r < m; r++)
                    if (ck_deg[r] == min_deg) n_cand++;
                int pick = rand() % n_cand;
                for (int r = 0; r < m; r++) {
                    if (ck_deg[r] == min_deg && pick-- == 0) { chosen = r; break; }
                }
            } else {
                int *c_dist = (int *)malloc(m * sizeof(int));
                int *v_visited = (int *)calloc(n, sizeof(int));
                for (int r = 0; r < m; r++) c_dist[r] = -1;

                Queue *q = q_new(32000);

                /* Seed BFS from check nodes already connected to var j */
                for (int t = 0; t < k; t++) {
                    int r = var_edge[j][t];
                    c_dist[r] = 0;
                    q_push(q, 1, r, 0);
                }

                int deepest = -1;
                int *deepest_ck = (int *)malloc(m * sizeof(int));
                int n_deep = 0;

                while (!q_empty(q)) {
                    QItem item = q_pop(q);
                    int type = item.type, idx = item.idx, dist = item.dist;

                    if (type == 1) {
                        /* Check node: expand to all connected vars */
                        for (int t = 0; t < ck_deg[idx]; t++) {
                            int v = ck_edge[idx][t];
                            if (!v_visited[v]) {
                                v_visited[v] = 1;
                                q_push(q, 0, v, dist + 1);
                            }
                        }
                    } else {
                        /* Variable node: expand to ALL its check neighbors
                           For fully-processed vars (idx < j) use all c edges.
                           For current var (idx == j) use only k edges so far. */
                        int max_t = (idx == j) ? k : c;
                        for (int t = 0; t < max_t; t++) {
                            int r2 = var_edge[idx][t];
                            if (c_dist[r2] == -1) {
                                c_dist[r2] = dist + 1;
                                q_push(q, 1, r2, dist + 1);
                                int d = dist + 1;
                                if (d > deepest) {
                                    deepest = d;
                                    n_deep = 0;
                                    deepest_ck[n_deep++] = r2;
                                } else if (d == deepest) {
                                    int dup = 0;
                                    for (int x = 0; x < n_deep; x++)
                                        if (deepest_ck[x] == r2) { dup = 1; break; }
                                    if (!dup) deepest_ck[n_deep++] = r2;
                                }
                            }
                        }
                    }
                }

                q_free(q);
                free(v_visited);
                /* Note: c_dist is still valid here; freed after unreached check */

                if (n_deep > 0) {
                    /* First check for UNREACHED check nodes (infinite distance) */
                    int n_unreached = 0;
                    for (int r = 0; r < m; r++)
                        if (c_dist[r] == -1) n_unreached++;
                    
                    if (n_unreached > 0) {
                        int min_deg = m + 1;
                        for (int r = 0; r < m; r++)
                            if (c_dist[r] == -1 && ck_deg[r] < min_deg) min_deg = ck_deg[r];
                        int n_cand = 0;
                        for (int r = 0; r < m; r++)
                            if (c_dist[r] == -1 && ck_deg[r] == min_deg) n_cand++;
                        int pick = rand() % n_cand;
                        for (int r = 0; r < m; r++) {
                            if (c_dist[r] == -1 && ck_deg[r] == min_deg && pick-- == 0)
                                { chosen = r; break; }
                        }
                    } else {
                        /* All checks reachable: among deepest, pick min-degree */
                        int min_deg = m + 1;
                        for (int i = 0; i < n_deep; i++)
                            if (ck_deg[deepest_ck[i]] < min_deg) min_deg = ck_deg[deepest_ck[i]];
                        int n_cand = 0;
                        for (int i = 0; i < n_deep; i++)
                            if (ck_deg[deepest_ck[i]] == min_deg) n_cand++;
                        int pick = rand() % n_cand;
                        for (int i = 0; i < n_deep; i++) {
                            if (ck_deg[deepest_ck[i]] == min_deg && pick-- == 0)
                                { chosen = deepest_ck[i]; break; }
                        }
                    }
                    free(c_dist);
                } else {
                    /* No new checks found: pick unconnected min-degree */
                    int *connected = (int *)calloc(m, sizeof(int));
                    for (int t = 0; t < k; t++) connected[var_edge[j][t]] = 1;
                    int n_unconn = 0;
                    for (int r = 0; r < m; r++) if (!connected[r]) n_unconn++;

                    if (n_unconn > 0) {
                        int min_deg = m + 1;
                        for (int r = 0; r < m; r++)
                            if (!connected[r] && ck_deg[r] < min_deg) min_deg = ck_deg[r];
                        int n_cand = 0;
                        for (int r = 0; r < m; r++)
                            if (!connected[r] && ck_deg[r] == min_deg) n_cand++;
                        int pick = rand() % n_cand;
                        for (int r = 0; r < m; r++) {
                            if (!connected[r] && ck_deg[r] == min_deg && pick-- == 0)
                                { chosen = r; break; }
                        }
                    } else {
                        int min_deg = m + 1;
                        for (int r = 0; r < m; r++)
                            if (ck_deg[r] < min_deg) min_deg = ck_deg[r];
                        int n_cand = 0;
                        for (int r = 0; r < m; r++)
                            if (ck_deg[r] == min_deg) n_cand++;
                        int pick = rand() % n_cand;
                        for (int r = 0; r < m; r++) {
                            if (ck_deg[r] == min_deg && pick-- == 0)
                                { chosen = r; break; }
                        }
                    }
                    free(connected);
                    free(c_dist);
                }
                free(deepest_ck);
            }

            var_edge[j][k] = chosen;
            ck_deg[chosen]++;
            if (ck_deg[chosen] > ck_cap[chosen]) {
                ck_cap[chosen] *= 2;
                ck_edge[chosen] = (int *)realloc(ck_edge[chosen],
                                                 ck_cap[chosen] * sizeof(int));
            }
            ck_edge[chosen][ck_deg[chosen] - 1] = j;
            total_edges++;
        }
    }

    fprintf(stderr, "\nDone: %d edges\n", total_edges);

    FILE *f = fopen(outfile, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", outfile); return 1; }

    fprintf(f, "%d\n%d\n%d\n0\n", m, n, c);
    for (int j = 0; j < n; j++) {
        int *rows = var_edge[j];
        /* Selection sort by row index */
        for (int a = 0; a < c - 1; a++) {
            int mi = a;
            for (int b = a + 1; b < c; b++)
                if (rows[b] < rows[mi]) mi = b;
            int t = rows[a]; rows[a] = rows[mi]; rows[mi] = t;
        }
        for (int kk = 0; kk < c; kk++) {
            if (kk > 0) fputc(' ', f);
            fprintf(f, "%d:1", rows[kk]);
        }
        fputc('\n', f);
    }
    fclose(f);
    printf("Written %s  %dx%d c=%d\n", outfile, m, n, c);

    free(ck_deg);
    for (int j = 0; j < n; j++) free(var_edge[j]);
    free(var_edge);
    for (int r = 0; r < m; r++) free(ck_edge[r]);
    free(ck_edge);
    free(ck_cap);
    return 0;
}