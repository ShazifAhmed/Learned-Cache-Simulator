/*
 * gen_real_trace.c — capture REAL memory access traces from real workloads.
 *
 * Rather than synthesising an access distribution, this program runs an actual
 * algorithm and prints the cache-line address of every memory location it touches,
 * in the order it touches them. The resulting trace is a genuine artifact of the
 * computation — its locality emerges from the data structure, not from a chosen
 * parameter. That makes the benchmark credible: these are the access patterns real
 * code produces.
 *
 * Cache-line granularity: we print (address >> 6), i.e. 64-byte lines, which is what
 * a real cache indexes on. scripts/capture_traces.sh then remaps the raw line
 * addresses to a dense 0..K id space so the committed traces are compact and stable.
 *
 * Usage: gen_real_trace <workload> <size> <max_accesses>
 *   workloads: matmul | linkedlist | bst
 *
 * Build:  cc -O2 -o gen_real_trace tools/gen_real_trace.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static unsigned long emitted = 0;
static unsigned long max_accesses = 0;

/* Print the cache line touched at pointer p (one line per access). */
static inline void emit(const void *p) {
    if (max_accesses && emitted >= max_accesses) return;
    printf("%lu\n", (unsigned long)((uintptr_t)p >> 6));
    emitted++;
}

static int done(void) { return max_accesses && emitted >= max_accesses; }

/* Naive i-j-k matrix multiply: strong, regular reuse over A, B, C. */
static void workload_matmul(int n) {
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = calloc(n * n, sizeof(double));
    for (int i = 0; i < n * n; i++) { A[i] = i * 0.5; B[i] = i * 0.25; }

    for (int i = 0; i < n && !done(); i++)
        for (int j = 0; j < n && !done(); j++)
            for (int k = 0; k < n && !done(); k++) {
                emit(&A[i * n + k]);
                emit(&B[k * n + j]);
                emit(&C[i * n + j]);
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
    free(A); free(B); free(C);
}

/* Pointer-chasing: traverse a shuffled linked list repeatedly (cache-hostile loop). */
typedef struct Node { struct Node *next; long payload; } Node;

static void workload_linkedlist(int nodes) {
    Node *arr = malloc(sizeof(Node) * nodes);
    int *order = malloc(sizeof(int) * nodes);
    for (int i = 0; i < nodes; i++) order[i] = i;
    for (int i = nodes - 1; i > 0; i--) {           /* Fisher-Yates shuffle */
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    for (int i = 0; i < nodes; i++)                 /* link in shuffled order */
        arr[order[i]].next = &arr[order[(i + 1) % nodes]];

    Node *cur = &arr[order[0]];
    while (!done()) { emit(cur); cur = cur->next; }
    free(arr); free(order);
}

/* Binary search tree: random lookups, so nodes near the root are hot (skewed reuse). */
typedef struct TNode { struct TNode *l, *r; long key; } TNode;

static TNode *bst_insert(TNode *root, TNode *pool, int *used, long key) {
    if (!root) { TNode *n = &pool[(*used)++]; n->key = key; n->l = n->r = NULL; return n; }
    TNode *cur = root;
    for (;;) {
        if (key < cur->key) { if (cur->l) cur = cur->l; else { cur->l = &pool[(*used)++]; cur->l->key = key; cur->l->l = cur->l->r = NULL; return root; } }
        else                { if (cur->r) cur = cur->r; else { cur->r = &pool[(*used)++]; cur->r->key = key; cur->r->l = cur->r->r = NULL; return root; } }
    }
}

static void workload_bst(int keys) {
    TNode *pool = malloc(sizeof(TNode) * keys);
    int used = 0;
    TNode *root = NULL;
    for (int i = 0; i < keys; i++) root = bst_insert(root, pool, &used, rand() % (keys * 4));

    while (!done()) {                               /* random lookups */
        long target = rand() % (keys * 4);
        TNode *cur = root;
        while (cur && !done()) {
            emit(cur);
            if (target == cur->key) break;
            cur = (target < cur->key) ? cur->l : cur->r;
        }
    }
    free(pool);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <matmul|linkedlist|bst> <size> <max_accesses>\n", argv[0]);
        return 2;
    }
    const char *workload = argv[1];
    int size = atoi(argv[2]);
    max_accesses = strtoul(argv[3], NULL, 10);
    srand(12345);  /* fixed seed: structure (not absolute addresses) is reproducible */

    if (!strcmp(workload, "matmul")) workload_matmul(size);
    else if (!strcmp(workload, "linkedlist")) workload_linkedlist(size);
    else if (!strcmp(workload, "bst")) workload_bst(size);
    else { fprintf(stderr, "unknown workload: %s\n", workload); return 2; }
    return 0;
}
