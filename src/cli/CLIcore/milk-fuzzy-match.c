/**
 * @file milk-fuzzy-match.c
 * @brief Bigram-based fuzzy string matcher
 *
 * Reads lines from stdin or a file and scores
 * them against a search query using bigram
 * (Dice coefficient) similarity. Outputs lines
 * that exceed a configurable threshold, sorted
 * by score (descending).
 *
 * Usage:
 *   milk-fuzzy-match [-t THRESH] QUERY [FILE]
 *
 * If FILE is omitted, reads from stdin.
 * Default threshold is 0.3 (30% similarity).
 *
 * Output format (tab-separated):
 *   SCORE<TAB>LINE
 *
 * No external dependencies (no ncurses, no BLAS,
 * no FFTW). Pure POSIX C99.
 */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BIGRAMS 512
#define MAX_LINES   4096
#define MAX_LINE_LEN 2048

/**
 * struct bigram_set - set of character bigrams
 * @bigrams: array of 2-char bigrams (lowercase)
 * @count:   number of bigrams in the set
 */
struct bigram_set {
    char bigrams[MAX_BIGRAMS][2];
    int  count;
};

/**
 * struct scored_line - a line with its match score
 * @score: Dice coefficient [0.0, 1.0]
 * @line:  the original text line
 */
struct scored_line {
    double score;
    char   line[MAX_LINE_LEN];
};

/**
 * make_bigrams() - extract bigrams from a string
 * @s:   input string
 * @out: output bigram set
 *
 * Converts to lowercase and extracts all
 * consecutive 2-character pairs, skipping
 * non-alphanumeric characters.
 */
static void make_bigrams(
    const char *s,
    struct bigram_set *out)
{
    out->count = 0;
    int len = (int) strlen(s);
    char lower[MAX_LINE_LEN];

    /* lowercase copy */
    for (int i = 0; i < len && i < MAX_LINE_LEN - 1;
         i++) {
        lower[i] = (char) tolower((unsigned char) s[i]);
    }
    lower[len < MAX_LINE_LEN - 1
              ? len
              : MAX_LINE_LEN - 1] = '\0';

    for (int i = 0;
         lower[i] != '\0' && lower[i + 1] != '\0';
         i++) {
        if (!isalnum((unsigned char) lower[i]) ||
            !isalnum((unsigned char) lower[i + 1])) {
            continue;
        }
        if (out->count >= MAX_BIGRAMS) {
            break;
        }
        out->bigrams[out->count][0] = lower[i];
        out->bigrams[out->count][1] = lower[i + 1];
        out->count++;
    }
}

/**
 * dice_coefficient() - compute Dice similarity
 * @a: first bigram set
 * @b: second bigram set
 *
 * Returns the Dice coefficient: 2*|A∩B| / (|A|+|B|)
 * Range: [0.0, 1.0]. 1.0 = identical bigram sets.
 */
static double dice_coefficient(
    const struct bigram_set *a,
    const struct bigram_set *b)
{
    if (a->count == 0 || b->count == 0) {
        return 0.0;
    }

    int matches = 0;
    /* mark b bigrams as used to avoid double-count */
    int used[MAX_BIGRAMS] = {0};

    for (int i = 0; i < a->count; i++) {
        for (int j = 0; j < b->count; j++) {
            if (used[j]) {
                continue;
            }
            if (a->bigrams[i][0] == b->bigrams[j][0] &&
                a->bigrams[i][1] == b->bigrams[j][1]) {
                matches++;
                used[j] = 1;
                break;
            }
        }
    }

    return (2.0 * matches) /
           (a->count + b->count);
}

/**
 * cmp_score_desc() - compare scored lines (desc)
 */
static int cmp_score_desc(
    const void *a,
    const void *b)
{
    const struct scored_line *sa = a;
    const struct scored_line *sb = b;

    if (sb->score > sa->score) {
        return 1;
    }
    if (sb->score < sa->score) {
        return -1;
    }
    return 0;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [-t THRESHOLD] QUERY [FILE]\n"
        "\n"
        "Fuzzy-match lines against QUERY using\n"
        "bigram (Dice coefficient) similarity.\n"
        "\n"
        "Options:\n"
        "  -t THRESHOLD  Minimum score (0.0-1.0,"
        " default 0.3)\n"
        "\n"
        "Output: SCORE<TAB>LINE  (sorted by score"
        " descending)\n",
        prog);
}

int main(int argc, char **argv)
{
    double threshold = 0.3;
    const char *query = NULL;
    const char *filename = NULL;

    /* Parse args */
    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        if (strcmp(argv[argi], "-t") == 0) {
            if (argi + 1 >= argc) {
                print_usage(argv[0]);
                return 1;
            }
            threshold = atof(argv[argi + 1]);
            argi += 2;
        } else if (strcmp(argv[argi], "-h") == 0 ||
                   strcmp(argv[argi],
                          "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr,
                    "Unknown option: %s\n",
                    argv[argi]);
            return 1;
        }
    }

    if (argi >= argc) {
        print_usage(argv[0]);
        return 1;
    }
    query = argv[argi++];
    if (argi < argc) {
        filename = argv[argi];
    }

    FILE *fp = stdin;
    if (filename != NULL) {
        fp = fopen(filename, "r");
        if (fp == NULL) {
            perror(filename);
            return 1;
        }
    }

    /* Build query bigrams */
    struct bigram_set query_bg;
    make_bigrams(query, &query_bg);

    /* Read and score lines */
    static struct scored_line results[MAX_LINES];
    int nresults = 0;
    char line[MAX_LINE_LEN];

    while (fgets(line, sizeof(line), fp) != NULL) {
        /* Strip trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        if (line[0] == '\0') {
            continue;
        }

        struct bigram_set line_bg;
        make_bigrams(line, &line_bg);

        double score =
            dice_coefficient(&query_bg, &line_bg);

        if (score >= threshold &&
            nresults < MAX_LINES) {
            results[nresults].score = score;
            strncpy(results[nresults].line, line,
                    MAX_LINE_LEN - 1);
            results[nresults]
                .line[MAX_LINE_LEN - 1] = '\0';
            nresults++;
        }
    }

    if (filename != NULL) {
        fclose(fp);
    }

    /* Sort by score descending */
    qsort(results, (size_t) nresults,
          sizeof(struct scored_line),
          cmp_score_desc);

    /* Output */
    for (int i = 0; i < nresults; i++) {
        printf("%.3f\t%s\n",
               results[i].score,
               results[i].line);
    }

    return 0;
}
