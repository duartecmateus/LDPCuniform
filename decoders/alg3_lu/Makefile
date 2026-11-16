CC       = cc
CFLAGS   = -O3 -Wall -Wextra -std=c99
LDFLAGS  = -lm

SRCS     = lu_solve.c gf16.c gf256.c gf4.c sparse_utils.c \
           alg3prunegf16.c alg3prun256.c alg3prunegf4.c alg3prunebin.c \
           test_lu.c
OBJS     = $(SRCS:.c=.o)
TARGET   = test_lu
BENCH_SRC = bench_solver.c
BENCH_BIN = /tmp/lu_bench

.PHONY: all clean test bench list-matrices

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c lu_solve.h
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TARGET)
	@echo "=== alg3 LU: pure C decoder (all 4 fields) ==="
	@echo "Usage: ./$(TARGET) <matrix> <field> <err_pct> [threshold] [max_w] [tries]"
	@echo ""
	@echo "Examples:"
	@echo "  ./$(TARGET) H_GF16_1600x4000_c6_R060.txt 16 5 4 200 20"
	@echo "  ./$(TARGET) H_bin_1600x8000_c6_R080.txt  0 3 4 200 20"
	@echo "  ./$(TARGET) H_GF4_n1000_R0.60.txt        4 5 4 200 20"
	@echo "  ./$(TARGET) H_GF256_BER2_2000.txt        1 2 4 200 20"

bench: $(TARGET) $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $(BENCH_BIN) $(BENCH_SRC) lu_solve.o $(LDFLAGS)
	$(BENCH_BIN)

clean:
	rm -f $(OBJS) $(TARGET) $(BENCH_BIN)

list-matrices:
	@echo "Available matrix files (.txt) in parent directories:"
	@find .. -maxdepth 2 -name '*.txt' -type f 2>/dev/null | head -30
