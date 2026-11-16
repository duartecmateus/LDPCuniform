# Makefile for alg4LDprefilt (consistency pre-filter decoder)
# Reuses shared source files from ../alg3_lu/

CC       = cc
CFLAGS   = -O3 -Wall -Wextra -std=c99 -I../alg3_lu
LDFLAGS  = -lm

# Shared source files from alg3_LU
SHARED_DIR = ../alg3_lu
SHARED_SRC = $(SHARED_DIR)/lu_solve.c $(SHARED_DIR)/gf4.c $(SHARED_DIR)/gf16.c \
             $(SHARED_DIR)/sparse_utils.c

# New source files
NEW_SRC   = alg4prefiltgf4.c alg4prefiltgf16.c test_alg4prefilt.c

# Object files in current directory
NEW_OBJS  = $(NEW_SRC:.c=.o)
SHARED_OBJS = lu_solve.o gf4.o gf16.o sparse_utils.o
OBJS      = $(NEW_OBJS) $(SHARED_OBJS)
TARGET    = test_alg4prefilt

.PHONY: all clean test run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build shared objects from alg3_LU source
lu_solve.o: $(SHARED_DIR)/lu_solve.c $(SHARED_DIR)/lu_solve.h
	$(CC) $(CFLAGS) -c -o $@ $(SHARED_DIR)/lu_solve.c

gf4.o: $(SHARED_DIR)/gf4.c $(SHARED_DIR)/gf4.h
	$(CC) $(CFLAGS) -c -o $@ $(SHARED_DIR)/gf4.c

gf16.o: $(SHARED_DIR)/gf16.c $(SHARED_DIR)/gf16.h
	$(CC) $(CFLAGS) -c -o $@ $(SHARED_DIR)/gf16.c

sparse_utils.o: $(SHARED_DIR)/sparse_utils.c $(SHARED_DIR)/sparse_utils.h
	$(CC) $(CFLAGS) -c -o $@ $(SHARED_DIR)/sparse_utils.c

# Build new objects
%.o: %.c alg4prefiltgf4.h alg4prefiltgf16.h
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TARGET)
	@echo "=== alg4LDprefilt: consistency pre-filter decoder ==="
	@echo ""
	@echo "Examples:"
	@echo "  ./$(TARGET) ../alg3_lu/H_GF4_n1000_R0.60.txt 4 1,2,3 50"
	@echo "  ./$(TARGET) ../alg3_lu/H_GF16_1600x4000_c6_R060.txt 16 1,2,3 50"
	@echo ""

run: $(TARGET)
	@echo "=== Running GF(4) test at 1%,2%,3% BER (50 trials each) ==="
	-./$(TARGET) ../alg3_lu/H_GF4_n1000_R0.60.txt 4 1,2,3 50
	@echo ""
	@echo "=== Running GF(16) test at 1%,2%,3% BER (20 trials each) ==="
	-./$(TARGET) ../alg3_lu/H_GF16_1600x4000_c6_R060.txt 16 1,2,3 20

clean:
	rm -f $(OBJS) $(TARGET)
