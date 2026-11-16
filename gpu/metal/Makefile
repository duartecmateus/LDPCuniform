CC      ?= clang
OBJC    ?= clang
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter -fPIC -I. -DENABLE_ALG2_METAL
OBJCFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter -fPIC -fobjc-arc -I.
LDFLAGS ?= -framework Foundation -framework Metal -framework AppKit -fobjc-arc

BUILD_DIR = build
TARGET = test
METAL_SOURCE = kernel.metal
METAL_AIR = $(BUILD_DIR)/kernel.air
METAL_LIB = $(BUILD_DIR)/kernel.metallib

METAL_XCRUN := $(shell xcrun -find metal 2>/dev/null)
METALLIB_XCRUN := $(shell xcrun -find metallib 2>/dev/null)
METAL_CRYPTEX := $(firstword $(wildcard /var/run/com.apple.security.cryptexd/mnt/*/Metal.xctoolchain/usr/bin/metal))
METALLIB_CRYPTEX := $(firstword $(wildcard /var/run/com.apple.security.cryptexd/mnt/*/Metal.xctoolchain/usr/bin/metallib))
METAL ?= $(if $(METAL_XCRUN),$(METAL_XCRUN),$(if $(METAL_CRYPTEX),$(METAL_CRYPTEX),metal))
METALLIB ?= $(if $(METALLIB_XCRUN),$(METALLIB_XCRUN),$(if $(METALLIB_CRYPTEX),$(METALLIB_CRYPTEX),metallib))

C_SOURCES = test.c alg2.c gf256.c alg3prun256.c alg3prunegf16.c alg3prunegf4.c alg3prunebin.c sparse_utils.c gf16.c gf4.c
PRUNE_SOURCES = testprune_metal.c alg3prun256.c alg3prunegf16.c alg3prunegf4.c alg3prunebin.c sparse_utils.c gf256.c gf16.c gf4.c
BENCH_SOURCES = bench_bp_vs_prune.c alg3prunebin.c sparse_utils.c ldpcdec_ms.c
OBJC_SOURCES = metal_bridge.m
C_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
PRUNE_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(PRUNE_SOURCES))
BENCH_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/%.o,$(BENCH_SOURCES))
OBJC_OBJECTS = $(patsubst %.m,$(BUILD_DIR)/%.o,$(OBJC_SOURCES))
OBJECTS = $(C_OBJECTS) $(OBJC_OBJECTS)

.PHONY: all run clean info check-metal

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@echo "[CC] $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.m | $(BUILD_DIR)
	@echo "[OBJC] $<"
	$(OBJC) $(OBJCFLAGS) -c $< -o $@

$(METAL_AIR): $(METAL_SOURCE) | $(BUILD_DIR)
	@echo "[METAL] $<"
	@if [ ! -x "$(METAL)" ] && ! command -v "$(METAL)" >/dev/null 2>&1; then \
		echo "Metal compiler not found: $(METAL)"; exit 1; \
	fi
	$(METAL) -c -O2 $< -o $@

$(METAL_LIB): $(METAL_AIR)
	@echo "[METALLIB] $@"
	@if [ ! -x "$(METALLIB)" ] && ! command -v "$(METALLIB)" >/dev/null 2>&1; then \
		echo "metallib not found: $(METALLIB)"; exit 1; \
	fi
	$(METALLIB) $< -o $@

$(BUILD_DIR)/$(TARGET): $(OBJECTS) $(METAL_LIB)
	@echo "[LD] $@"
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	cp $(METAL_LIB) $(BUILD_DIR)/kernel.metallib

$(BUILD_DIR)/testprune: $(PRUNE_OBJECTS) $(OBJC_OBJECTS) $(METAL_LIB)
	@echo "[LD] $@"
	$(CC) $(PRUNE_OBJECTS) $(OBJC_OBJECTS) -o $@ $(LDFLAGS)
	@if [ "$(METAL_LIB)" != "$(BUILD_DIR)/kernel.metallib" ]; then cp $(METAL_LIB) $(BUILD_DIR)/kernel.metallib; fi

$(BUILD_DIR)/bench_bp: $(BENCH_OBJECTS) $(OBJC_OBJECTS) $(METAL_LIB)
	@echo "[LD] $@"
	$(CC) $(BENCH_OBJECTS) $(OBJC_OBJECTS) -o $@ $(LDFLAGS) -lm
	@if [ "$(METAL_LIB)" != "$(BUILD_DIR)/kernel.metallib" ]; then cp $(METAL_LIB) $(BUILD_DIR)/kernel.metallib; fi

run: $(BUILD_DIR)/$(TARGET)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET) <matrix_file> <error_rate_percent> <max_syndrome_intersections> <extra_added_rows> <max_error_weight> <max_tries> <csv_file> [seed]"

prune: $(BUILD_DIR)/testprune
	@echo "Usage: ./$(BUILD_DIR)/testprune <matrix_file> <field> <error_rate_pct> [threshold_v] [max_weight] [max_tries]"

check-metal:
	@echo "METAL=$(METAL)"
	@echo "METALLIB=$(METALLIB)"

info:
	@echo "Target: $(BUILD_DIR)/$(TARGET)"
	@echo "C sources: $(C_SOURCES)"
	@echo "ObjC sources: $(OBJC_SOURCES)"
	@echo "Metal source: $(METAL_SOURCE)"
	@echo "METAL=$(METAL)"
	@echo "METALLIB=$(METALLIB)"
	@echo "CFLAGS=$(CFLAGS)"

clean:
	rm -rf $(BUILD_DIR)
