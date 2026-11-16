CC      ?= clang
CFLAGS  ?= -O3 -Wall -Wextra -Wno-unused-parameter -I.
LDFLAGS ?= -lm

BUILD_DIR = build

TARGET_PRUNEG  = testprune
TARGET_PRUNEB  = testprunebin
TARGET_AUGMB   = testaugmentbin
TARGET_CMP     = compare
TARGET_PRUNEGF16 = testprunegf16
TARGET_PRUNEGF4  = testprunegf4

SOURCES_PRUNEG   = testprune.c alg2prune.c sparse_utils.c gf256.c
SOURCES_PRUNEB   = testprunebin.c alg2prunebin.c sparse_utils.c
SOURCES_AUGMB    = testaugmentbin.c alg2augmentbin.c sparse_utils.c
SOURCES_CMP      = compare.c alg2prunebin.c alg2augmentbin.c sparse_utils.c ldpcdec_ms.c
SOURCES_PRUNEGF16 = testprunegf16.c alg2prunegf16.c sparse_utils.c gf16.c
SOURCES_PRUNEGF4  = testprunegf4.c alg2prunegf4.c sparse_utils.c gf4.c

OBJECTS_PRUNEG   = $(patsubst %.c,$(BUILD_DIR)/pruneg_%.o,$(SOURCES_PRUNEG))
OBJECTS_PRUNEB   = $(patsubst %.c,$(BUILD_DIR)/pruneb_%.o,$(SOURCES_PRUNEB))
OBJECTS_AUGMB    = $(patsubst %.c,$(BUILD_DIR)/augmb_%.o,$(SOURCES_AUGMB))
OBJECTS_CMP      = $(patsubst %.c,$(BUILD_DIR)/cmp_%.o,$(SOURCES_CMP))
OBJECTS_PRUNEGF16 = $(patsubst %.c,$(BUILD_DIR)/gf16pr_%.o,$(SOURCES_PRUNEGF16))
OBJECTS_PRUNEGF4  = $(patsubst %.c,$(BUILD_DIR)/gf4pr_%.o,$(SOURCES_PRUNEGF4))

.PHONY: all run-prune-gf256 run-prune-bin run-augment-bin run-cmp run-prune-gf16 run-prune-gf4 clean info

all: $(BUILD_DIR)/$(TARGET_PRUNEG) \
     $(BUILD_DIR)/$(TARGET_PRUNEB) $(BUILD_DIR)/$(TARGET_AUGMB) \
     $(BUILD_DIR)/$(TARGET_CMP) \
     $(BUILD_DIR)/$(TARGET_PRUNEGF16) \
     $(BUILD_DIR)/$(TARGET_PRUNEGF4)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/pruneg_%.o: %.c sparse_utils.h alg2prune.h gf256.h | $(BUILD_DIR)
	@echo "[CC] Prune GF256: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/pruneb_%.o: %.c sparse_utils.h alg2prunebin.h | $(BUILD_DIR)
	@echo "[CC] Prune Binary: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/augmb_%.o: %.c sparse_utils.h alg2augmentbin.h | $(BUILD_DIR)
	@echo "[CC] Augment Binary: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/cmp_%.o: %.c sparse_utils.h alg2prunebin.h alg2augmentbin.h ldpcdec.h | $(BUILD_DIR)
	@echo "[CC] Compare: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gf16pr_%.o: %.c sparse_utils.h alg2prunegf16.h gf16.h | $(BUILD_DIR)
	@echo "[CC] Prune GF16: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/gf4pr_%.o: %.c sparse_utils.h alg2prunegf4.h gf4.h | $(BUILD_DIR)
	@echo "[CC] Prune GF4: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET_PRUNEG): $(OBJECTS_PRUNEG)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_PRUNEG) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_PRUNEB): $(OBJECTS_PRUNEB)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_PRUNEB) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_AUGMB): $(OBJECTS_AUGMB)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_AUGMB) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_CMP): $(OBJECTS_CMP)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_CMP) -o $@ $(LDFLAGS) -lm

$(BUILD_DIR)/$(TARGET_PRUNEGF16): $(OBJECTS_PRUNEGF16)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_PRUNEGF16) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_PRUNEGF4): $(OBJECTS_PRUNEGF4)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_PRUNEGF4) -o $@ $(LDFLAGS)

run-prune-gf256: $(BUILD_DIR)/$(TARGET_PRUNEG)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_PRUNEG) <matrix> <err_pct> <v> <max_w> <tries> <csv> [seed]"

run-prune-bin: $(BUILD_DIR)/$(TARGET_PRUNEB)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_PRUNEB) <matrix> <err_pct> <v> <max_w> <tries> <csv> [seed]"

run-augment-bin: $(BUILD_DIR)/$(TARGET_AUGMB)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_AUGMB) <matrix> <err_pct> <max_inter> <extra> <max_w> <tries> <csv> [seed]"

run-cmp: $(BUILD_DIR)/$(TARGET_CMP)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_CMP) <matrix> <err_pct> <prune_v> <aug_inter> <aug_extra>"

run-prune-gf16: $(BUILD_DIR)/$(TARGET_PRUNEGF16)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_PRUNEGF16) <matrix> <err_pct> <v> <max_w> <tries> <csv> [seed]"

run-prune-gf4: $(BUILD_DIR)/$(TARGET_PRUNEGF4)
	@echo "Usage: ./$(BUILD_DIR)/$(TARGET_PRUNEGF4) <matrix> <err_pct> <v> <max_w> <tries> <csv> [seed]"

info:
	@echo "prune-gf256:      $(BUILD_DIR)/$(TARGET_PRUNEG)"
	@echo "prune-bin:        $(BUILD_DIR)/$(TARGET_PRUNEB)"
	@echo "augment-bin:      $(BUILD_DIR)/$(TARGET_AUGMB)"
	@echo "compare:          $(BUILD_DIR)/$(TARGET_CMP)"
	@echo "prune-gf16:       $(BUILD_DIR)/$(TARGET_PRUNEGF16)"
	@echo "prune-gf4:        $(BUILD_DIR)/$(TARGET_PRUNEGF4)"

clean:
	rm -rf $(BUILD_DIR)
