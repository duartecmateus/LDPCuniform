CC      = clang
CFLAGS  = -O3 -Wall -Wextra -Wno-unused-parameter -I.
LDFLAGS =

BUILD_DIR = build

TARGET_GENUNIF = generateUnifMatrix
TARGET_GENU    = genU

SOURCES_GENUNIF = generateUnifMatrix.c gf256.c
SOURCES_GENU    = genU.c gf256.c

OBJECTS_GENUNIF = $(patsubst %.c,$(BUILD_DIR)/genunif_%.o,$(SOURCES_GENUNIF))
OBJECTS_GENU    = $(patsubst %.c,$(BUILD_DIR)/genu_%.o,$(SOURCES_GENU))

TARGET_PEG    = genpeg
SOURCES_PEG   = genpeg.c
OBJECTS_PEG   = $(patsubst %.c,$(BUILD_DIR)/peg_%.o,$(SOURCES_PEG))

DEFAULT_M         = 3000
DEFAULT_N         = 4000
DEFAULT_C         = 30
DEFAULT_THRESHOLD = 3
DEFAULT_OUTPUT    = H_unif_matrix.txt
DEFAULT_TRIALS    = 32

all: $(BUILD_DIR)/$(TARGET_GENUNIF) $(BUILD_DIR)/$(TARGET_GENU) $(BUILD_DIR)/$(TARGET_PEG)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/genunif_%.o: %.c gf256.h | $(BUILD_DIR)
	@echo "[CC] Compiling for generateUnifMatrix: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/genu_%.o: %.c gf256.h | $(BUILD_DIR)
	@echo "[CC] Compiling for genU: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET_GENUNIF): $(OBJECTS_GENUNIF)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_GENUNIF) -o $@ $(LDFLAGS)

$(BUILD_DIR)/$(TARGET_GENU): $(OBJECTS_GENU)
	@echo "[LD] Linking: $@"
	$(CC) $(OBJECTS_GENU) -o $@ $(LDFLAGS)

run-genunif: $(BUILD_DIR)/$(TARGET_GENUNIF)
	./$(BUILD_DIR)/$(TARGET_GENUNIF)

run-genu: $(BUILD_DIR)/$(TARGET_GENU)
	./$(BUILD_DIR)/$(TARGET_GENU)

generate-matrix: $(BUILD_DIR)/$(TARGET_GENU)
	./$(BUILD_DIR)/$(TARGET_GENU) $(DEFAULT_M) $(DEFAULT_N) $(DEFAULT_C) $(DEFAULT_THRESHOLD) $(DEFAULT_OUTPUT) $(DEFAULT_TRIALS)

clean:
	rm -rf $(BUILD_DIR)

info:
	@echo "Target generateUnifMatrix: $(BUILD_DIR)/$(TARGET_GENUNIF)"
	@echo "Target genU:               $(BUILD_DIR)/$(TARGET_GENU)"
	@echo "Build dir:                 $(BUILD_DIR)"
	@echo "Sources generateUnifMatrix: $(SOURCES_GENUNIF)"
	@echo "Sources genU:               $(SOURCES_GENU)"
	@echo "Default m:                 $(DEFAULT_M)"
	@echo "Default n:                 $(DEFAULT_N)"
	@echo "Default c:                 $(DEFAULT_C)"
	@echo "Default threshold:         $(DEFAULT_THRESHOLD)"
	@echo "Default output:            $(DEFAULT_OUTPUT)"
	@echo "Default trials:            $(DEFAULT_TRIALS)"

.PHONY: all run-genunif run-genu generate-matrix clean info

SOURCES_GENU16    = genU.c gf256.c gf16.c gf4.c
OBJECTS_GENU16    = $(patsubst %.c,$(BUILD_DIR)/genu_%.o,$(SOURCES_GENU16))

$(BUILD_DIR)/$(TARGET_GENU): $(OBJECTS_GENU16)
	@echo "[LD] Linking: $@ with GF16+GF4"
	$(CC) $(OBJECTS_GENU16) -o $@ $(LDFLAGS)

$(BUILD_DIR)/genu_gf16.o: gf16.c gf16.h | $(BUILD_DIR)
	@echo "[CC] gf16.c for genU"
	$(CC) $(CFLAGS) -c gf16.c -o $@

$(BUILD_DIR)/genu_gf4.o: gf4.c gf4.h | $(BUILD_DIR)
	@echo "[CC] gf4.c for genU"
	$(CC) $(CFLAGS) -c gf4.c -o $@

$(BUILD_DIR)/$(TARGET_PEG):
	$(CC) $(CFLAGS) -o $@ genpeg.c -lm
