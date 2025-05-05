CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -mfma -ffast-math -g

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
TEST_DIR = test
PRO_DIR = pro

# SLEEF configuration
SLEEF_DIR = sleef
SLEEF_INCLUDES = -I$(SLEEF_DIR)/include
SLEEF_STATIC_LIB = $(SLEEF_DIR)/lib/libsleef.a

# Other libraries and includes
INCLUDES = -I$(SRC_DIR) $(SLEEF_INCLUDES)
LIBS = -lpthread -ldl

# Source files
ALODISTRIBUTE_CPP = $(SRC_DIR)/engine/alo/alodistribute.cpp
ALODISTRIBUTE_H = $(SRC_DIR)/engine/alo/alodistribute.h
ALOENGINE_H = $(SRC_DIR)/engine/alo/aloengine.h
ENGINE_SRC := $(shell find $(SRC_DIR)/engine/alo -name '*.cpp')
ENGINE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(ENGINE_SRC))
TEST_SRC = $(wildcard $(SRC_DIR)/engine/alo/test/*.cpp)
TEST_OBJ = $(patsubst $(SRC_DIR)/engine/alo/test/%.cpp,$(BUILD_DIR)/engine/alo/test/%.o,$(TEST_SRC))

# Output files
ALO_LIB = $(LIB_DIR)/libalo.a
TEST_EXEC = $(BIN_DIR)/alo_test
SLEEF_TEST = $(BIN_DIR)/sleef_test
SIMD_TEST = $(BIN_DIR)/simd_test

all: directories sleef $(ALO_LIB) tests

directories:
	@mkdir -p $(BUILD_DIR)/engine/alo/mod \
	          $(BUILD_DIR)/engine/alo/num \
	          $(BUILD_DIR)/engine/alo/opt \
	          $(BUILD_DIR)/engine/alo/test \
	          $(BIN_DIR) \
	          $(LIB_DIR)

# Build SLEEF
sleef: $(SLEEF_STATIC_LIB)

$(SLEEF_STATIC_LIB):
	@echo "Building SLEEF library..."
	cd $(SLEEF_DIR) && \
	mkdir -p build && \
	cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=.. \
	      -DBUILD_SHARED_LIBS=OFF \
	      -DBUILD_TESTS=OFF \
	      -DBUILD_DFT=OFF \
	      -DBUILD_GNUABI_LIBS=OFF \
	      -DBUILD_INLINE_HEADERS=ON .. && \
	$(MAKE) && \
	$(MAKE) install

# Build the ALO library
$(ALO_LIB): $(ENGINE_OBJ)
	@echo "Archiving library $@"
	ar rcs $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(wildcard $(SRC_DIR)/engine/alo/*.h) $(wildcard $(SRC_DIR)/engine/alo/mod/*.h) $(wildcard $(SRC_DIR)/engine/alo/num/*.h) $(wildcard $(SRC_DIR)/engine/alo/opt/*.h)
	@mkdir -p $(dir $@)
	@echo "Compiling $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
	@# Process dependencies slightly differently for robustness
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d) | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d.processed)
	@mv -f $(@:.o=.d.processed) $(@:.o=.P)
	@rm -f $(@:.o=.d)

# Build the tests
tests: $(TEST_EXEC) $(SLEEF_TEST) $(SIMD_TEST)

$(TEST_EXEC): $(BUILD_DIR)/engine/alo/test/alo_test.o $(ALO_LIB) $(SLEEF_STATIC_LIB)
	@echo "Linking executable $@"
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_STATIC_LIB) $(LIBS)

$(SLEEF_TEST): $(BUILD_DIR)/engine/alo/test/sleef_test.o $(ALO_LIB) $(SLEEF_STATIC_LIB)
	@echo "Linking executable $@"
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_STATIC_LIB) $(LIBS)

$(SIMD_TEST): $(BUILD_DIR)/engine/alo/test/simd_test.o $(ALO_LIB) $(SLEEF_STATIC_LIB)
	@echo "Linking executable $@"
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_STATIC_LIB) $(LIBS)

# --- Run Targets ---
test: $(TEST_EXEC)
	@echo "Running alo_test..."
	./$(TEST_EXEC)

test_sleef: $(SLEEF_TEST)
	@echo "Running sleef_test..."
	./$(SLEEF_TEST)

test_simd: $(SIMD_TEST)
	@echo "Running simd_test..."
	./$(SIMD_TEST)

# --- Cleanup ---
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)
	@echo "Cleaning SLEEF build..."
	rm -rf $(SLEEF_DIR)/build $(SLEEF_DIR)/lib $(SLEEF_DIR)/include

# --- Utility Targets ---
full: all test test_sleef test_simd

ifneq ($(MAKECMDGOALS),clean)
-include $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.P,$(ENGINE_SRC))
endif

.PHONY: all directories tests test test_sleef test_simd clean full sleef
.SECONDARY: $(ENGINE_OBJ) $(TEST_OBJ)