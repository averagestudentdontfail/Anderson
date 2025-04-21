# Makefile for ALO Engine Build

# Compiler settings
CXX = g++
# Added -g for debug symbols, kept -march=native as gperftools usually handles it
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -ffast-math -g

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
TEST_DIR = test

# OpenMP flags
OMPFLAGS = -fopenmp
CXXFLAGS += $(OMPFLAGS)

# MPI configuration - use mpic++ to get the correct flags
MPI_CXX := mpic++
# Handle potential errors if mpic++ isn't found or fails
MPI_COMPILE_FLAGS := $(shell $(MPI_CXX) --showme:compile 2>/dev/null)
MPI_LINK_FLAGS := $(shell $(MPI_CXX) --showme:link 2>/dev/null)
CXXFLAGS += $(MPI_COMPILE_FLAGS) -DUSE_MPI

# Sleef library - Updated paths to match your project structure
# Using $(CURDIR) assumes Makefile is run from the project root where 'vec' resides
SLEEF_INCLUDE = -I$(CURDIR)/vec/include
SLEEF_LIB = -L$(CURDIR)/vec/lib -lsleef

# gperftools profiler library flag
PROFILER_LIB = -lprofiler

# Include directories
INCLUDES = -I$(SRC_DIR) $(SLEEF_INCLUDE)

# Dependencies - we need to track header dependencies to ensure proper ordering
# Assuming headers are within $(SRC_DIR) structure corresponding to .cpp files
ALODISTRIBUTE_CPP = $(SRC_DIR)/engine/alo/alodistribute.cpp
ALODISTRIBUTE_H = $(SRC_DIR)/engine/alo/alodistribute.h
ALOENGINE_H = $(SRC_DIR)/engine/alo/aloengine.h

# Source files - Using find for more robustness
# Make sure your source files are actually in these subdirectories relative to SRC_DIR
ENGINE_SRC := $(shell find $(SRC_DIR)/engine/alo -name '*.cpp')

# Object files
ENGINE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(ENGINE_SRC))

# Test source files
TEST_SRC = $(wildcard $(SRC_DIR)/engine/alo/test/*.cpp)
# Ensure test object paths match test source locations
TEST_OBJ = $(patsubst $(SRC_DIR)/engine/alo/test/%.cpp,$(BUILD_DIR)/engine/alo/test/%.o,$(TEST_SRC))

# Library output
ALO_LIB = $(LIB_DIR)/libalo.a

# Test executables
TEST_EXEC = $(BIN_DIR)/alo_test
SLEEF_TEST = $(BIN_DIR)/sleef_test
MPI_TEST = $(BIN_DIR)/mpi_test

# Default target
all: directories $(ALO_LIB) tests

# Create directories using a single command
directories:
	@mkdir -p $(BUILD_DIR)/engine/alo/mod \
	           $(BUILD_DIR)/engine/alo/num \
	           $(BUILD_DIR)/engine/alo/opt \
	           $(BUILD_DIR)/engine/alo/test \
	           $(BIN_DIR) \
	           $(LIB_DIR)

# Build the ALO library
$(ALO_LIB): $(ENGINE_OBJ)
	@echo "Archiving library $@"
	ar rcs $@ $^

# Compile engine source files with automatic dependency tracking
# This rule should handle all engine .cpp files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(wildcard $(SRC_DIR)/engine/alo/*.h) $(wildcard $(SRC_DIR)/engine/alo/mod/*.h) $(wildcard $(SRC_DIR)/engine/alo/num/*.h) $(wildcard $(SRC_DIR)/engine/alo/opt/*.h)
	@mkdir -p $(dir $@)
	@echo "Compiling $< -> $@"
	$(MPI_CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
	@# Process dependencies slightly differently for robustness
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d) | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d.processed)
	@mv -f $(@:.o=.d.processed) $(@:.o=.P)
	@rm -f $(@:.o=.d)


# Build the tests
tests: $(TEST_EXEC) $(SLEEF_TEST)

# Rule to link alo_test
$(TEST_EXEC): $(BUILD_DIR)/engine/alo/test/alo_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB)

# Rule to link sleef_test
$(SLEEF_TEST): $(BUILD_DIR)/engine/alo/test/sleef_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB)

# MPI test target
mpi_test: $(MPI_TEST)
	mpirun -np 4 ./$(MPI_TEST)

# Rule to link mpi_test
$(MPI_TEST): $(BUILD_DIR)/engine/alo/test/mpi_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB)

# --- Run Targets ---

# Run alo_test
test: $(TEST_EXEC)
	@echo "Running alo_test..."
	./$(TEST_EXEC)

# Run sleef_test
test_sleef: $(SLEEF_TEST)
	@echo "Running sleef_test..."
	./$(SLEEF_TEST)

# --- Cleanup ---

# Clean build files
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)

# --- Utility Targets ---

# Build everything and run tests
full: all test test_sleef

# Include automatic dependencies only if they exist
ifneq ($(MAKECMDGOALS),clean)
-include $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.P,$(ENGINE_SRC))
endif

# Phony targets
.PHONY: all directories tests test test_sleef mpi_test clean full

# Prevent deletion of intermediate object files
.SECONDARY: $(ENGINE_OBJ) $(TEST_OBJ)