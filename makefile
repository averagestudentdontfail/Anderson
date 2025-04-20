# Makefile for ALO Engine Build

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -ffast-math

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
TEST_DIR = test

# OpenMP flags
OMPFLAGS = -fopenmp
CXXFLAGS += $(OMPFLAGS)

# MPI configuration - use mpicc to get the correct flags
MPI_COMPILE_FLAGS := $(shell mpic++ --showme:compile)
MPI_LINK_FLAGS := $(shell mpic++ --showme:link)
CXXFLAGS += $(MPI_COMPILE_FLAGS) -DUSE_MPI

# Sleef library - Updated paths to match your project structure
SLEEF_INCLUDE = -I$(CURDIR)/vec/include
SLEEF_LIB = -L$(CURDIR)/vec/lib -lsleef

# Include directories
INCLUDES = -I$(SRC_DIR) $(SLEEF_INCLUDE)

# Source files - List alodistribute.cpp first to ensure proper dependency resolution
ENGINE_SRC = $(SRC_DIR)/engine/alo/alodistribute.cpp \
             $(filter-out $(SRC_DIR)/engine/alo/alodistribute.cpp, $(wildcard $(SRC_DIR)/engine/alo/*.cpp)) \
             $(wildcard $(SRC_DIR)/engine/alo/mod/*.cpp) \
             $(wildcard $(SRC_DIR)/engine/alo/num/*.cpp) \
             $(wildcard $(SRC_DIR)/engine/alo/opt/*.cpp)

# Object files
ENGINE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(ENGINE_SRC))

# Test source files
TEST_SRC = $(wildcard $(SRC_DIR)/engine/alo/test/*.cpp)
TEST_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TEST_SRC))

# Library output
ALO_LIB = $(LIB_DIR)/libalo.a

# Test executables
TEST_EXEC = $(BIN_DIR)/alo_test
SLEEF_TEST = $(BIN_DIR)/sleef_test
MPI_TEST = $(BIN_DIR)/mpi_test

# Default target
all: directories $(ALO_LIB) tests

# Create directories
directories:
	@mkdir -p $(BUILD_DIR)/engine/alo/mod
	@mkdir -p $(BUILD_DIR)/engine/alo/num
	@mkdir -p $(BUILD_DIR)/engine/alo/opt
	@mkdir -p $(BUILD_DIR)/engine/alo/test
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(LIB_DIR)

# Build the ALO library
$(ALO_LIB): $(ENGINE_OBJ)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# Special rule for alodistribute.cpp to ensure it's compiled first
$(BUILD_DIR)/engine/alo/alodistribute.o: $(SRC_DIR)/engine/alo/alodistribute.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile engine source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Build the tests
tests: $(TEST_EXEC) $(SLEEF_TEST)

$(TEST_EXEC): $(BUILD_DIR)/engine/alo/test/alo_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB)

$(SLEEF_TEST): $(BUILD_DIR)/engine/alo/test/sleef_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB)

# MPI test
mpi_test: $(MPI_TEST)
	mpirun -np 4 ./$(BIN_DIR)/mpi_test

$(MPI_TEST): $(BUILD_DIR)/engine/alo/test/mpi_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS)

# Run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

test_sleef: $(SLEEF_TEST)
	./$(SLEEF_TEST)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	rm -rf $(LIB_DIR)

# Build everything and run tests
full: all test test_sleef mpi_test

# Phony targets
.PHONY: all directories tests test test_sleef mpi_test clean full