# Makefile for ALO Engine Build

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -ffast-math -fopenmp

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
TEST_DIR = test

# Check if MPI is available
HAS_MPI := $(shell which mpicc 2>/dev/null)
ifdef HAS_MPI
    CXXFLAGS += -DUSE_MPI
    LDFLAGS += -lmpi
endif

# OpenMP flags
OMPFLAGS = -fopenmp

# Sleef library - Updated paths to match your project structure
SLEEF_INCLUDE = -I$(CURDIR)/vec/include
SLEEF_LIB = -L$(CURDIR)/vec/lib -lsleef

# Include directories
INCLUDES = -I$(SRC_DIR) $(SLEEF_INCLUDE)

# Source files
ENGINE_SRC = $(wildcard $(SRC_DIR)/engine/alo/*.cpp) \
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

# Compile engine source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

# Build the tests
tests: $(TEST_EXEC) $(SLEEF_TEST)

$(TEST_EXEC): $(BUILD_DIR)/engine/alo/test/alo_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB)

$(SLEEF_TEST): $(BUILD_DIR)/engine/alo/test/sleef_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB)

# Run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

test_sleef: $(SLEEF_TEST)
	./$(SLEEF_TEST)

# MPI distributed version (optional)
ifdef HAS_MPI
mpi_test: $(BIN_DIR)/mpi_test
	mpirun -np 4 ./$(BIN_DIR)/mpi_test

$(BIN_DIR)/mpi_test: $(BUILD_DIR)/engine/alo/test/mpi_test.o $(ALO_LIB)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) -lmpi
endif

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	rm -rf $(LIB_DIR)

# Build everything and run tests
full: all test test_sleef

# Install the library and headers (optional)
install: all
	mkdir -p /usr/local/include/alo
	mkdir -p /usr/local/lib
	cp -r $(SRC_DIR)/engine/alo/*.h /usr/local/include/alo/
	cp -r $(SRC_DIR)/engine/alo/mod/*.h /usr/local/include/alo/mod/
	cp -r $(SRC_DIR)/engine/alo/num/*.h /usr/local/include/alo/num/
	cp -r $(SRC_DIR)/engine/alo/opt/*.h /usr/local/include/alo/opt/
	cp $(ALO_LIB) /usr/local/lib/

# Phony targets
.PHONY: all directories tests test test_sleef clean full install
ifdef HAS_MPI
.PHONY: mpi_test
endif