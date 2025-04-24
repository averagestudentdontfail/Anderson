CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -ffast-math -g

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
TEST_DIR = test
PRO_DIR = pro
PRO_DIR = pro

# OpenMP flags
# OpenMP flags
OMPFLAGS = -fopenmp
CXXFLAGS += $(OMPFLAGS)

# MPI configuration

# MPI configuration
MPI_CXX := mpic++
MPI_COMPILE_FLAGS := $(shell $(MPI_CXX) --showme:compile 2>/dev/null)
MPI_LINK_FLAGS := $(shell $(MPI_CXX) --showme:link 2>/dev/null)
CXXFLAGS += $(MPI_COMPILE_FLAGS) -DUSE_MPI

# Tracy configuration
TRACY_ENABLE ?= 1
TRACY_INCLUDE = -I$(CURDIR)/$(PRO_DIR)/include
TRACY_LIB = -L$(CURDIR)/$(PRO_DIR)/lib -ltracy

ifeq ($(TRACY_ENABLE),1)
    CXXFLAGS += $(TRACY_INCLUDE) -DTRACY_ENABLE
    LDFLAGS += $(TRACY_LIB)
endif

# Other libraries and includes
# Tracy configuration
TRACY_ENABLE ?= 1
TRACY_INCLUDE = -I$(CURDIR)/$(PRO_DIR)/include
TRACY_LIB = -L$(CURDIR)/$(PRO_DIR)/lib -ltracy

ifeq ($(TRACY_ENABLE),1)
    CXXFLAGS += $(TRACY_INCLUDE) -DTRACY_ENABLE
    LDFLAGS += $(TRACY_LIB)
endif

# Other libraries and includes
SLEEF_INCLUDE = -I$(CURDIR)/vec/include
SLEEF_LIB = -L$(CURDIR)/vec/lib -lsleef
PROFILER_LIB = -lprofiler
INCLUDES = -I$(SRC_DIR) $(SLEEF_INCLUDE)

# Source files
# Source files
ALODISTRIBUTE_CPP = $(SRC_DIR)/engine/alo/alodistribute.cpp
ALODISTRIBUTE_H = $(SRC_DIR)/engine/alo/alodistribute.h
ALOENGINE_H = $(SRC_DIR)/engine/alo/aloengine.h
ENGINE_SRC := $(shell find $(SRC_DIR)/engine/alo -name '*.cpp')
ENGINE_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(ENGINE_SRC))
TEST_SRC = $(wildcard $(SRC_DIR)/engine/alo/test/*.cpp)
TEST_OBJ = $(patsubst $(SRC_DIR)/engine/alo/test/%.cpp,$(BUILD_DIR)/engine/alo/test/%.o,$(TEST_SRC))

# Output files
# Output files
ALO_LIB = $(LIB_DIR)/libalo.a
TEST_EXEC = $(BIN_DIR)/alo_test
SLEEF_TEST = $(BIN_DIR)/sleef_test
MPI_TEST = $(BIN_DIR)/mpi_test

all: directories $(ALO_LIB) tests

directories:
	@mkdir -p $(BUILD_DIR)/engine/alo/mod \
               $(BUILD_DIR)/engine/alo/num \
               $(BUILD_DIR)/engine/alo/opt \
               $(BUILD_DIR)/engine/alo/test \
               $(BIN_DIR) \
               $(LIB_DIR) \
               $(BUILD_DIR)/engine/alo/num \
               $(BUILD_DIR)/engine/alo/opt \
               $(BUILD_DIR)/engine/alo/test \
               $(BIN_DIR) \
               $(LIB_DIR)

# Build the ALO library
$(ALO_LIB): $(ENGINE_OBJ)
	@echo "Archiving library $@"
	ar rcs $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(wildcard $(SRC_DIR)/engine/alo/*.h) $(wildcard $(SRC_DIR)/engine/alo/mod/*.h) $(wildcard $(SRC_DIR)/engine/alo/num/*.h) $(wildcard $(SRC_DIR)/engine/alo/opt/*.h)
	@mkdir -p $(dir $@)
	@echo "Compiling $< -> $@"
	$(MPI_CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@
	@# Process dependencies slightly differently for robustness
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:.o=.d) | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(@:.o=.d.processed)
	@mv -f $(@:.o=.d.processed) $(@:.o=.P)
	@rm -f $(@:.o=.d)

# Build the tests
tests: $(TEST_EXEC) $(SLEEF_TEST) $(MPI_TEST)
tests: $(TEST_EXEC) $(SLEEF_TEST) $(MPI_TEST)

$(TEST_EXEC): $(BUILD_DIR)/engine/alo/test/alo_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)

$(SLEEF_TEST): $(BUILD_DIR)/engine/alo/test/sleef_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)

$(MPI_TEST): $(BUILD_DIR)/engine/alo/test/mpi_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)

$(MPI_TEST): $(BUILD_DIR)/engine/alo/test/mpi_test.o $(ALO_LIB)
	@echo "Linking executable $@"
	$(MPI_CXX) $(CXXFLAGS) $< -o $@ -L$(LIB_DIR) -lalo $(SLEEF_LIB) $(MPI_LINK_FLAGS) $(PROFILER_LIB) $(LDFLAGS)

mpi_test: $(MPI_TEST)
	mpirun -np 4 ./$(MPI_TEST)

# --- Run Targets with Tracy profiling ---
test: $(TEST_EXEC)
	@echo "Running alo_test with Tracy profiling..."
	@echo "Running alo_test with Tracy profiling..."
	./$(TEST_EXEC)
	@echo "Launch the Tracy profiler from $(PRO_DIR)/bin/tracy to view profiling data"
	@echo "Launch the Tracy profiler from $(PRO_DIR)/bin/tracy to view profiling data"

test_sleef: $(SLEEF_TEST)
	@echo "Running sleef_test with Tracy profiling..."
	@echo "Running sleef_test with Tracy profiling..."
	./$(SLEEF_TEST)
	@echo "Launch the Tracy profiler from $(PRO_DIR)/bin/tracy to view profiling data"
	@echo "Launch the Tracy profiler from $(PRO_DIR)/bin/tracy to view profiling data"

# --- Cleanup ---
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR)

# Tracy-specific targets
tracy_clean:
	@echo "Cleaning Tracy artifacts..."
	rm -rf $(PRO_DIR)

# --- Utility Targets ---
full: all test test_sleef

ifneq ($(MAKECMDGOALS),clean)
-include $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.P,$(ENGINE_SRC))
endif

.PHONY: all directories tests test test_sleef mpi_test clean full
.SECONDARY: $(ENGINE_OBJ) $(TEST_OBJ)