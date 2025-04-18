# Makefile for building the ALO engine and test program
# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -mavx2 -mfma -I$(CURDIR)/vec/include
LDFLAGS = -pthread -lm -L$(CURDIR)/vec/lib -lsleef -Wl,-rpath,$(CURDIR)/vec/lib

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
ENGINE_SOURCES = \
    $(SRC_DIR)/engine/alo/aloengine.cpp \
    $(SRC_DIR)/engine/alo/mod/american.cpp \
    $(SRC_DIR)/engine/alo/mod/european.cpp \
    $(SRC_DIR)/engine/alo/num/chebyshev.cpp \
    $(SRC_DIR)/engine/alo/num/integrate.cpp \
    $(SRC_DIR)/engine/alo/opt/cache.cpp \
    $(SRC_DIR)/engine/alo/opt/vector.cpp

# Test file is in engine/alo/test directory
TEST_SOURCES = \
    $(SRC_DIR)/engine/alo/test/alo_test.cpp

# Object files
ENGINE_OBJECTS = $(ENGINE_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BIN_DIR)/alo_test

# Create necessary directories including test directory
$(shell mkdir -p $(BIN_DIR) $(BUILD_DIR)/engine/alo/mod $(BUILD_DIR)/engine/alo/num $(BUILD_DIR)/engine/alo/opt $(BUILD_DIR)/engine/alo/test)

# Default target
all: $(TARGET)

# Build target executable
$(TARGET): $(ENGINE_OBJECTS) $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Build engine object files
$(BUILD_DIR)/engine/alo/%.o: $(SRC_DIR)/engine/alo/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/engine/alo/mod/%.o: $(SRC_DIR)/engine/alo/mod/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/engine/alo/num/%.o: $(SRC_DIR)/engine/alo/num/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/engine/alo/opt/%.o: $(SRC_DIR)/engine/alo/opt/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build test object files from engine/alo/test directory
$(BUILD_DIR)/engine/alo/test/%.o: $(SRC_DIR)/engine/alo/test/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)/* $(BIN_DIR)/*

# Run test program
test: $(TARGET)
	./$(TARGET)

# Alternate name for the test target
run: test

# Install
install: $(TARGET)
	mkdir -p /usr/local/bin
	cp $(TARGET) /usr/local/bin/

# Phony targets
.PHONY: all clean run test install