# Makefile for C++ NMS library
# Compiles to shared library for Python ctypes integration

CXX = g++
CXXFLAGS = -std=c++11 -O3 -fPIC -Wall -Wextra
LDFLAGS = -shared

TARGET = libnms_merger.so
SRC = nms_merger.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo "✓ Built $(TARGET)"
	@echo "  Test with: nm -D $(TARGET)"

clean:
	rm -f $(TARGET)
	@echo "✓ Cleaned"

test: $(TARGET)
	@echo "Testing library symbols..."
	nm -D $(TARGET) | grep -E "nms_merge_detections|nms_merge_tiles|nms_version"

install: $(TARGET)
	cp $(TARGET) ..
	@echo "✓ Installed to parent directory"

.PHONY: all clean test install
