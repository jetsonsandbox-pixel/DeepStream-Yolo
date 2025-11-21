# Makefile for CUDA Tile Extraction Library
# Compiles to shared library for Python/PyTorch integration

NVCC = nvcc
CUDA_PATH = /usr/local/cuda
NVCCFLAGS = -std=c++11 -O3 --compiler-options '-fPIC' -shared
NVCCFLAGS += -gencode arch=compute_87,code=sm_87  # Jetson Orin NX

TARGET = libcuda_tiles.so
SRC = tile_extractor.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	@echo "✓ Built $(TARGET)"
	@echo "  Test with: nm -D $(TARGET) | grep launch"

clean:
	rm -f $(TARGET)
	@echo "✓ Cleaned"

test: $(TARGET)
	@echo "Testing library symbols..."
	@nm -D $(TARGET) | grep -E "launchTileExtraction|cuda_tile_version"

install: $(TARGET)
	cp $(TARGET) ..
	@echo "✓ Installed to parent directory"

.PHONY: all clean test install
