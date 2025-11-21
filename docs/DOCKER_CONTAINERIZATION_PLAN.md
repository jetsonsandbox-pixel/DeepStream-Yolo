# Docker Containerization Plan

**Project**: Jetson YOLO Tiled Inference with ByteTrack  
**Target Platform**: Jetson Orin NX (ARM64)  
**Date**: November 21, 2025  
**Status**: Planning Phase  

---

## 🎯 Objectives

1. **Simplify Deployment**: One-command setup instead of 30+ minute manual installation
2. **Ensure Reproducibility**: Eliminate "works on my machine" issues
3. **Isolate Dependencies**: Avoid conflicts with system packages
4. **Enable Scalability**: Run multiple inference containers if needed
5. **Facilitate Updates**: Rebuild image instead of manual dependency management

---

## 📋 Requirements Analysis

### Current System Dependencies

| Component | Version | Source | Critical? |
|-----------|---------|--------|-----------|
| **CUDA** | 12.6 | NVIDIA L4T | ✅ Yes |
| **cuDNN** | 9.5.1 | NVIDIA L4T | ✅ Yes |
| **TensorRT** | 10.7.0 | NVIDIA L4T | ✅ Yes |
| **PyTorch** | 2.5.0 | NVIDIA wheel | ✅ Yes |
| **OpenCV** | 4.8.0 | System/pip | ✅ Yes |
| **NumPy** | 1.24+ | pip | ✅ Yes |
| **Ultralytics** | Latest | Local copy | ⚠️ For tracking |
| **GCC/G++** | 11+ | System | ⚠️ For C++ NMS |

### Application Files

```
DeepStream-Yolo/
├── tiled_yolo_inference.py          # Core inference pipeline
├── realtime_tiled_detection.py      # Video processing script
├── test_tiled_pipeline.py           # Testing script
├── labels.txt                       # Class labels
├── libnms_merger.so                 # C++ NMS library
├── nms_cpp/                         # C++ source
│   ├── nms_merger.cpp
│   ├── Makefile
│   └── README.md
├── model_b8_gpu0_fp32.engine        # TensorRT engine (mount as volume)
└── ultralytics/                     # ByteTrack dependency (local)
```

---

## 🏗️ Architecture Design

### Container Structure

```
┌───────────────────────────────────────────────────────────────┐
│                  Docker Container                              │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Application Layer                                    │    │
│  │  - Python scripts (inference, detection, testing)    │    │
│  │  - C++ NMS library (compiled for ARM64)              │    │
│  │  - Labels and configuration files                    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Python Dependencies                                  │    │
│  │  - PyTorch 2.5.0 (NVIDIA Jetson wheel)               │    │
│  │  - NumPy, OpenCV                                      │    │
│  │  - Ultralytics (ByteTrack)                           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  NVIDIA Stack (from base image)                       │    │
│  │  - CUDA 12.6                                          │    │
│  │  - cuDNN 9.5.1                                        │    │
│  │  - TensorRT 10.7.0 (runtime)                         │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Base OS: Ubuntu 22.04 (L4T)                          │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                                │
└───────────────────────────────────────────────────────────────┘
                              ↕
                    Volume Mounts (Host ↔ Container)
                              ↕
┌───────────────────────────────────────────────────────────────┐
│  Host System (Jetson Orin NX)                                 │
├───────────────────────────────────────────────────────────────┤
│  /models/         → TensorRT engines (.engine files)          │
│  /videos/         → Input videos (.mp4, .MOV)                 │
│  /output/         → Detection results (annotated videos)      │
│  /logs/           → Application logs                          │
└───────────────────────────────────────────────────────────────┘
```

### Volume Mount Strategy

| Host Path | Container Path | Access | Purpose |
|-----------|----------------|--------|---------|
| `./model_b8_gpu0_fp32.engine` | `/models/model.engine` | RO | TensorRT engine |
| `/home/jet-nx8/Sandbox/test-data` | `/videos` | RO | Input videos |
| `./output` | `/output` | RW | Detection results |
| `./logs` | `/logs` | RW | Debug logs |

**Rationale**:
- **Read-only for models/videos**: Prevents accidental modification
- **Read-write for output/logs**: Allows saving results
- **External mounts**: Avoids rebuilding image when data changes

---

## 📦 Implementation Plan

### Phase 1: Base Dockerfile Creation

**File**: `Dockerfile`

**Strategy**:
1. Use official NVIDIA L4T TensorRT base image
2. Install system dependencies (build tools, OpenCV deps)
3. Install Python packages (numpy, opencv-python)
4. Download and install PyTorch wheel for Jetson
5. Copy application code and Ultralytics
6. Compile C++ NMS library
7. Configure environment variables
8. Set entrypoint

**Key Decisions**:
- **Base Image**: `nvcr.io/nvidia/l4t-tensorrt:r10.7.0-runtime`
  - Includes CUDA 12.6, cuDNN 9.5.1, TensorRT 10.7.0
  - ~4GB compressed, ~8GB uncompressed
  - Runtime version (smaller than devel)
  
- **PyTorch Installation**: NVIDIA pre-built wheel
  - URL: `https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/`
  - File: `torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl`
  - Size: ~180MB

**Estimated Image Size**: 8-10 GB

---

### Phase 2: Docker Compose Configuration

**File**: `docker-compose.yml`

**Services**:
1. **yolo-inference**: Main inference service
   - GPU access enabled
   - Volume mounts for models/videos/output
   - Configurable via environment variables
   - Auto-restart on failure

2. **yolo-api** (Future - TODO #5): REST API service
   - Depends on yolo-inference
   - Exposes port 8000
   - Handles remote detection requests

**Configuration**:
```yaml
Runtime: nvidia (GPU access)
Network: host (best performance on Jetson)
Restart: unless-stopped
Logging: JSON file (10MB max, 3 files)
```

---

### Phase 3: Build & Run Scripts

#### **build.sh**
- Check if running on Jetson (verify `tegra` in `/proc/cpuinfo`)
- Build Docker image with proper tags
- Display image size
- Provide usage instructions

#### **run.sh**
- Parse command-line arguments (input, output, conf, tracking)
- Set up volume mounts dynamically
- Run container with proper GPU flags
- Stream logs to terminal
- Clean up on exit

**Features**:
- Default values for common use cases
- Argument parsing for flexibility
- Timestamp-based output filenames
- GPU validation before running

---

### Phase 4: Helper Files

#### **requirements.txt**
```
numpy>=1.24.0
opencv-python>=4.8.0
# Note: PyTorch installed separately (Jetson wheel)
# Note: Ultralytics copied into container
```

#### **.dockerignore**
```
# Exclude from build context
__pycache__/
*.pyc
output*.mp4
*.log
.git/
docs/
.vscode/
Sandbox/
*.engine  # Too large, mount as volume instead
```

#### **docker-healthcheck.sh**
```bash
#!/bin/bash
# Verify critical components are working
python3 -c "import tensorrt; import torch; import cv2" || exit 1
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader || exit 1
```

---

## 🔧 Technical Challenges & Solutions

### Challenge 1: PyTorch Wheel Compatibility

**Problem**: Standard PyPI PyTorch wheels don't work on Jetson ARM64 architecture

**Solution**:
```dockerfile
# Download NVIDIA-built PyTorch wheel for Jetson
RUN wget https://developer.download.nvidia.com/.../torch-2.5.0-...-aarch64.whl \
    && pip3 install torch-2.5.0-...-aarch64.whl \
    && rm torch-2.5.0-...-aarch64.whl
```

**Alternative**: Use NVIDIA PyTorch container as base
```dockerfile
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3
```

---

### Challenge 2: TensorRT Engine Portability

**Problem**: TensorRT engines are device-specific and cannot be built inside Docker during image creation

**Solution**:
- Build engine on host Jetson
- Mount as read-only volume
- Document engine building process

**Process**:
```bash
# On host (outside Docker)
./build_engine.sh  # Creates model_b8_gpu0_fp32.engine

# In Docker
docker run -v ./model_b8_gpu0_fp32.engine:/models/model.engine:ro ...
```

**Note**: Must rebuild engine if moving to different Jetson model (NX vs AGX)

---

### Challenge 3: Image Size Optimization

**Problem**: Base image + dependencies = 8-10 GB (large for edge device)

**Solutions**:

1. **Multi-stage build** (Recommended):
```dockerfile
# Stage 1: Build C++ library
FROM nvcr.io/nvidia/l4t-tensorrt:r10.7.0-devel AS builder
COPY nms_cpp/ /build/nms_cpp/
RUN cd /build/nms_cpp && make

# Stage 2: Runtime (smaller)
FROM nvcr.io/nvidia/l4t-tensorrt:r10.7.0-runtime
COPY --from=builder /build/nms_cpp/libnms_merger.so /app/
# Only runtime dependencies, no build tools
```

2. **Aggressive cleanup**:
```dockerfile
RUN apt-get update && apt-get install -y ... \
    && rm -rf /var/lib/apt/lists/*  # Remove package cache
    && pip3 install ... --no-cache-dir  # Don't cache pip downloads
```

3. **Use `.dockerignore`**:
- Exclude docs, tests, output files from build context
- Reduces upload time to Docker daemon

**Expected Reduction**: 8-10 GB → 6-7 GB

---

### Challenge 4: GPU Access in Container

**Problem**: Docker needs special configuration to access NVIDIA GPU

**Solutions**:

1. **Docker runtime**:
```yaml
services:
  yolo-inference:
    runtime: nvidia  # Enable NVIDIA Container Runtime
```

2. **Environment variables**:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
```

3. **Device access** (if runtime not available):
```yaml
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
```

**Validation**:
```bash
docker run --rm --runtime nvidia jetson-yolo nvidia-smi
```

---

### Challenge 5: Shared Memory for Video Processing

**Problem**: OpenCV video decoding may need more shared memory than Docker default (64MB)

**Solution**:
```yaml
services:
  yolo-inference:
    shm_size: '2gb'  # Increase shared memory
```

Or mount host's `/dev/shm`:
```yaml
volumes:
  - /dev/shm:/dev/shm
```

---

## 📊 Performance Considerations

### Expected Overhead

| Metric | Native | Docker | Overhead |
|--------|--------|--------|----------|
| **Inference FPS** | 10.0 | 9.8-10.0 | ~0-2% |
| **Memory Usage** | 2.5 GB | 2.7 GB | ~200 MB |
| **Startup Time** | 5s | 8s | +3s |
| **Disk Space** | 500 MB | 8 GB | +7.5 GB |

**Overhead Sources**:
- Container filesystem layer
- Docker daemon overhead
- Virtualized networking (if not using `--network host`)

**Mitigation**:
- Use `--network host` on Jetson (no network virtualization)
- Mount `/dev/shm` for shared memory
- Use GPU directly (no virtualization needed)

### Benchmarking Plan

**Before Deployment**:
```bash
# Native
python3 realtime_tiled_detection.py --input test.mp4 --max-frames 100

# Docker
docker run ... python3 realtime_tiled_detection.py --input test.mp4 --max-frames 100
```

**Metrics to Compare**:
- FPS (should be within 5%)
- Memory usage (acceptable < 500MB increase)
- GPU utilization (should match native)
- Startup time (acceptable < 10s)

---

## 🚀 Deployment Workflow

### Step 1: Prepare Host System

```bash
# Install Docker on Jetson (if not already installed)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-runtime

# Configure Docker to use nvidia runtime
sudo systemctl restart docker
```

### Step 2: Build Image

```bash
cd /home/jet-nx8/DeepStream-Yolo

# Create Dockerfile and supporting files
./build.sh

# Expected output:
# ✅ Image built: jetson-yolo-tiled:latest (8.2 GB)
```

### Step 3: Test Container

```bash
# Quick test (help command)
docker run --rm jetson-yolo-tiled:latest python3 realtime_tiled_detection.py --help

# GPU test
docker run --rm --runtime nvidia jetson-yolo-tiled:latest nvidia-smi

# Full inference test
./run.sh --input /videos/test.mp4 --output /output/result.mp4
```

### Step 4: Production Deployment

```bash
# Using docker-compose (recommended)
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📝 File Creation Checklist

### Required Files (Priority 1)

- [ ] **Dockerfile** - Main container definition
- [ ] **docker-compose.yml** - Service orchestration
- [ ] **requirements.txt** - Python dependencies
- [ ] **.dockerignore** - Build context exclusions
- [ ] **build.sh** - Image build script
- [ ] **run.sh** - Container run script

### Supporting Files (Priority 2)

- [ ] **docker-healthcheck.sh** - Health verification
- [ ] **docs/DOCKER_USAGE.md** - User guide
- [ ] **docs/DOCKER_TROUBLESHOOTING.md** - Common issues

### Optional Files (Priority 3)

- [ ] **Dockerfile.api** - REST API service (TODO #5)
- [ ] **Dockerfile.multi** - Multi-stage optimized build
- [ ] **.github/workflows/docker-build.yml** - CI/CD automation

---

## 🧪 Testing Strategy

### Unit Tests (Container Components)

1. **Base Image Test**:
```bash
docker run --rm nvcr.io/nvidia/l4t-tensorrt:r10.7.0-runtime \
  python3 -c "import tensorrt; print(tensorrt.__version__)"
# Expected: 10.7.0
```

2. **Build Test**:
```bash
docker build -t jetson-yolo-test .
# Should complete without errors
```

3. **GPU Access Test**:
```bash
docker run --rm --runtime nvidia jetson-yolo-test nvidia-smi
# Should show GPU info
```

### Integration Tests (Full Pipeline)

1. **Simple Inference Test**:
```bash
./run.sh --input /videos/short_clip.mp4 --output /output/test1.mp4
# Should complete and produce output video
```

2. **Tracking Test**:
```bash
./run.sh --input /videos/test.mp4 --output /output/test2.mp4 --enable-tracking
# Should show track IDs in output
```

3. **Performance Test**:
```bash
./run.sh --input /videos/long_clip.mp4 --max-frames 1000
# Should maintain ~10 FPS
```

### Stress Tests

1. **Long-Running Test**:
```bash
# Process entire video (5000+ frames)
./run.sh --input /videos/full_video.mp4 --output /output/full.mp4
# Should complete without memory leaks or crashes
```

2. **Rapid Restart Test**:
```bash
for i in {1..10}; do
  ./run.sh --input /videos/short.mp4 --output /output/test_$i.mp4
done
# Should handle rapid container creation/destruction
```

3. **Concurrent Container Test** (if supported):
```bash
./run.sh --input /videos/video1.mp4 --output /output/out1.mp4 &
./run.sh --input /videos/video2.mp4 --output /output/out2.mp4 &
wait
# Should handle multiple containers (if GPU supports it)
```

---

## 📈 Success Metrics

### Quantitative Goals

| Metric | Target | Method |
|--------|--------|--------|
| **Build Time** | < 15 min | Time `docker build` command |
| **Image Size** | < 10 GB | `docker images` |
| **FPS Performance** | ≥ 9.5 FPS | Compare with native (10 FPS) |
| **Memory Overhead** | < 500 MB | `docker stats` |
| **Startup Time** | < 10 sec | Time to first frame processed |
| **Deployment Time** | < 2 min | From image to running container |

### Qualitative Goals

- ✅ **Ease of Use**: Single command deployment (`docker-compose up`)
- ✅ **Reproducibility**: Same results across different Jetson devices
- ✅ **Maintainability**: Easy to update dependencies (rebuild image)
- ✅ **Documentation**: Clear instructions for users
- ✅ **Debugging**: Easy log access and error diagnosis

---

## 🔮 Future Enhancements

### Phase 2 Features (After Initial Deployment)

1. **Docker Registry Integration**:
```bash
# Push to private registry
docker tag jetson-yolo:latest registry.example.com/jetson-yolo:latest
docker push registry.example.com/jetson-yolo:latest

# Pull on other Jetson devices
docker pull registry.example.com/jetson-yolo:latest
```

2. **Environment Configuration**:
```yaml
# .env file support
CONFIDENCE_THRESHOLD=0.25
ENABLE_TRACKING=true
INPUT_VIDEO=/videos/default.mp4
```

3. **Health Monitoring**:
```yaml
# Enhanced healthcheck
healthcheck:
  test: ["CMD", "/app/docker-healthcheck.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

4. **Prometheus Metrics** (for monitoring):
```python
# Expose metrics endpoint
from prometheus_client import start_http_server, Counter, Gauge

frames_processed = Counter('frames_processed_total', 'Total frames processed')
fps_gauge = Gauge('inference_fps', 'Current inference FPS')
```

### Phase 3 Features (Advanced)

1. **Kubernetes Deployment** (Multi-Jetson cluster):
```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: yolo-inference
spec:
  selector:
    matchLabels:
      app: yolo
  template:
    spec:
      runtimeClassName: nvidia
      containers:
      - name: yolo
        image: jetson-yolo:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

2. **Auto-scaling** (based on queue depth):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: yolo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: yolo-inference
  minReplicas: 1
  maxReplicas: 4
```

3. **CI/CD Pipeline** (GitHub Actions):
```yaml
name: Build and Push Docker Image
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: [self-hosted, jetson]
    steps:
      - uses: actions/checkout@v2
      - name: Build image
        run: docker build -t jetson-yolo:${{ github.sha }} .
      - name: Push to registry
        run: docker push jetson-yolo:${{ github.sha }}
```

---

## ⚠️ Risks & Mitigation

### Risk 1: Docker Performance Degradation

**Risk**: Container overhead reduces FPS below acceptable threshold (< 9 FPS)

**Likelihood**: Low  
**Impact**: High  

**Mitigation**:
- Use `--network host` to eliminate network overhead
- Mount `/dev/shm` for shared memory
- Benchmark before production deployment
- Profile with `nsys` to identify bottlenecks

**Rollback**: Continue using native deployment if overhead > 10%

---

### Risk 2: Image Size Too Large

**Risk**: 10+ GB image doesn't fit on Jetson storage or takes too long to transfer

**Likelihood**: Medium  
**Impact**: Medium  

**Mitigation**:
- Implement multi-stage build to reduce size
- Use `.dockerignore` aggressively
- Consider separate API container (smaller)
- Document storage requirements clearly

**Alternative**: Create "slim" image without tracking or C++ NMS (6-7 GB)

---

### Risk 3: PyTorch Wheel Compatibility Issues

**Risk**: NVIDIA wheel doesn't work with specific Jetson software version

**Likelihood**: Low  
**Impact**: High  

**Mitigation**:
- Test on multiple JetPack versions
- Document exact compatible versions
- Provide alternative wheels (different PyTorch versions)
- Fall back to building from source if needed

**Detection**: Unit test PyTorch import during build

---

### Risk 4: TensorRT Engine Incompatibility

**Risk**: Engine built on host doesn't work in container due to library version mismatch

**Likelihood**: Very Low  
**Impact**: High  

**Mitigation**:
- Use same TensorRT version in container as host
- Document engine building requirements
- Provide engine rebuild script inside container
- Test engine loading in container before deployment

**Validation**: Add healthcheck to load engine on startup

---

## 💰 Resource Estimates

### Development Time

| Phase | Tasks | Estimated Hours |
|-------|-------|-----------------|
| **Phase 1** | Dockerfile creation, testing | 4-6 hours |
| **Phase 2** | docker-compose.yml, volume setup | 2-3 hours |
| **Phase 3** | Build/run scripts, automation | 2-3 hours |
| **Phase 4** | Documentation, testing | 3-4 hours |
| **Phase 5** | Debugging, optimization | 2-4 hours |
| **Total** | | **13-20 hours** |

### Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| **Base Image** | 4 GB | L4T TensorRT runtime |
| **PyTorch Wheel** | 180 MB | Jetson-specific build |
| **Application Code** | 10 MB | Python + C++ |
| **Python Packages** | 200 MB | NumPy, OpenCV, etc. |
| **Build Layers** | 1-2 GB | Intermediate Docker layers |
| **Final Image** | 6-8 GB | Compressed/optimized |
| **Working Space** | 2 GB | Container runtime |
| **Total** | **10-12 GB** | On Jetson storage |

### Network Requirements

| Operation | Data Transfer | Notes |
|-----------|---------------|-------|
| **Pull Base Image** | 4 GB | One-time (cached) |
| **Build Context Upload** | 50 MB | Each build |
| **PyTorch Download** | 180 MB | During build |
| **Push to Registry** | 6-8 GB | If using registry |
| **Pull from Registry** | 6-8 GB | On other Jetsons |

---

## ✅ Acceptance Criteria

### Must Have (P0)

- [ ] Dockerfile builds successfully on Jetson Orin NX
- [ ] Container can access GPU (verified with `nvidia-smi`)
- [ ] TensorRT engine loads without errors
- [ ] Inference runs at ≥ 9.5 FPS (within 5% of native)
- [ ] Output video is saved to host filesystem
- [ ] Container logs are accessible via `docker logs`
- [ ] `docker-compose up` starts container successfully
- [ ] Documentation includes basic usage instructions

### Should Have (P1)

- [ ] Build script validates environment (Jetson check)
- [ ] Run script accepts command-line arguments
- [ ] Health check monitors critical components
- [ ] Image size < 10 GB
- [ ] Startup time < 10 seconds
- [ ] Tracking works with `--enable-tracking`
- [ ] Container restarts automatically on failure

### Nice to Have (P2)

- [ ] Multi-stage build for size optimization
- [ ] CI/CD pipeline for automated builds
- [ ] Prometheus metrics exposed
- [ ] REST API container (TODO #5)
- [ ] Kubernetes deployment manifests
- [ ] Docker registry integration

---

## 📚 Documentation Deliverables

### User Documentation

1. **DOCKER_USAGE.md**:
   - Quick start guide
   - Command examples
   - Common workflows
   - Volume mount explanations

2. **DOCKER_TROUBLESHOOTING.md**:
   - Common errors and solutions
   - Performance tuning tips
   - GPU access issues
   - Log analysis guide

3. **README.md Updates**:
   - Add Docker deployment section
   - Link to Docker documentation
   - Prerequisites for Docker usage

### Developer Documentation

1. **DOCKER_ARCHITECTURE.md** (This document):
   - System design
   - Technical decisions
   - Performance analysis

2. **DOCKER_BUILD_PROCESS.md**:
   - Build step-by-step explanation
   - Customization guide
   - Optimization techniques

3. **Inline Comments**:
   - Dockerfile comments explaining each step
   - docker-compose.yml annotations
   - Script comments for maintainability

---

## 🎯 Implementation Timeline

### Week 1: Foundation

**Days 1-2**: Setup & Base Image
- [ ] Create Dockerfile (base image, dependencies)
- [ ] Test base image on Jetson
- [ ] Document PyTorch wheel installation

**Days 3-4**: Application Integration
- [ ] Copy application code into container
- [ ] Compile C++ NMS library
- [ ] Test inference pipeline in container

**Days 5**: Volume Mounts & Testing
- [ ] Configure volume mounts
- [ ] Test with real video files
- [ ] Benchmark performance vs native

### Week 2: Automation & Documentation

**Days 6-7**: Scripts & Compose
- [ ] Create build.sh script
- [ ] Create run.sh script
- [ ] Write docker-compose.yml
- [ ] Test orchestration

**Days 8-9**: Documentation
- [ ] Write DOCKER_USAGE.md
- [ ] Write DOCKER_TROUBLESHOOTING.md
- [ ] Update main README.md
- [ ] Create example commands

**Day 10**: Final Testing & Deployment
- [ ] Full integration test
- [ ] Performance validation
- [ ] Deploy to production
- [ ] Create release notes

---

## 📞 Support & Maintenance Plan

### Monitoring

- Container health checks every 30 seconds
- Log rotation (10MB max, 3 files)
- GPU utilization tracking
- Performance metrics collection

### Updates

- **Monthly**: Check for base image updates
- **Quarterly**: Update Python dependencies
- **As-needed**: Security patches, bug fixes

### Backup Strategy

- Git repository for all Dockerfiles/configs
- Tag stable releases (`jetson-yolo:v1.0`, `v1.1`, etc.)
- Keep last 3 production images

---

## 🎉 Conclusion

This Docker containerization plan provides a **comprehensive roadmap** for packaging the Jetson YOLO tiled inference system into a reproducible, deployable container.

**Key Benefits**:
- ✅ Simplified deployment (one command)
- ✅ Reproducible environment
- ✅ Isolated dependencies
- ✅ Scalable architecture
- ✅ Minimal performance overhead

**Next Steps**:
1. Review and approve this plan
2. Begin Phase 1 implementation (Dockerfile)
3. Test incrementally on Jetson Orin NX
4. Deploy to production after validation

**Estimated Completion**: 2 weeks  
**Estimated Effort**: 13-20 hours  
**Required Storage**: 10-12 GB  

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Author**: GitHub Copilot  
**Status**: Ready for Implementation  
