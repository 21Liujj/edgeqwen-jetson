# EdgeQwen - Jetson VLM Runtime

**Qwen2-VL-2B TensorRT Inference Runtime on NVIDIA Jetson Orin**

A production-ready C++ runtime library for vision-language model inference on edge devices, achieving ~24 tokens/second on Jetson Orin Nano Super (8GB).

## Project Status

**✅ Stage 13 COMPLETE - Production Runtime Library**

This project has progressed through 13 stages from initial model conversion to a deliverable C++ runtime library:
- ✅ Model conversion and quantization (FP16)
- ✅ TensorRT engine optimization
- ✅ CUDA kernel profiling and memory optimization
- ✅ Long-term stability validation (28 min / 500 inferences)
- ✅ Production C++ Runtime SDK with clean API

## Architecture

```
Application Code
      ↓
EdgeQwen Runtime (libedgeqwen_runtime.a)
      ↓
┌─────────────┬──────────────┐
│ Visual Enc  │  LLM Engine  │
│ (TensorRT)  │  (TensorRT)  │
└─────────────┴──────────────┘
      ↓
CUDA / TensorRT / Jetson GPU
```

**Key Components:**
- **Visual Encoder**: Qwen2-VL vision tower (TensorRT FP16)
- **LLM Engine**: Qwen2-2B language model with KV cache management
- **Runtime SDK**: Single-header C++ API (`edgeqwen/runtime.h`)

## Performance Metrics

| Metric | Value |
|--------|-------|
| Generation Speed | ~23.8 tokens/second |
| Vision Latency | ~229ms (mean) |
| Prefill Latency | ~173ms (mean) |
| Decode Latency | ~42ms (mean) |
| Peak Unified Memory | ~2.2GB |
| Power Mode | 25W (Jetson Orin Nano Super) |

## Hardware Requirements

- **Platform**: NVIDIA Jetson Orin Nano Super (8GB RAM)
- **JetPack**: 6.2 (L4T R36.4.3)
- **TensorRT**: 10.3.0
- **CUDA**: 12.x
- **Compute Capability**: 8.7 (Orin)

## Quick Start

### 1. Prerequisites

```bash
# Verify JetPack installation
dpkg-query --show nvidia-jetpack

# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake git
```

### 2. Clone and Build

```bash
git clone https://github.com/21Liujj/edgeqwen-jetson.git
cd edgeqwen-jetson

# Build the TensorRT engines (REQUIRED - not included in repo due to size)
# This step downloads models and builds TensorRT engines (~4.6GB)
./build_llm.sh

# Build the runtime library and examples
cd TensorRT-Edge-LLM
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. Run Demo

```bash
# Simple inference example
cd build/examples/runtime_sdk
./edgeqwen_runtime_demo \
  --engines-dir ~/edgeqwen/engines/llm \
  --visual-engines-dir ~/edgeqwen/engines/visual \
  --image /path/to/image.jpg \
  --prompt "Describe this image."
```

## Project Structure

```
edgeqwen/
├── TensorRT-Edge-LLM/          # Core inference engine (submodule)
│   ├── cpp/                    # C++ implementation
│   ├── examples/
│   │   └── runtime_sdk/        # EdgeQwen Runtime SDK
│   │       ├── include/edgeqwen/runtime.h  # Public API
│   │       ├── src/runtime.cpp
│   │       └── demo/
│   └── CMakeLists.txt
├── scripts/                    # Automation and testing scripts
│   ├── preflight_edgeqwen.sh
│   ├── reproduce_benchmarks.sh
│   └── stability_project_driver.sh
├── benchmark/                  # Benchmark runners (VQA, GQA, COCO)
├── docs/                       # Additional documentation
├── build_llm.sh               # Engine build script
└── STAGE*.md                  # Stage completion reports (10-13)
```

## Development Stages

This project evolved through 13 documented stages:

| Stage | Focus | Status |
|-------|-------|--------|
| 1-7 | Model conversion, engine building | ✅ PASS |
| 8-9 | Benchmarking (TextVQA, GQA, VQAv2) | ✅ PASS |
| 10 | CUDA kernel profiling | ✅ PASS |
| 11 | Memory bottleneck analysis | ✅ PASS |
| 12 | Long-term stability (500 inferences) | ✅ PASS |
| 13 | C++ Runtime SDK | ✅ PASS |

See `STAGE10_REPORT.md` through `STAGE13_REPORT.md` for detailed technical reports.

## API Usage

```cpp
#include <edgeqwen/runtime.h>

// Initialize runtime
edgeqwen::EngineConfig config{
    .engineDir = "/path/to/llm/engines",
    .multimodalEngineDir = "/path/to/visual/engines"
};
edgeqwen::Runtime runtime(config);

// Optional warmup
runtime.warmup();

// Prepare input
std::vector<edgeqwen::ChatMessage> messages = {
    {.role = "user", .text = "Describe this image.", 
     .imagePaths = {"/path/to/image.jpg"}}
};

// Generate response
auto result = runtime.generate(messages);
std::cout << result.text << std::endl;

// Get profiling data
auto profile = runtime.getLastProfile();
std::cout << profile.toJson() << std::endl;
```

## Building Engines from Scratch

**Note**: Pre-built engines are NOT included in this repository due to size (~4.6GB). You must build them:

```bash
# Download Qwen2-VL-2B model and convert to TensorRT
./build_llm.sh

# This will:
# 1. Download model weights from HuggingFace
# 2. Convert to ONNX
# 3. Build TensorRT FP16 engines
# 4. Place engines in engines/llm/ and engines/visual/
```

**Expected outputs:**
- `engines/llm/` - Language model engine (~2.3GB)
- `engines/visual/` - Vision encoder engine (~2.3GB)

## Reproducing Benchmarks

```bash
# Run all benchmarks (TextVQA, GQA, VQAv2)
./scripts/reproduce_benchmarks.sh

# Run stability test (500 consecutive inferences)
./scripts/stability_project_driver.sh
```

## Known Limitations

1. **No Multi-turn Conversation**: KV cache reset not exposed; each `generate()` is single-turn
2. **Thread Safety**: Multiple threads share one `Runtime` instance safely, but calls serialize (mutex-protected)
3. **Static Library Only**: No shared library (.so) or C ABI in this release

## Technical Deep Dives

- **STAGE10_REPORT.md**: CUDA kernel profiling and bottleneck identification
- **STAGE11_REPORT.md**: Memory subsystem analysis
- **STAGE12_REPORT.md**: Long-term stability validation methodology
- **STAGE13_REPORT.md**: Runtime library architecture and API design decisions
- **bottleneck_analysis.md**: Comprehensive performance analysis

## License

See `TensorRT-Edge-LLM/LICENSE` for license information.

## Citation

```bibtex
@software{edgeqwen2026,
  title={EdgeQwen: Production VLM Runtime for NVIDIA Jetson},
  author={Liu, Junjie},
  year={2026},
  url={https://github.com/21Liujj/edgeqwen-jetson}
}
```

## Contact

- **Author**: Liu Junjie (21Liujj)
- **Email**: 3154288405@qq.com
- **GitHub**: [@21Liujj](https://github.com/21Liujj)

---

**Built for Edge AI | Optimized for Jetson | Production Ready**
