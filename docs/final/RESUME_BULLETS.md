# EdgeQwen 简历项目描述

**适用岗位**: Edge AI Infrastructure / 端侧大模型部署 / TensorRT Runtime Engineer / Embedded AI

---

## 中文版本 (3-4 条)

### 版本 1: 技术深度强调版

**EdgeQwen: Qwen3-VL-2B 端侧多模态大模型部署**
- 在 Jetson Orin Nano 8GB 上完成 Qwen3-VL-2B (2B params) 的完整部署链路，包括 ONNX 导出、TensorRT FP16 引擎构建和 Runtime 优化，实现 345ms TTFT、18.1 tok/s 解码吞吐，峰值内存 2.65GB
- 通过 Nsight Systems kernel-level profiling 定位 KV-reuse 参数的反直觉行为，发现 reuseKVLen 为 additive context 导致 attention span 从 512 扩展到 1024 tokens，使 FMHA kernel 时间增长 2.27x
- 完成 28 分钟 500 次连续推理稳定性验证，P99 latency drift 4%，零内存泄漏，零热降频；将 Runtime 封装为 C++ 静态库 (Pimpl pattern, 176KB)，实现零内部依赖泄漏
- 解决 CuTe DSL CUDA 版本不兼容导致的 visual encoder 推理失败 (cudaErrorUnknown)，通过本地重编译 TensorRT-Edge-LLM 并显式指定 CUDA 12.6 + Compute Capability 8.7 彻底解决

---

### 版本 2: 工程完整性强调版

**EdgeQwen: Qwen3-VL-2B Jetson Orin Nano 端侧部署工程**
- 从 HuggingFace checkpoint 到 Jetson 端侧推理的完整工程实现：ONNX 导出 (3.8GB) → TensorRT FP16 引擎构建 (3.9GB) → C++ Runtime 封装 (176KB 静态库)，实现 345ms TTFT、18.1 tok/s 吞吐
- 系统性 Runtime 参数消融实验：KV Cache depth 呈 sub-linear scaling (60x depth → 1.11x latency)，Visual resolution O(N^1.8) scaling (GPU O(N^1.3) + CPU O(N²) preprocessing)
- 长时间稳定性验证 (500 requests / 28.1 min)：P50/P95/P99 drift +0.3%/+2.3%/+4.0%，零内存泄漏，零热降频 (cooling device cur_state=0 throughout)
- 解决 CuTe DSL CUDA 13.3/12.6 不兼容问题，定位 fmha_v2_vit_d64 kernel load failure 根因，通过 target-specific rebuild 与 Nsight profiling 验证 KV-reuse 反直觉行为

---

### 版本 3: 问题解决强调版

**EdgeQwen: Qwen3-VL-2B 端侧 VLM 部署与性能优化**
- Jetson Orin Nano 8GB 上实现 Qwen3-VL-2B 端到端推理 (345ms TTFT, 18.1 tok/s)，包含 Visual Encoder + LLM 的 ONNX 导出、TensorRT FP16 引擎构建和 C++ Runtime API 封装
- 通过 Nsight Systems 发现 KV-reuse 导致性能退化 34-75% 的根因：reuseKVLen 为 additive context 而非 subtractive，使 FMHA kernel 从 0.37ms 增长到 0.84ms (2.27x)
- 解决 Stage 7 visual inference cudaErrorUnknown 问题：从 plugin loading error 定位到 CuTe DSL CUDA 版本不兼容 (预编译 artifacts 为 CUDA 13.3, Jetson 为 12.6)，通过本地 rebuild 解决
- 完成生产级验证：500 次推理稳定性测试 (28 min, P99 drift 4%)，C++ 库工程化 (Pimpl pattern, INTERFACE linking, CUDA_SEPARABLE_COMPILATION)，1500+ 行技术文档

---

### 版本 4: 简洁版 (2 条，适合空间受限简历)

**EdgeQwen: Qwen3-VL-2B Jetson 端侧部署**
- Jetson Orin Nano 8GB 上完成 Qwen3-VL-2B 部署链路 (ONNX → TensorRT → Runtime)，实现 345ms TTFT、18.1 tok/s 吞吐，峰值内存 2.65GB；通过 Nsight Systems profiling 发现 KV-reuse additive 行为导致 2.27x FMHA latency 增长
- 解决 CuTe DSL CUDA 版本不兼容 (13.3 vs 12.6) 导致的 visual encoder 失败，完成 500 次推理稳定性验证 (P99 drift 4%, 零内存泄漏)，封装为 C++ 静态库 (Pimpl, 176KB)

---

## English Version (3-4 bullets)

### Version 1: Technical Depth Focus

**EdgeQwen: Qwen3-VL-2B Multimodal LLM Deployment on Edge**
- Completed full deployment pipeline of Qwen3-VL-2B (2B params) on Jetson Orin Nano 8GB, including ONNX export, TensorRT FP16 engine build, and runtime optimization, achieving 345ms TTFT, 18.1 tok/s decode throughput, 2.65GB peak unified memory
- Identified counter-intuitive KV-reuse behavior via Nsight Systems kernel-level profiling: reuseKVLen is additive context (not subtractive), expanding attention span from 512 to 1024 tokens, causing 2.27x FMHA kernel time increase (0.37ms → 0.84ms)
- Validated production-grade stability through 500-request / 28-minute stress test with P99 latency drift 4%, zero memory leak, zero thermal throttling; wrapped runtime into C++ static library (Pimpl pattern, 176KB) with zero internal dependency leak
- Resolved CuTe DSL CUDA compatibility issue causing visual encoder inference failure (cudaErrorUnknown) by rebuilding TensorRT-Edge-LLM locally with explicit CUDA 12.6 + Compute Capability 8.7 flags

---

### Version 2: Engineering Completeness Focus

**EdgeQwen: Qwen3-VL-2B End-to-End Deployment on Jetson Orin Nano**
- Implemented complete engineering pipeline from HuggingFace checkpoint to Jetson edge inference: ONNX export (3.8GB) → TensorRT FP16 engine build (3.9GB) → C++ runtime wrapper (176KB static lib), achieving 345ms TTFT and 18.1 tok/s throughput
- Conducted systematic runtime parameter ablation: KV cache depth shows sub-linear scaling (60x depth → 1.11x latency), visual resolution shows O(N^1.8) scaling (GPU O(N^1.3) + CPU O(N²) preprocessing)
- Validated long-run stability (500 requests / 28.1 min): P50/P95/P99 drift +0.3%/+2.3%/+4.0%, zero memory leak, zero thermal throttling (all cooling devices cur_state=0 throughout)
- Debugged CuTe DSL CUDA 13.3/12.6 incompatibility, traced fmha_v2_vit_d64 kernel load failure to root cause, resolved via target-specific rebuild and verified KV-reuse behavior through Nsight profiling

---

### Version 3: Problem-Solving Focus

**EdgeQwen: Qwen3-VL-2B Edge VLM Deployment & Performance Analysis**
- Deployed Qwen3-VL-2B on Jetson Orin Nano 8GB with end-to-end inference (345ms TTFT, 18.1 tok/s), including Visual Encoder + LLM ONNX export, TensorRT FP16 engine build, and C++ runtime API wrapper
- Discovered KV-reuse performance degradation (34-75% latency increase) root cause via Nsight Systems: reuseKVLen is additive context (not subtractive), causing FMHA kernel time to increase from 0.37ms to 0.84ms (2.27x)
- Resolved Stage 7 visual inference cudaErrorUnknown by tracing plugin loading error to CuTe DSL CUDA version mismatch (prebuilt artifacts for CUDA 13.3, Jetson runs 12.6), fixed through local rebuild
- Delivered production-ready validation: 500-request stability test (28 min, P99 drift 4%), C++ library engineering (Pimpl pattern, INTERFACE linking, CUDA_SEPARABLE_COMPILATION), 1500+ lines technical documentation

---

### Version 4: Concise (2 bullets, space-constrained resumes)

**EdgeQwen: Qwen3-VL-2B Jetson Edge Deployment**
- Completed Qwen3-VL-2B deployment pipeline on Jetson Orin Nano 8GB (ONNX → TensorRT → Runtime) with 345ms TTFT, 18.1 tok/s throughput, 2.65GB peak memory; discovered KV-reuse additive behavior causing 2.27x FMHA latency increase via Nsight Systems profiling
- Resolved CuTe DSL CUDA compatibility issue (13.3 vs 12.6) causing visual encoder failure, validated stability through 500-request test (P99 drift 4%, zero memory leak), wrapped into C++ static library (Pimpl, 176KB)

---

## 关键词密度优化

**面向 ATS (Applicant Tracking System) 的关键词覆盖**:

必备技术关键词:
- ✅ TensorRT / TensorRT-LLM / TensorRT-Edge-LLM
- ✅ ONNX / ONNX Export / ONNX Runtime (对比)
- ✅ CUDA / CUDA Kernel / CUDA Compatibility
- ✅ Jetson / Jetson Orin Nano / Embedded AI / Edge AI
- ✅ Nsight Systems / Nsight Profiling / Kernel Profiling
- ✅ C++ / C++ Library / Static Library / Pimpl Pattern
- ✅ FlashAttention / FMHA / Attention Optimization
- ✅ KV Cache / KV Cache Management / Memory Optimization
- ✅ FP16 / Mixed Precision / Quantization (未做但提及)
- ✅ Vision-Language Model / Multimodal LLM / Qwen3-VL

性能指标关键词:
- ✅ TTFT (Time To First Token)
- ✅ Throughput / Tokens per second / Decode speed
- ✅ Latency / P50 / P95 / P99
- ✅ Memory footprint / Peak memory / Unified Memory
- ✅ Stability / Long-run test / Stress test

工程能力关键词:
- ✅ Debugging / Root cause analysis / Troubleshooting
- ✅ Profiling / Performance analysis / Bottleneck identification
- ✅ Documentation / Technical report / Reproducibility
- ✅ API design / Library engineering / Dependency management
- ✅ CMake / Build system / Cross-compilation

---

## 使用建议

### 选择哪个版本:

**版本 1 (技术深度强调)**:
- 适合: 对 profiling/kernel-level 分析有明确要求的岗位
- 适合: TensorRT Runtime Engineer / Performance Engineer
- 强调: Nsight Systems, FMHA kernel, CUDA debugging

**版本 2 (工程完整性强调)**:
- 适合: Edge AI Infrastructure / MLOps / Deployment Engineer
- 强调: 完整链路, 稳定性验证, 工程化封装

**版本 3 (问题解决强调)**:
- 适合: 综合性岗位 / 强调 troubleshooting 能力
- 强调: CuTe DSL debugging, KV-reuse 发现, 系统性分析

**版本 4 (简洁版)**:
- 适合: 简历空间受限 (如已有多个项目)
- 保留核心亮点: profiling 发现, CUDA debugging, stability

### 如何定制:

1. **看 JD 关键词**: 如果 JD 强调 "TensorRT", "Profiling", "CUDA" → 用版本 1
2. **看岗位级别**: Senior/Staff 级别更看重 troubleshooting → 用版本 3
3. **看公司方向**: Robotics/IoT 公司更关注 edge deployment → 用版本 2
4. **空间约束**: 如果已有 3+ 个项目 → 用版本 4

### 面试时的展开策略:

**30 秒版本** (简历上的bullet points 口述):
- "我在 Jetson 上部署了 Qwen3-VL-2B，做了完整的 profiling 和稳定性验证"

**2 分钟版本** (面试官追问 "说说这个项目"):
- 参考 INTERVIEW_BRIEF.md 的 "2分钟项目介绍"

**5-10 分钟版本** (深度技术面):
- 挑 1-2 个亮点深入讲解 (如 CuTe DSL debugging 或 KV-reuse 发现)
- 展示 Nsight profiling 截图或 CSV 数据
- 讨论 trade-offs 和未来改进方向

---

## 简历格式建议

```markdown
### EdgeQwen: Qwen3-VL-2B 端侧多模态大模型部署  
**技术栈**: TensorRT, ONNX, CUDA, C++, Nsight Systems, CMake  
**时间**: 2026.08 (2 周)  
**环境**: Jetson Orin Nano 8GB (CUDA 12.6, TensorRT 10.3)

- 在 Jetson Orin Nano 8GB 上完成 Qwen3-VL-2B (2B params) 的完整部署链路...
- 通过 Nsight Systems kernel-level profiling 定位 KV-reuse 参数的反直觉行为...
- 完成 28 分钟 500 次连续推理稳定性验证...
- 解决 CuTe DSL CUDA 版本不兼容导致的 visual encoder 推理失败...
```

**为什么标注时间和环境**:
- 时间: 证明高效执行力 (2 周完成完整部署+验证+文档)
- 环境: 展示对硬件约束的理解 (8GB, CUDA 12.6)
- 技术栈: 提高 ATS 匹配率

---

**简历准备完毕！根据目标岗位选择合适版本，面试前复习 INTERVIEW_BRIEF.md**
