# ENGINE_PROVENANCE_AUDIT

**Date**: 2026-08-28  
**Auditor**: Claude (EdgeQwen Project Final Delivery Review)  
**Scope**: `/home/nvidia/edgeqwen/engines/llm/llm.engine` provenance chain

---

## Executive Summary

**CRITICAL FINDING**: The existing Jetson `llm.engine` (Stage 7-13) **WAS BUILT FROM LORA-MERGED ONNX**, not base model. The current rebuild attempt was triggered by a **FALSE ASSUMPTION** that Jetson was running base model.

**Status**: Rebuild is **NOT NECESSARY**. Current OOM is classified as **TENSORRT_BUILDER_MEMORY_LIMIT**, not a MODEL_RUNTIME_LIMIT issue.

---

## 1. Existing llm.engine Source

**File**: `/home/nvidia/edgeqwen/engines/llm/llm.engine`  
**Size**: 3.3 GB (3,454,390,164 bytes)  
**Created**: 2026-08-25 16:55:38 (Birth: 16:55:23)  
**Build log**: `/home/nvidia/edgeqwen/logs/stage7_llm_build_tmux.log`  
**Build command**: (from script `/home/nvidia/edgeqwen/build_llm.sh`)
```bash
llm_build \
  --onnxDir /home/nvidia/edgeqwen/onnx/llm \
  --engineDir /home/nvidia/edgeqwen/engines/llm \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 1
```

**Input ONNX path**: `/home/nvidia/edgeqwen/onnx/llm/`

---

## 2. Source ONNX SHA256

**LLM ONNX components**:
```
22bbc5cf4ed4ae9c0fbf4bb6ebe4e5dd15a2f44a0f3116f39ab0163dbf026749  model.onnx
cd1d2e3b20477fa0d68a29565f7b9169a16a22e0c64ee5147b88ecae575aa597  model.onnx.data
af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5  embedding.safetensors
```

**Visual ONNX components**:
```
6c8ac386f3ea5b8e60bc0d8451ec9ec85e99a3d70e1d4d1d8e61d01bca82f1c6  model.onnx
eb7ec7df7718225cf9682cad7d572e1453fe2c83f901401648d9385e78bbe83b  model.onnx.data
```

**ONNX timestamp**: 2026-08-24 18:23 (all components)  
**Checksum file preserved**: `/home/nvidia/edgeqwen/onnx/checksums.sha256`

---

## 3. Source Checkpoint

**Cloud-side Stage 6 metadata** (`/root/autodl-tmp/edgeqwen/outputs/stage6_onnx_export/onnx_export_metadata.json`):
```json
{
  "source_checkpoint": "/root/autodl-tmp/edgeqwen/artifacts/merged_models/domain_lora_v1_merged",
  "onnx_output_dir": "/root/autodl-tmp/edgeqwen/artifacts/onnx/domain_lora_v1",
  "export_time_sec": 83.65,
  "dtype": "float16"
}
```

**Cloud LoRA-merged model.safetensors SHA256**:
```
3b0ff0abc1804c88f40577bf3314a75f386feeae3172c56e95ef5954ec15f2c0
```

**Cloud base model.safetensors SHA256** (for comparison):
```
7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0
```

**SHA256 Mismatch confirms**: LoRA-merged checkpoint ≠ base checkpoint.

---

## 4. Is it LoRA-merged: **YES**

**Evidence chain**:
1. Cloud Stage 4 merged `base + LoRA adapter` → `domain_lora_v1_merged` (SHA256: `3b0ff0a...`)
2. Cloud Stage 6 exported ONNX from `domain_lora_v1_merged` (metadata explicitly records source)
3. Cloud ONNX SHA256 matches Jetson ONNX SHA256 **EXACTLY** (all 5 files: llm model.onnx, model.onnx.data, embedding.safetensors, visual model.onnx, visual model.onnx.data)
4. Jetson Stage 7 built `llm.engine` from this ONNX on 2026-08-25 16:55
5. Jetson Stage 9-13 benchmarked/profiled this engine

**Transfer method**: Unknown (no explicit transfer log found), but SHA256 match proves provenance.

**Jetson base model.safetensors** at `/home/nvidia/edgeqwen/models/Qwen3-VL-2B-Instruct/model.safetensors`:
- SHA256: `7de1838...` (matches cloud base, NOT merged)
- Timestamp: 2026-08-26 11:24 (**downloaded AFTER Stage 7 engine build**)
- **Never used for ONNX export or engine build** (no corresponding ONNX with this SHA256)

---

## 5. Evidence

### 5.1 SHA256 Match Proof

**Jetson `/home/nvidia/edgeqwen/onnx/llm/`**:
```
22bbc5cf4ed4ae9c0fbf4bb6ebe4e5dd15a2f44a0f3116f39ab0163dbf026749  model.onnx
cd1d2e3b20477fa0d68a29565f7b9169a16a22e0c64ee5147b88ecae575aa597  model.onnx.data
af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5  embedding.safetensors
```

**Cloud `/root/autodl-tmp/edgeqwen/artifacts/onnx/domain_lora_v1/llm/`**:
```
22bbc5cf4ed4ae9c0fbf4bb6ebe4e5dd15a2f44a0f3116f39ab0163dbf026749  model.onnx
cd1d2e3b20477fa0d68a29565f7b9169a16a22e0c64ee5147b88ecae575aa597  model.onnx.data
af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5  embedding.safetensors
```

**Jetson `/home/nvidia/edgeqwen/onnx_lora/llm/` (today's transfer)**:
```
22bbc5cf4ed4ae9c0fbf4bb6ebe4e5dd15a2f44a0f3116f39ab0163dbf026749  model.onnx
cd1d2e3b20477fa0d68a29565f7b9169a16a22e0c64ee5147b88ecae575aa597  model.onnx.data
af9dbce9b96ca45a1cdc7c5ebdd18dc780567e707388e7cead78fbf1adb256d5  embedding.safetensors
```

**All three match → Same ONNX artifact.**

### 5.2 Timeline Evidence

- **2026-08-24 17:21-18:23**: Cloud Stage 6 ONNX export (from LoRA-merged)
- **2026-08-24 18:23**: Jetson `/home/nvidia/edgeqwen/onnx/llm/` files created (same timestamp as cloud export completion → immediate transfer)
- **2026-08-25 16:55**: Jetson `llm.engine` built from this ONNX
- **2026-08-26 11:24**: Jetson base model downloaded (AFTER engine build, never used)
- **2026-08-27**: Stage 9-13 benchmarks/profiling (using LoRA-merged engine)
- **2026-08-28 09:34**: Today's `onnx_lora/` transfer (duplicate, SHA256-identical to existing `onnx/`)

### 5.3 Stage 7 Build Log Evidence

`/home/nvidia/edgeqwen/logs/stage7_llm_build_tmux.log` line 8:
```
[16:01:11.830] [INFO] [llmBuilder.cpp:175:build] Parsing ONNX model: /home/nvidia/edgeqwen/onnx/llm/model.onnx
```

Peak GPU memory during build: 7022 MiB (from Stage 7 successful build)

### 5.4 Jetson FINAL_REPORT Error

`/home/nvidia/edgeqwen/docs/final/FINAL_REPORT.md` line 152:
```
| **Stage 1-6** | Cloud Training Pipeline | NOT EXECUTED | 项目直接在 Jetson 上部署 HF base model，跳过 LoRA/Quantization | ...
```

**This line is FACTUALLY INCORRECT.**

Correction:
```
| **Stage 1-6** | Cloud Training Pipeline | PASS | LoRA training, merge, ONNX export completed on cloud; ONNX transferred to Jetson for Stage 7 | Cloud STAGE*_REPORT.md
```

---

## 6. Why Current Rebuild Was Triggered

**Root cause**: Auditor (Claude) **misread FINAL_REPORT.md line 152** and assumed:
1. Jetson used base model ONNX
2. Today's transfer was introducing LoRA-merged ONNX for the first time
3. Rebuild was necessary to "打通云→边链路"

**Actual situation**: 
1. Jetson **already had** LoRA-merged ONNX since 2026-08-24
2. Jetson Stage 7-13 **already ran** on LoRA-merged engine
3. Today's transfer was a **duplicate** (SHA256-identical)
4. Rebuild attempt was **redundant**

**False documentation propagated the error**:
- FINAL_REPORT.md §3 Stage Timeline table: "Stage 1-6: NOT EXECUTED"
- FINAL_REPORT.md §4.2: "本项目未执行以下阶段: LoRA fine-tuning, LoRA merge, INT4 quantization"
- INTERVIEW_BRIEF.md Q5: "本项目未执行 LoRA fine-tuning"

All three statements are **contradicted by SHA256 evidence**.

---

## 7. Is Rebuild Actually Necessary: **NO**

**Current state**:
- ✅ Jetson `llm.engine` is built from LoRA-merged ONNX
- ✅ Stage 9-13 benchmarks/profiling are valid for LoRA-merged model
- ✅ `onnx/` and `onnx_lora/` are SHA256-identical (no new artifact)

**Rebuild would only**:
- Reproduce the exact same engine (same ONNX input)
- Waste 15+ minutes and risk OOM again
- Provide zero new information

**What IS necessary**:
- Update FINAL_REPORT.md to correct "Stage 1-6: NOT EXECUTED" → "Stage 1-6: PASS (Cloud)"
- Update INTERVIEW_BRIEF.md to reflect actual LoRA training results
- Add provenance chain documentation to explain cloud→Jetson transfer

---

## 8. Current OOM Should Be Classified As

**Classification**: `TENSORRT_BUILDER_MEMORY_LIMIT`

**Rationale**:
1. **Not MODEL_RUNTIME_LIMIT**: Stage 9-13 runtime inference succeeded with LoRA-merged engine (peak 2.65GB RAM, well under 8GB)
2. **Builder-specific issue**: OOM occurred during `llm_build` autotuning phase, not runtime
3. **Evidence from dmesg**:
   - PID 8787: anon-rss 3.9GB + file-rss 362MB (build process)
   - PID 9174: anon-rss 3.8GB + file-rss 363MB (build process)
   - Both killed during builder tactic profiling (GPU 7586 MiB peak in debug logs)
4. **Same ONNX built successfully on 2026-08-25**: Peak 7022 MiB, under 8GB threshold
5. **Today's rebuild attempt hit 7586 MiB**: 564 MiB higher, pushed over OOM threshold

**Why the 564 MiB difference**:
- Non-deterministic TensorRT builder tactic selection
- Different memory fragmentation state (previous successful build may have had cleaner memory layout)
- Timing cache differences (first build vs rebuild)
- Background processes (current system had slightly less free memory at start)

**Conclusion**: This is a **borderline OOM** specific to TensorRT Builder autotuning phase on 8GB Jetson, not a fundamental incompatibility with LoRA-merged model. The successful Stage 7 build on 2026-08-25 proves the model CAN be built on this hardware.

---

## 9. Recommendations

### 9.1 Immediate Actions

1. **Stop all rebuild attempts** ✅ (already done)
2. **Delete duplicate `onnx_lora/` directory** (4.6GB waste)
   ```bash
   rm -rf /home/nvidia/edgeqwen/onnx_lora
   ```
3. **Correct FINAL_REPORT.md Stage Timeline**:
   - Replace "Stage 1-6: NOT EXECUTED" with full cloud stage table
   - Add cloud Stage 3-6 sections (LoRA training, Merge, Quantization, ONNX export)
   - Add "Deployment Provenance" section explaining cloud→Jetson transfer

### 9.2 Documentation Updates Required

**Files to update**:
1. `FINAL_REPORT.md`:
   - §3 Stage Timeline: Add cloud Stages 1-6 with PASS status
   - §4 Model Adaptation: Replace "未执行" with actual LoRA training results
   - New §5 "Cloud Training Pipeline": Full Stage 3-6 narrative from cloud STAGE*_REPORT.md
   - New §6 "Deployment Provenance": SHA256-based evidence chain

2. `INTERVIEW_BRIEF.md`:
   - Q5 "LoRA 做了什么": Replace "本项目未执行" with real results (8.7M params, 70%→83% accuracy)
   - Q6 "为什么要 Merge": Add real numbers (8.70GB→4.38GB VRAM)
   - Q7 "INT4 为什么没变快": Add BitsAndBytes NF4 real data (+25% latency, -58.8% memory)

3. `METRICS_SUMMARY.md`:
   - Add new section "Cloud Training Metrics (Stage 1-6)"
   - Add LoRA training table, Merge regression table, INT4 regression table

4. `README.md`:
   - Update Key Results to mention cloud LoRA training
   - Clarify that Jetson deployment used LoRA-merged model (not base)

5. `STAGE15_REPORT.md`:
   - Update Stage Map with cloud stages marked PASS/PASS WITH CAVEAT
   - Correct "Cloud vs Jetson Pipeline Discontinuity" → "Cloud-Jetson Integrated Pipeline"

### 9.3 What NOT to Do

- ❌ Do not rebuild engine (already exists, SHA256-proven provenance)
- ❌ Do not claim "LoRA-merged cannot deploy on 8GB Jetson" (Stage 7-13 prove it can)
- ❌ Do not retry today's OOM build (non-deterministic, not worth the risk)
- ❌ Do not fabricate new evidence (all numbers must come from existing cloud/Jetson reports)

---

## 10. Audit Conclusion

**Status**: ✅ **AUDIT COMPLETE**

**Finding**: The EdgeQwen project **successfully completed** the full cloud→Jetson deployment pipeline:
- Cloud Stage 1-6: LoRA training → Merge → Quantization eval → ONNX export
- Transfer: Cloud ONNX → Jetson (SHA256-verified, timestamp 2026-08-24 18:23)
- Jetson Stage 7-13: TensorRT build → Runtime → Benchmark → Profiling → Stability → C++ API

The "Stage 1-6: NOT EXECUTED" documentation was **incorrect**. All cloud stages were executed and their artifacts were successfully deployed to Jetson.

**Next step**: Update final documentation to reflect this integrated cloud-edge pipeline, using the cloud STAGE*_REPORT.md files as evidence sources.

---

**Audit signature**: Claude Sonnet 5  
**Audit date**: 2026-08-28 10:26 UTC+8  
**Evidence preservation**: All SHA256 checksums, timestamps, and log excerpts recorded above
