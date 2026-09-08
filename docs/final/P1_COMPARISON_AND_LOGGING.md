# P1 Runtime Evidence and Comparison

Date: 2026-08-31

## Jetson Runtime

- Device: Jetson Orin Nano Super, 8 GB unified memory
- Power mode: 25W; L4T R36.4.3
- TensorRT: 10.3.0.30-1+cuda12.5
- Edge-LLM: 0.10.0
- Engine precision: TensorRT FP16 with FP16 KV cache
- Plugin: libNvInfer_edgellm_plugin.so.1.0, aarch64 ELF, 40,656,200 bytes

## Stage-Level Structured Evidence

Clean-process probe evidence is stored in closure_runs/preflight_probe_20260831/. It completed engine deserialization, plugin initialization, and one multimodal request. The profile recorded vision encoder 183.92 ms, LLM prefill 154.58 ms for 501 computed tokens, decode 41.60 ms average per generation run, and peak unified memory 1,576.73 MB. LD_DEBUG=libs recorded the resolved plugin path and TensorRT dependency loading.

The project-local stability driver is scripts/stability_project_driver.sh. It writes batch inputs, timestamps, profile JSON, stdout/stderr, failure records, and tegrastats under closure_runs/stability_run_20260831/. The historical Stage 12 aggregate remains separately identified as report-backed evidence.

## Cloud Versus Jetson Scope

Cloud BF16 merged checkpoint latency was reported as 0.32 s/sample and BitsAndBytes NF4/INT4 as 0.40 s/sample, while Jetson Stage 9 TTFT was about 345 ms. These are different runtimes and workloads; no direct speedup claim is made. Cloud NF4 reduced memory but increased latency by 25%. Jetson uses TensorRT FP16, not TensorRT INT4.

## Limitations

ScienceQA is NOT_EVALUATED because answer labels are absent. Historical 500-request raw /tmp telemetry is unavailable, so its aggregate is not relabeled as a fresh replay. The reproducible default evaluates frozen predictions offline; inference generation is explicit and hardware-dependent. P2 kernel optimization, dynamic/concurrent workloads, backend adapter, and nano-vLLM comparison remain deferred.

## Fresh Stability Evidence (2026-08-31)

The project-local driver completed 25 batches x 20 requests (500/500 requests) in 1701 seconds. All 25 output JSON files and 25 profile JSON files are present; no failures.log was produced. Batch wall time was 67-74 seconds (mean 68.04 seconds). The persisted tegrastats capture contains 1,691 samples: SoC junction temperature 51.718-75.968 C (mean 71.548 C), RAM used 1,830-7,338 MB, and VDD_IN power 4,428-20,584 mW (mean 16,587.9 mW). Full machine-readable evidence is closure_runs/stability_run_20260831/stability_summary.json.
