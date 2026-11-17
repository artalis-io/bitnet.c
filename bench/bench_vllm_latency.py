#!/usr/bin/env python3
"""Loaded-engine greedy token delivery timings; install compatibility fixes first.

ENGINE.json contains LLM keyword arguments. CASES.json is a list of objects with
name, prompt_ids and optional oracle8. No tokenizer or package installer runs in
the measured interval. Use an isolated trusted process for apply_model's RPC.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path


def inspect_compiled_kernels():
    """Describe cached/warmup kernels; this is not a request launch trace.

    vLLM can freeze the GC after initialization, so enumerate the code cache
    directly instead of scanning gc.get_objects().
    """
    import hashlib
    try:
        from torch._inductor import config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except ImportError as error:
        return dict(available=False, reason=str(error), profiles=[])

    modules = []
    for attribute in ("modules", "modules_no_attr", "cache"):
        cache = getattr(PyCodeCache, attribute, ())
        modules.extend(cache.values() if isinstance(cache, dict) else cache)
    seen, profiles = set(), []
    for module in modules:
        for kernel in vars(module).values():
            if id(kernel) in seen or not isinstance(kernel, CachingAutotuner):
                continue
            seen.add(id(kernel))
            selected = getattr(getattr(kernel, "_cached_launcher", None), "config", None)
            filename = getattr(kernel, "filename", None)
            source_hash, source_error = None, None
            if filename:
                try:
                    source_hash = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                except OSError as error:
                    source_error = str(error)
            kwargs = getattr(selected, "kwargs", {})
            kwargs = {str(k): v if v is None or isinstance(v, (bool, int, float, str))
                      else str(v) for k, v in kwargs.items()}
            profiles.append(dict(name=kernel.fn.__name__, source=filename,
                                 source_sha256=source_hash, source_error=source_error,
                                 selected_kwargs=kwargs,
                                 num_warps=getattr(selected, "num_warps", None),
                                 num_stages=getattr(selected, "num_stages", None),
                                 enable_fp_fusion=kernel.triton_meta.get("enable_fp_fusion")))
    return dict(available=True,
                scope="Cached/warmup configurations, not per-request kernel use",
                emulate_precision_casts=getattr(config, "emulate_precision_casts", None),
                profiles=profiles)


def inspect_model(model):
    import collections
    import torch

    def tensors(items):
        return [dict(name=name, device=str(t.device), dtype=str(t.dtype),
                     shape=list(t.shape), bytes=t.numel() * t.element_size())
                for name, t in items]

    methods = collections.Counter()
    gdn_plans = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            methods[type(method).__module__ + "." + type(method).__name__] += 1
        if hasattr(module, "gdn_decode_kernel"):
            gdn_plans.append(dict(name=name,
                                  prefill=getattr(module, "gdn_prefill_backend", None),
                                  decode=module.gdn_decode_kernel,
                                  fused_decode=getattr(module, "enable_fused_gdn_decode", None)))
    return dict(parameters=tensors(model.named_parameters()),
                buffers=tensors(model.named_buffers()), quant_methods=dict(methods),
                gdn_plans=gdn_plans,
                compiled_kernels=inspect_compiled_kernels(),
                compute_precision=dict(
                    allow_bf16_reduced_precision_reduction=
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    allow_fp16_reduced_precision_reduction=
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                    allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    float32_matmul_precision=torch.get_float32_matmul_precision()),
                allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                gpu_name=torch.cuda.get_device_name(0))


def benchmark(engine, cases, outputs, runs, warmups, destination):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    results = {}
    for case in cases:
        samples = []
        for run in range(warmups + runs):
            params = SamplingParams(temperature=0, max_tokens=outputs,
                                    ignore_eos=True, detokenize=False,
                                    output_kind=RequestOutputKind.CUMULATIVE)
            request = f"{case['name']}-{run}"
            start = time.perf_counter()
            engine.add_request(request, {"prompt_token_ids": case["prompt_ids"]}, params)
            tokens, delivery, batches, metric = [], [], [], None
            while engine.has_unfinished_requests():
                updates = engine.step()
                now = time.perf_counter()
                for update in updates:
                    assert update.request_id == request
                    ids = list(update.outputs[0].token_ids)
                    assert ids[:len(tokens)] == tokens
                    if len(ids) > len(tokens):
                        batches.append(len(ids) - len(tokens))
                        delivery.extend([now] * (len(ids) - len(tokens)))
                        tokens = ids
                    if update.finished:
                        metric = update.metrics
            assert len(tokens) == outputs and len(delivery) == outputs
            oracle_match = tokens[:8] == case["oracle8"] if "oracle8" in case else None
            sample = dict(run=run, warmup=run < warmups, oracle_match=oracle_match,
                          ttft_ms=(delivery[0] - start) * 1000,
                          decode_mean_ms=(delivery[-1] - delivery[0]) * 1000 / (outputs - 1),
                          decode_steps_ms=[(b - a) * 1000
                                           for a, b in zip(delivery, delivery[1:])],
                          delivery_batches=batches, tokens=tokens,
                          metrics=dataclasses.asdict(metric)
                          if dataclasses.is_dataclass(metric) else str(metric))
            samples.append(sample)
            results[case["name"]] = dict(prompt_tokens=len(case["prompt_ids"]), samples=samples)
            destination.write_text(json.dumps(results, indent=2, default=str) + "\n")
            print(case["name"], run, sample["ttft_ms"], sample["decode_mean_ms"], flush=True)
            assert oracle_match is not False, (case["name"], tokens[:8], case["oracle8"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--require-fp32-reduction", action="store_true",
                        help="reject workers permitting reduced-precision BF16 reduction; "
                             "configure worker precision before engine initialization")
    args = parser.parse_args()
    if args.outputs < 8 or args.runs < 1 or args.warmups < 0:
        parser.error("require outputs >= 8, runs >= 1, warmups >= 0")
    settings = json.loads(args.engine.read_text())
    cases = json.loads(args.cases.read_text())
    assert cases and len({c["name"] for c in cases}) == len(cases)
    assert all(c["prompt_ids"] and all(type(i) is int and i >= 0
                                      for i in c["prompt_ids"]) for c in cases)
    assert settings.get("skip_tokenizer_init") is True
    assert settings.get("enable_prefix_caching") is False
    assert settings.get("max_num_seqs") == 1
    assert all(len(c["prompt_ids"]) + args.outputs <= settings["max_model_len"] for c in cases)
    from vllm import LLM
    llm = LLM(**settings)
    placement = llm.apply_model(inspect_model)
    args.output.with_suffix(".placement.json").write_text(json.dumps(placement, indent=2) + "\n")
    if args.require_fp32_reduction:
        assert all(not worker["compute_precision"]["allow_bf16_reduced_precision_reduction"]
                   for worker in placement), "BF16 reduction precision differs from native FP32 reduction"
    assert all(str(t["device"]).startswith("cuda") or t["bytes"] <= 8
               for worker in placement for t in worker["parameters"]), "nonresident learned parameter"
    config = llm.llm_engine.vllm_config
    effective = {key: str(getattr(config, key)) for key in
                 ["cache_config", "scheduler_config", "compilation_config",
                  "mamba_config", "quant_config"]}
    args.output.with_suffix(".config.json").write_text(json.dumps(effective, indent=2) + "\n")
    benchmark(llm.llm_engine, cases, args.outputs, args.runs, args.warmups, args.output)


if __name__ == "__main__":
    main()
