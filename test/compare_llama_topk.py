#!/usr/bin/env python3
"""Compare bitnet.c top-logit coherence and throughput against llama.cpp.

Start a matching llama-server first, for example:
  llama-server -m model.gguf -ngl 99 -fa on -c 512 -np 1 --host 127.0.0.1 --port 8027

With --benchmark, the default --min-throughput-ratio is 1.0.
Use --check-output-tokens to also require exact generated token ID parity.
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import urllib.request


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In the year 2020, the world",
    "The quick brown fox jumps over the lazy",
    "Once upon a time, there was a",
    "The sum of 2 + 2 =",
    "HTTP status code 404 means",
    "The color of the sky is",
    "Python is a programming language created by",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--bitnet", default="./bitnet",
                   help="bitnet executable to compare")
    p.add_argument("--prompt", action="append", dest="prompts")
    p.add_argument("--prompt-token-ids", action="append", dest="prompt_token_ids",
                   help="comma-separated prompt token IDs; may be repeated")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-overlap", type=int, default=3)
    p.add_argument("--skip-topk", action="store_true",
                   help="skip top-logit checks and only run requested token/benchmark checks")
    p.add_argument("--check-output-tokens", action="store_true")
    p.add_argument("--output-tokens", type=int, default=8)
    p.add_argument("--llama-generation-temperature", type=float, default=0.0)
    p.add_argument("--maxseq", type=int, default=512)
    p.add_argument("-t", "--threads", type=int)
    p.add_argument("--kv16", action="store_true")
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--metal", action="store_true")
    p.add_argument("--llama-metal", action="store_true")
    p.add_argument("--llama-server-url", default="http://127.0.0.1:8027")
    p.add_argument("--flash", action="store_true")
    p.add_argument("--small-dense-exact-native-tail", "--q4-q8-tail-native",
                   dest="small_dense_exact_native_tail", type=int)
    p.add_argument("--small-dense-exact-native-attn-only", "--q4-q8-attn-only",
                   dest="small_dense_exact_native_attn_only",
                   action="store_true")
    p.add_argument("--small-dense-exact-native-ffn-only", "--q4-q8-ffn-only",
                   dest="small_dense_exact_native_ffn_only",
                   action="store_true")
    p.add_argument("--small-dense-exact-native-disable-gateup",
                   "--q4-q8-disable-gateup",
                   dest="small_dense_exact_native_disable_gateup",
                   action="store_true")
    p.add_argument("--small-dense-exact-native-disable-ffn-down",
                   "--q4-q8-disable-ffn-down",
                   dest="small_dense_exact_native_disable_ffn_down",
                   action="store_true")
    p.add_argument("--gpu-flash-min-kv", type=int)
    p.add_argument("--gpu-max-storage-binding-mb", type=int)
    p.add_argument("--metal-disable-barriers", action="store_true")
    p.add_argument("--metal-disable-small-dense-exact-native",
                   "--metal-disable-q4-q8",
                   dest="metal_disable_small_dense_exact_native",
                   action="store_true")
    p.add_argument("--metal-specialized-native-quant", action="store_true")
    p.add_argument("--metal-enable-q6-q8k", action="store_true",
                   dest="metal_specialized_native_quant")
    p.add_argument("--metal-native-quant-prepared", "--metal-q4-prepared",
                   dest="metal_native_quant_prepared", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--bench-tokens", type=int, default=128)
    p.add_argument("--bench-runs", type=int, default=1)
    p.add_argument("--llama-throughput", choices=("server", "bench"),
                   default="server")
    p.add_argument("--min-throughput-ratio", type=float, default=1.0)
    return p.parse_args()


def parse_prompt_token_ids(s):
    try:
        ids = [int(part.strip()) for part in s.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid --prompt-token-ids value: {s!r}") from exc
    if not ids or any(token_id < 0 for token_id in ids):
        raise ValueError(f"invalid --prompt-token-ids value: {s!r}")
    return ids


def prompt_label(prompt):
    if isinstance(prompt, list):
        return "ids:[" + ",".join(str(token_id) for token_id in prompt) + "]"
    return repr(prompt)


def append_bitnet_prompt(cmd, prompt):
    if isinstance(prompt, list):
        token_ids = ",".join(str(token_id) for token_id in prompt)
        cmd += ["--prompt-token-ids", token_ids]
    else:
        cmd += ["-p", prompt]


def append_bitnet_common_args(cmd, args):
    if args.metal:
        cmd.append("--metal")
    if args.kv16:
        cmd.append("--kv16")
    if args.no_prefill:
        cmd.append("--no-prefill")
    if args.threads is not None:
        cmd += ["-t", str(args.threads)]
    if args.flash:
        cmd.append("--flash")
    if args.small_dense_exact_native_tail is not None:
        cmd += ["--small-dense-exact-native-tail",
                str(args.small_dense_exact_native_tail)]
    if args.small_dense_exact_native_attn_only:
        cmd.append("--small-dense-exact-native-attn-only")
    if args.small_dense_exact_native_ffn_only:
        cmd.append("--small-dense-exact-native-ffn-only")
    if args.small_dense_exact_native_disable_gateup:
        cmd.append("--small-dense-exact-native-disable-gateup")
    if args.small_dense_exact_native_disable_ffn_down:
        cmd.append("--small-dense-exact-native-disable-ffn-down")
    if args.gpu_flash_min_kv is not None:
        cmd += ["--gpu-flash-min-kv", str(args.gpu_flash_min_kv)]
    if args.gpu_max_storage_binding_mb is not None:
        cmd += ["--gpu-max-storage-binding-mb",
                str(args.gpu_max_storage_binding_mb)]
    if args.metal_disable_barriers:
        cmd.append("--metal-disable-barriers")
    if args.metal_disable_small_dense_exact_native:
        cmd.append("--metal-disable-small-dense-exact-native")
    if args.metal_specialized_native_quant:
        cmd.append("--metal-specialized-native-quant")
    if args.metal_native_quant_prepared:
        cmd.append("--metal-native-quant-prepared")


def run_bitnet_topk(args, prompt):
    cmd = [
        args.bitnet,
        args.model,
        "-n", "1",
        "--temp", "0",
        "--repeat-penalty", "1",
        "--quiet",
        "--top-logits", str(args.top_k),
        "--maxseq", str(args.maxseq),
    ]
    append_bitnet_prompt(cmd, prompt)
    append_bitnet_common_args(cmd, args)

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    rows = []
    for line in proc.stderr.splitlines():
        m = re.match(
            r"top_logit step=0 rank=(\d+) token=(\d+) logit=([-+0-9.eE]+)",
            line,
        )
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    if len(rows) < args.top_k:
        raise RuntimeError(f"bitnet top-logit dump incomplete for prompt {prompt_label(prompt)}")
    rows.sort()
    return rows


def run_bitnet_tokens(args, prompt):
    cmd = [
        args.bitnet,
        args.model,
        "-n", str(args.output_tokens),
        "--temp", "0",
        "--repeat-penalty", "1",
        "--token-ids",
        "--maxseq", str(args.maxseq),
    ]
    append_bitnet_prompt(cmd, prompt)
    append_bitnet_common_args(cmd, args)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return [int(m.group(1))
            for m in re.finditer(r"(?:^|\s)token_id=(\d+)", proc.stderr)]


def extract_llama_token_ids(payload):
    probs = payload.get("completion_probabilities") or []
    if probs:
        ids = []
        for row in probs:
            token_id = row.get("id")
            if token_id is None:
                return []
            ids.append(int(token_id))
        return ids

    tokens = payload.get("tokens")
    if isinstance(tokens, list):
        ids = []
        for row in tokens:
            if isinstance(row, int):
                ids.append(row)
            elif isinstance(row, dict):
                token_id = row.get("id", row.get("token_id"))
                if token_id is not None:
                    ids.append(int(token_id))
            else:
                return []
        return ids

    ids = []
    for row in probs:
        if "id" in row:
            ids.append(int(row["id"]))
        elif "token" in row and isinstance(row["token"], dict):
            token_id = row["token"].get("id", row["token"].get("token_id"))
            if token_id is not None:
                ids.append(int(token_id))
    return ids


def llama_tokens(server_url, prompt, n_tokens, temperature, n_probs=1):
    body = {
        "prompt": prompt,
        "n_predict": n_tokens,
        "temperature": temperature,
        "repeat_penalty": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
        "n_probs": n_probs,
        "post_sampling_probs": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        server_url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    ids = extract_llama_token_ids(payload)
    if len(ids) < n_tokens:
        raise RuntimeError(f"llama-server returned incomplete token IDs: {payload}")
    return ids[:n_tokens], payload.get("completion_probabilities") or []


def llama_topk(server_url, prompt, top_k):
    body = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": -1,
        "repeat_penalty": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": False,
        "n_probs": top_k,
        "post_sampling_probs": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        server_url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    probs = payload.get("completion_probabilities") or []
    if not probs:
        raise RuntimeError("llama-server did not return completion_probabilities")
    top = probs[0].get("top_logprobs") or []
    if len(top) < top_k:
        raise RuntimeError("llama-server top_logprobs incomplete")
    return [(i + 1, int(row["id"]), float(row["logprob"]))
            for i, row in enumerate(top[:top_k])]


def run_bitnet_bench(args, prompt):
    cmd = [
        args.bitnet, args.model, "-n", str(args.bench_tokens),
        "--quiet", "--temp", "0", "--repeat-penalty", "1",
        "--maxseq", str(args.maxseq),
    ]
    append_bitnet_prompt(cmd, prompt)
    if args.metal:
        cmd.append("--metal")
    if args.flash:
        cmd.append("--flash")
    if args.small_dense_exact_native_tail is not None:
        cmd += ["--small-dense-exact-native-tail",
                str(args.small_dense_exact_native_tail)]
    if args.small_dense_exact_native_attn_only:
        cmd.append("--small-dense-exact-native-attn-only")
    if args.small_dense_exact_native_ffn_only:
        cmd.append("--small-dense-exact-native-ffn-only")
    if args.small_dense_exact_native_disable_gateup:
        cmd.append("--small-dense-exact-native-disable-gateup")
    if args.small_dense_exact_native_disable_ffn_down:
        cmd.append("--small-dense-exact-native-disable-ffn-down")
    if args.gpu_flash_min_kv is not None:
        cmd += ["--gpu-flash-min-kv", str(args.gpu_flash_min_kv)]
    if args.gpu_max_storage_binding_mb is not None:
        cmd += ["--gpu-max-storage-binding-mb",
                str(args.gpu_max_storage_binding_mb)]
    if args.metal_disable_barriers:
        cmd.append("--metal-disable-barriers")
    if args.metal_disable_small_dense_exact_native:
        cmd.append("--metal-disable-small-dense-exact-native")
    if args.metal_specialized_native_quant:
        cmd.append("--metal-specialized-native-quant")
    if args.metal_native_quant_prepared:
        cmd.append("--metal-native-quant-prepared")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    m = re.search(r"tok/s=([0-9.]+)", proc.stderr)
    if not m:
        raise RuntimeError("bitnet throughput not found")
    return float(m.group(1))


def run_llama_bench(args):
    cmd = [
        "llama-bench", "-m", args.model, "-n", str(args.bench_tokens),
        "-p", "0", "-fa", "on" if args.flash else "off",
    ]
    cmd += ["-ngl", "99" if args.llama_metal else "0"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    matches = re.findall(r"tg\d+\s+\|\s+([0-9.]+)\s+±", proc.stdout)
    if not matches:
        raise RuntimeError("llama-bench throughput not found")
    return float(matches[-1])


def run_llama_server_bench(args, prompt):
    body = {
        "prompt": prompt,
        "n_predict": args.bench_tokens,
        "temperature": 0,
        "repeat_penalty": 1.0,
        "cache_prompt": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        args.llama_server_url.rstrip("/") + "/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    timings = payload.get("timings") or {}
    tps = timings.get("predicted_per_second")
    if tps is None:
        predicted_n = timings.get("predicted_n")
        predicted_ms = timings.get("predicted_ms")
        if predicted_n is not None and predicted_ms:
            tps = float(predicted_n) / (float(predicted_ms) / 1000.0)
    if tps is None:
        raise RuntimeError("llama-server throughput not found")
    return float(tps)


def main():
    args = parse_args()
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_token_ids:
        try:
            prompts.extend(parse_prompt_token_ids(ids)
                           for ids in args.prompt_token_ids)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if not prompts:
        prompts = DEFAULT_PROMPTS
    if not args.skip_topk and args.top_k < 1:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    if not args.skip_topk and args.min_overlap > args.top_k:
        print("--min-overlap cannot exceed --top-k", file=sys.stderr)
        return 2
    if args.output_tokens < 1:
        print("--output-tokens must be positive", file=sys.stderr)
        return 2
    if args.bench_runs < 1:
        print("--bench-runs must be positive", file=sys.stderr)
        return 2

    print("Top-logit coherence: bitnet.c vs llama.cpp")
    print(f"Model: {args.model}")
    print(f"llama-server: {args.llama_server_url}")
    if args.skip_topk:
        print("Top-K: skipped")
    else:
        print(f"Top-K: {args.top_k}, required overlap: {args.min_overlap}")
    if args.check_output_tokens:
        print(f"Output token parity: enabled, tokens: {args.output_tokens}")
    print("---")

    failed = 0
    if not args.skip_topk:
        top1_matches = 0
        overlap_total = 0
        for prompt in prompts:
            b_rows = run_bitnet_topk(args, prompt)
            l_rows = llama_topk(args.llama_server_url, prompt, args.top_k)
            b_ids = [r[1] for r in b_rows]
            l_ids = [r[1] for r in l_rows]
            overlap = len(set(b_ids) & set(l_ids))
            overlap_total += overlap
            top1 = b_ids[0] == l_ids[0]
            top1_matches += 1 if top1 else 0
            ok = top1 and overlap >= args.min_overlap
            failed += 0 if ok else 1
            status = "PASS" if ok else "FAIL"
            print(f"{status} {prompt_label(prompt)}: top1 bitnet={b_ids[0]} llama={l_ids[0]} "
                  f"overlap={overlap}/{args.top_k}")
            if not ok:
                print(f"  bitnet top: {[(tid, score) for _, tid, score in b_rows]}")
                print(f"  llama  top: {[(tid, score) for _, tid, score in l_rows]}")

        print("---")
        print(f"Top-1 matches: {top1_matches}/{len(prompts)}")
        print(f"Mean top-{args.top_k} overlap: {overlap_total / len(prompts):.2f}")

    if args.check_output_tokens:
        print("---")
        token_prompts = 0
        token_prefix_total = 0
        token_prefix_compared = 0
        for prompt in prompts:
            bitnet_ids = run_bitnet_tokens(args, prompt)
            llama_ids, llama_probs = llama_tokens(
                args.llama_server_url, prompt, args.output_tokens,
                args.llama_generation_temperature, args.top_k)
            token_prompts += 1
            token_prefix = 0
            for bitnet_id, llama_id in zip(bitnet_ids, llama_ids):
                if bitnet_id != llama_id:
                    break
                token_prefix += 1
            token_prefix_total += token_prefix
            token_prefix_compared += min(len(bitnet_ids), len(llama_ids))
            ok = (len(bitnet_ids) >= args.output_tokens and
                  bitnet_ids[:args.output_tokens] == llama_ids)
            failed += 0 if ok else 1
            status = "PASS" if ok else "FAIL"
            print(f"{status} {prompt_label(prompt)}: token prefix "
                  f"{token_prefix}/{args.output_tokens}")
            if not ok:
                print(f"  bitnet ids: {bitnet_ids[:args.output_tokens]}")
                print(f"  llama  ids: {llama_ids}")
                if llama_probs:
                    for idx, row in enumerate(llama_probs[:args.output_tokens]):
                        top = row.get("top_logprobs") or []
                        compact = [(int(item["id"]), float(item["logprob"]))
                                   for item in top[:args.top_k]]
                        print(f"  llama step {idx} sampled={row.get('id')} top: {compact}")
        print(f"Generated token-ID prefix matches: "
              f"{token_prefix_total}/{token_prefix_compared} tokens "
              f"across {token_prompts} prompts")

    if args.benchmark:
        bitnet_samples = [run_bitnet_bench(args, prompts[0])
                          for _ in range(args.bench_runs)]
        if args.llama_throughput == "bench":
            llama_samples = [run_llama_bench(args)
                             for _ in range(args.bench_runs)]
        else:
            llama_samples = [run_llama_server_bench(args, prompts[0])
                             for _ in range(args.bench_runs)]
        bitnet_tps = statistics.median(bitnet_samples)
        llama_tps = statistics.median(llama_samples)
        ratio = bitnet_tps / llama_tps if llama_tps > 0.0 else 0.0
        print("---")
        print(f"Throughput bitnet={bitnet_tps:.2f} tok/s "
              f"llama={llama_tps:.2f} tok/s "
              f"mode={args.llama_throughput} ratio={ratio:.3f}")
        if args.bench_runs > 1:
            bitnet_csv = ",".join(f"{v:.2f}" for v in bitnet_samples)
            llama_csv = ",".join(f"{v:.2f}" for v in llama_samples)
            print(f"Throughput samples bitnet=[{bitnet_csv}] "
                  f"llama=[{llama_csv}] median_runs={args.bench_runs}")
        if ratio < args.min_throughput_ratio:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
