#!/usr/bin/env python3
"""Compare bitnet.c top-logit coherence and throughput against llama.cpp.

Start a matching llama-server first, for example:
  llama-server -m model.gguf -ngl 99 -fa on -c 512 --host 127.0.0.1 --port 8027
"""

import argparse
import json
import re
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
    p.add_argument("--prompt", action="append", dest="prompts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-overlap", type=int, default=3)
    p.add_argument("--maxseq", type=int, default=512)
    p.add_argument("--metal", action="store_true")
    p.add_argument("--llama-metal", action="store_true")
    p.add_argument("--llama-server-url", default="http://127.0.0.1:8027")
    p.add_argument("--flash", action="store_true")
    p.add_argument("--q4-q8-tail-native", type=int)
    p.add_argument("--q4-q8-attn-only", action="store_true")
    p.add_argument("--q4-q8-ffn-only", action="store_true")
    p.add_argument("--gpu-max-storage-binding-mb", type=int)
    p.add_argument("--metal-disable-q4-q8", action="store_true")
    p.add_argument("--metal-enable-q6-q8k", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--bench-tokens", type=int, default=128)
    p.add_argument("--min-throughput-ratio", type=float, default=1.0)
    return p.parse_args()


def run_bitnet_topk(args, prompt):
    cmd = [
        "./bitnet",
        args.model,
        "-p", prompt,
        "-n", "1",
        "--temp", "0",
        "--repeat-penalty", "1",
        "--quiet",
        "--top-logits", str(args.top_k),
        "--maxseq", str(args.maxseq),
    ]
    if args.metal:
        cmd.append("--metal")
    if args.flash:
        cmd.append("--flash")
    if args.q4_q8_tail_native is not None:
        cmd += ["--q4-q8-tail-native", str(args.q4_q8_tail_native)]
    if args.q4_q8_attn_only:
        cmd.append("--q4-q8-attn-only")
    if args.q4_q8_ffn_only:
        cmd.append("--q4-q8-ffn-only")
    if args.gpu_max_storage_binding_mb is not None:
        cmd += ["--gpu-max-storage-binding-mb",
                str(args.gpu_max_storage_binding_mb)]
    if args.metal_disable_q4_q8:
        cmd.append("--metal-disable-q4-q8")
    if args.metal_enable_q6_q8k:
        cmd.append("--metal-enable-q6-q8k")

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
        raise RuntimeError(f"bitnet top-logit dump incomplete for prompt {prompt!r}")
    rows.sort()
    return rows


def llama_topk(server_url, prompt, top_k):
    body = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": False,
        "n_probs": top_k,
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
        "./bitnet", args.model, "-p", prompt, "-n", str(args.bench_tokens),
        "--quiet", "--maxseq", str(args.maxseq),
    ]
    if args.metal:
        cmd.append("--metal")
    if args.flash:
        cmd.append("--flash")
    if args.q4_q8_tail_native is not None:
        cmd += ["--q4-q8-tail-native", str(args.q4_q8_tail_native)]
    if args.q4_q8_attn_only:
        cmd.append("--q4-q8-attn-only")
    if args.q4_q8_ffn_only:
        cmd.append("--q4-q8-ffn-only")
    if args.gpu_max_storage_binding_mb is not None:
        cmd += ["--gpu-max-storage-binding-mb",
                str(args.gpu_max_storage_binding_mb)]
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


def main():
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    if args.top_k < 1:
        print("--top-k must be positive", file=sys.stderr)
        return 2
    if args.min_overlap > args.top_k:
        print("--min-overlap cannot exceed --top-k", file=sys.stderr)
        return 2

    print("Top-logit coherence: bitnet.c vs llama.cpp")
    print(f"Model: {args.model}")
    print(f"llama-server: {args.llama_server_url}")
    print(f"Top-K: {args.top_k}, required overlap: {args.min_overlap}")
    print("---")

    top1_matches = 0
    overlap_total = 0
    failed = 0
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
        print(f"{status} {prompt!r}: top1 bitnet={b_ids[0]} llama={l_ids[0]} "
              f"overlap={overlap}/{args.top_k}")
        if not ok:
            print(f"  bitnet top: {b_ids}")
            print(f"  llama  top: {l_ids}")

    print("---")
    print(f"Top-1 matches: {top1_matches}/{len(prompts)}")
    print(f"Mean top-{args.top_k} overlap: {overlap_total / len(prompts):.2f}")

    if args.benchmark:
        bitnet_tps = run_bitnet_bench(args, prompts[0])
        llama_tps = run_llama_bench(args)
        ratio = bitnet_tps / llama_tps if llama_tps > 0.0 else 0.0
        print("---")
        print(f"Throughput bitnet={bitnet_tps:.2f} tok/s "
              f"llama={llama_tps:.2f} tok/s ratio={ratio:.3f}")
        if ratio < args.min_throughput_ratio:
            failed += 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
