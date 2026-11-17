#!/usr/bin/env python3
"""Pair independent resumed-prefill token IDs with the bitnet replay tool.

Build test_prefill_reference and separately link llama_layer_probe.cpp to the
intended llama.cpp libraries. CUDA uses the first visible device and F16 KV;
CPU uses F32 KV. Both use eight threads, non-flash attention, and eight greedy
tokens per phase. This is correctness qualification, not a throughput benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def bounded_count(text):
    value = int(text)
    if not 1 <= value <= 8192:
        raise argparse.ArgumentTypeError("count must be between 1 and 8192")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=ROOT / "docs/inference.md")
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--reference-commit",
                        help="require a matching linked GGML Git revision (7-40 hex digits)")
    parser.add_argument("--runtime", type=Path, default=ROOT / "test_prefill_reference")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tokens", type=bounded_count, default=33)
    parser.add_argument("--chunk", type=bounded_count, default=9)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "status": "running", "backend": args.backend,
              "tokens": args.tokens, "chunk": args.chunk, "threads": 8,
              "resume_tokens": 64, "decode_steps": 8,
              "kv": "f16" if args.backend == "cuda" else "f32", "flash": False}

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        model, probe, runtime = [p.resolve(strict=True) for p in
                                 (args.model, args.probe, args.runtime)]
        for label, path in (("probe", probe), ("runtime", runtime)):
            report[label + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["model"] = str(model)
        if args.corpus.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("corpus exceeds replay tool limit")
        data = args.corpus.read_bytes()
        corpus = out / "corpus.txt"
        corpus.write_bytes(data)
        report["corpus_sha256"] = hashlib.sha256(data).hexdigest()
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_PROBE_")}
        env["LLAMA_PROBE_QUIET_LOG"] = "1"
        if args.backend == "cuda":
            env["BN_CUDA_DEVICE"] = "0"
        cuda = ["--cuda"] if args.backend == "cuda" else []
        common = [str(runtime), str(model), str(corpus), str(args.tokens), str(args.chunk)]

        def run(label, cmd):
            report[label + "_command"] = cmd
            save()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            (out / (label + ".log")).write_text(result.stdout + result.stderr)
            report[label + "_returncode"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{label} exited {result.returncode}; see {out / (label + '.log')}")
            return result.stdout

        if args.reference_commit:
            expected = args.reference_commit.lower()
            if not re.fullmatch(r"[0-9a-f]{7,40}", expected):
                raise ValueError("reference commit must contain 7-40 hex digits")
            info = run("reference_info", [str(probe), "--reference-info"])
            commits = re.findall(r"^ggml_commit=([0-9a-fA-F]{7,40})$", info, re.M)
            if len(commits) != 1:
                raise ValueError("probe must report exactly one linked GGML revision")
            actual = commits[0].lower()
            report["reference_commit"] = actual
            report["expected_reference_commit"] = expected
            if not (expected.startswith(actual) or actual.startswith(expected)):
                raise ValueError(f"linked GGML revision {actual} does not match {expected}")

        inputs = run("inputs", common + ["--print-input-ids"] + cuda)
        parsed = {}
        for key, count in (("prompt_token_ids", args.tokens), ("resume_token_ids", 64)):
            matches = re.findall(r"^" + key + r"=([0-9,]+)$", inputs, re.M)
            if len(matches) != 1:
                raise ValueError(f"missing or duplicated {key}")
            ids = [int(x) for x in matches[0].split(",")]
            if len(ids) != count or any(x > 2147483647 for x in ids):
                raise ValueError(f"invalid {key} count/range")
            parsed[key] = matches[0]
        (out / "input-ids.json").write_text(json.dumps(parsed, indent=2) + "\n")
        command = [str(probe), "-m", str(model), "--prompt-token-ids", parsed["prompt_token_ids"],
                   "--resume-token-ids", parsed["resume_token_ids"], "--generate", "8",
                   "--prefill-chunk", str(args.chunk), "--ctx", str(args.tokens + 80),
                   "-t", "8", "--no-flash", "--no-observer"]
        if args.backend == "cuda":
            command += ["--gpu-layers", "99", "--kv-f16"]
        oracle = run("oracle", command)
        initial = re.findall(r"^llama_token_id=(\d+)$", oracle, re.M)
        resumed = re.findall(r"^llama_resumed_token_id=(\d+)$", oracle, re.M)
        boundary = re.findall(r"^llama_resume_prefill pos=(\d+) tokens=(\d+) chunk=(\d+)$", oracle, re.M)
        if len(initial) != 8 or len(resumed) != 8:
            raise ValueError("oracle must emit exactly eight IDs per phase")
        if boundary != [(str(args.tokens + 8), "64", str(args.chunk))]:
            raise ValueError("oracle resumed at the wrong position or chunk schedule")
        reference = out / "reference-ids.txt"
        reference.write_text("\n".join(initial + resumed) + "\n")
        replay = run("runtime", common + [str(reference)] + cuda)
        if "Independent prefill reference PASSED: 16/16 token IDs match" not in replay:
            raise ValueError("runtime did not report complete independent token parity")
        report.update(passed=True, status="complete", matched_tokens=16)
        save()
        print(f"PASS {args.backend}: 16/16 independent IDs; artifacts: {out}")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        report.update(status="failed", error=str(error))
        save()
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
