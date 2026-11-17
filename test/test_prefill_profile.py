#!/usr/bin/env python3
"""Exercise profiling on real synthetic hybrid dense/MoE execution."""

import math
import os
from pathlib import Path
import subprocess
import sys


def run(binary, enabled):
    env = dict(os.environ)
    env.pop("BN_PREFILL_PROFILE", None)
    if enabled:
        env["BN_PREFILL_PROFILE"] = "1"
    return subprocess.run(
        [str(binary)], env=env, check=True, capture_output=True, text=True
    )


def main():
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "./test_qwen36").resolve()
    control = run(binary, 0)
    profiled = run(binary, 1)
    prefix = "[bn:prefill:profile] "
    assert prefix not in control.stderr
    # The fixture asserts identical KV, SSM, continuation and logits results.
    assert profiled.stdout == control.stdout
    records = [
        dict(field.split("=", 1) for field in line[len(prefix):].split())
        for line in profiled.stderr.splitlines() if line.startswith(prefix)
    ]
    assert records, "profiling must work with zero-worker and pooled runtimes"
    batch = [r for r in records if r["path"] == "batch"]
    assert {r["output"] for r in batch} == {"all", "last", "none"}
    stages = (
        "embed", "attn_norm", "qkv", "attn_cpu", "wo", "ssm", "moe",
        "chain", "attn_chain", "decode", "ffn_norm", "ffn", "residual", "logits",
    )
    for record in records:
        for field in (*stages, "total", "other", "gateup", "act", "down"):
            assert math.isfinite(float(record[field])), (field, record)
        assert float(record["total"]) > 0
        accounted = sum(float(record[field]) for field in stages)
        assert abs(float(record["total"]) - accounted - float(record["other"])) < 0.02
        assert float(record["other"]) >= -0.02, "operator timings must not overlap"
        if record["output"] == "none":
            assert float(record["logits"]) == 0
    for field in ("ssm", "moe"):
        assert any(float(r[field]) > 0 for r in batch), f"missing {field} accounting"
    assert any(float(r["logits"]) > 0 for r in batch if r["output"] == "last")
    print("Prefill profile execution tests PASSED")


if __name__ == "__main__":
    main()
