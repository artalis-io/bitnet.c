"""Audit the pinned upstream text-forward ordering without importing torch.

Usage: python3 test/reference_native_forward_order.py /path/to/exllamav3
Prints source locations and an orchestration contract, not numerical fixtures
or evidence that the native runtime implements a complete forward.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


SOURCES = {
    "architecture/deepseek_v4.py": "c7e0199c9002e929903d25739bd9e1d3c4dbec2b1ac7a20fcd077b50f6795e83",
    "architecture/deepseek_v41.py": "18e0dc275d300e68ee09e51d890a029907dfd32a2e1c13a53f1b2cc0a3a48aff",
    "modules/transformer.py": "a9a0a83dca3068018997587f3e8a5c49f4a4ae6fbc0640ab88ef2a9537777263",
    "modules/hyperconnections.py": "4d1d2af0f666e91c7059048d874923df8afe3beed0bbab4790455a71c4f484d7",
    "modules/ple.py": "c22d60b3d7d6765e37685a6383b54a2233f6063d2500dcb638f3c83010e598fc",
    "modules/rmsnorm.py": "2145364078c17de81291a9b786b6517a880d6af79bec9ce6ff2f7686a3b41273",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def method(tree, cls, name):
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
    require(len(classes) == 1, f"expected one class {cls}")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(methods) == 1, f"expected one method {cls}.{name}")
    return methods[0]


def calls(node, function):
    return sorted(
        [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == function],
        key=lambda n: (n.lineno, n.col_offset),
    )


def one_call(node, function):
    found = calls(node, function)
    require(len(found) == 1, f"expected one call to {function}")
    return found[0]


def audit(root):
    trees = {}
    for relative, expected in SOURCES.items():
        data = (root / relative).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, f"source digest mismatch: {relative}")
        trees[relative] = ast.parse(data, filename=relative)

    evidence = []

    def record(file, cls, name, rule):
        node = method(trees[file], cls, name)
        evidence.append({"file": file, "symbol": f"{cls}.{name}", "line": node.lineno,
                         "end_line": node.end_lineno, "rule": rule})
        return node

    assembly = record("architecture/deepseek_v4.py", "DeepseekV4Model", "__init__",
                      "Embedding, ExpandStreams, per-layer pre-block then TransformerBlock, HyperHead, final norm, head")
    assembly_order = ["Embedding", "ExpandStreams", "self.make_pre_block", "TransformerBlock",
                      "HyperHead", "RMSNorm", "Linear"]
    # Two RMSNorm constructors are inside TransformerBlock; the last one is
    # the model's final norm, following HyperHead.
    norms = calls(assembly, "RMSNorm")
    require(len(norms) == 3, "unexpected number of block/final norm constructors")
    sites = [(norms[-1] if name == "RMSNorm" else one_call(assembly, name)).lineno
             for name in assembly_order]
    require(sites == sorted(sites), "unexpected module construction order")

    prepare = record("architecture/deepseek_v41.py", "DeepseekV41Model", "prepare_inputs",
                     "Reset hc_pre to None at each forward, including resumed forwards")
    require(any(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "params['hc_pre']"
                and isinstance(n.value, ast.Constant) and n.value.value is None
                for n in ast.walk(prepare)), "missing per-forward carry reset")

    preblock = record("architecture/deepseek_v41.py", "DeepseekV41Model", "make_pre_block",
                      "Engram is a pre-block module selected by engram_layer_ids")
    one_call(preblock, "PLELayer")

    ple = record("modules/ple.py", "PLELayer", "forward",
                 "Flush deferred HC residual before preparing IDs/history or evaluating Engram")
    require(one_call(ple, "hc_flush").lineno < one_call(ple, "self._prepare_ids").lineno
            < one_call(ple, "self._state_history").lineno, "unexpected Engram ordering")

    block = record("modules/transformer.py", "TransformerBlock", "forward",
                   "Attention HC mix, F32-input norm to F16, attention, HC apply; then FFN equivalent")
    for site, branch, norm in [("attn_hc", "attn", "attn_norm"), ("mlp_hc", "mlp", "mlp_norm")]:
        mix = one_call(block, f"self.{site}.mix_norm")
        apply = one_call(block, f"self.{site}.apply_")
        normalizations = [n for n in calls(block, f"self.{norm}.forward")
                          if mix.lineno < n.lineno < apply.lineno]
        require(normalizations, f"missing {site} normalization")
        first = normalizations[0]
        require(first.args and isinstance(first.args[0], ast.Name) and first.args[0].id == "y"
                and any(k.arg == "out_dtype" and ast.unparse(k.value) == "torch.half"
                        for k in first.keywords), f"wrong {site} normalization precision")
        branch_calls = calls(block, f"self.{branch}.forward")
        require(any(first.lineno < n.lineno < apply.lineno for n in branch_calls),
                f"unexpected {branch} execution order")
    require(one_call(block, "self.attn_hc.apply_").lineno
            < one_call(block, "self.mlp_hc.mix_norm").lineno, "FFN precedes attention residual")

    mix = record("modules/hyperconnections.py", "HyperConnection", "_mix_unfused",
                 "Carry consumes previous pre, publishes current pre; first site collapses stream zero; small R uses half fn")
    carry = one_call(mix, "ext.hc_mix_carry")
    require(len(carry.args) == 13 and ast.unparse(carry.args[-2]) == "pre_in"
            and ast.unparse(carry.args[-1]) == "pre_out", "unexpected carry kernel signature")
    require(one_call(mix, "self.fn.half").lineno < carry.lineno, "missing small-row fn cast")
    record("modules/hyperconnections.py", "HyperConnection", "_carry",
           "Fallback confirms previous-site carry, with stream-zero initial collapse")
    record("modules/hyperconnections.py", "HyperConnection", "apply_",
           "Residual is post times branch plus transposed combination times streams; optional deferral must land before any reader")
    head = record("modules/hyperconnections.py", "HyperHead", "forward",
                  "Flush residual, then vendor matmul using final site's pre; no new HC site or mean collapse")
    require(one_call(head, "hc_flush").lineno < calls(head, "torch.matmul")[0].lineno,
            "head reads unflushed streams")
    record("modules/rmsnorm.py", "RMSNorm", "forward",
           "Normalize the original F32 collapse and request F16 output; do not cast input first")

    return {
        "status": "SOURCE ORDER AUDITED; runtime composition and numerical qualification remain open",
        "upstream_commit": "bd1f519172406a9560bd5c90e61000691bb69a07",
        "scope": "Ordinary text, carried-pre DeepSeek V4.1; no TP, multimodal, draft overrides or conversion passes",
        "native_adapter_limitation": "The qualified native Engram component explicitly adapts Diffbot INT4 BF16 lookup to Exllama F16 math; source ordering does not establish equivalence to the original Diffbot full forward",
        "source_sha256": SOURCES,
        "evidence": evidence,
        "forward_order": [
            "Stage the whole raw/mapped token window once; reset HC carry for this forward",
            "Gather BF16 embedding and expand to F32 streams [R,4,D]",
            "For each layer in checkpoint order: finish previous residual; run optional Engram on current streams",
            "Mix attention HC: collapse with previous pre (stream zero initially), publish own pre/post/comb",
            "Normalize F32 collapse with layer attention norm to F16; run attention producer/consumer",
            "Apply attention branch with that site's post/comb to F32 streams",
            "Mix FFN HC: collapse with attention pre, publish FFN pre/post/comb",
            "Normalize F32 collapse with layer FFN norm to F16; run source learned-router FFN",
            "Apply FFN branch with that site's post/comb to F32 streams; retain FFN pre for next layer",
            "Finish final residual; HyperHead matmul collapses with final FFN pre",
            "Normalize F32 final collapse to F16; project vocabulary logits",
            "After successful whole-model completion, publish logits and commit host history/position",
        ],
        "composition_requirements": [
            "Use one canonical request for all resident views and backend-owned immutable weights",
            "Ping-pong two disjoint F32 pre buffers; do not clear carry between sites or at Engram layers",
            "Preserve post/comb until the matching branch residual has consumed them",
            "Consume Engram out as the next stream stack; Engram does not advance HC carry",
            "Keep batch geometry: R<=32 changes HC fn precision; arbitrary row tiling is not equivalent",
            "Budget HC scratch and final norm/head workspace in addition to all existing component quotas",
            "Existing lower-level visibility commits are not whole-model commit or reversible snapshots",
            "After partial execution failure, invalidate the request and require reset/replay; host abort alone cannot undo KV/index/SSM writes",
            "Do not expose native full-forward admission until independent full-chain and generation gates pass",
        ],
        "qualification_required": [
            "Independent original-source captures of actual connected activations, not synthetic branch substitutes",
            "Every layer's Engram, HC pre/post/comb/collapse, normalized input, attention/FFN output and residual streams",
            "R1/7/17/32/33/129/513 matched-shape short/long/chunked/resumed forward and decode",
            "Final carried head collapse, normalization, logits and sampled token IDs",
            "Two interleaved sessions, history reset, BUSY handling, late failures and reset/replay recovery",
            "Allocation/copy/fence profile and aggregate bounded memory; no vLLM parity claim from operator timing",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.source_root)
    except (OSError, SyntaxError, ValueError) as exc:
        parser.exit(1, f"source-order audit failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
