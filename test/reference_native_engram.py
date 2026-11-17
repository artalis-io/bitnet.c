"""Capture independent DeepSeek V4.1 Engram token-map and hash references.

Runs original AST bodies from a caller-supplied pinned exllamav3 checkout;
does not import bitnet or execute a production inference provider. Outputs
are immutable: rerunning against an existing directory checks every byte.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import tokenizers
import torch

SOURCE_COMMIT = "bd1f519172406a9560bd5c90e61000691bb69a07"
SOURCE_HASHES = {
    "architecture/deepseek_v41.py": "18e0dc275d300e68ee09e51d890a029907dfd32a2e1c13a53f1b2cc0a3a48aff",
    "modules/ngram_embedding.py": "7affaa5ec9c1a31d2d3d1d95eeb6270db70f65e3e4fb33b10447accda5afd65c",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def execute(nodes, namespace, filename):
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="exllamav3 package directory")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def emit(name, data):
        path = args.output / name
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError(f"refusing to replace differing reference: {path}")
        else:
            path.write_bytes(data)
        manifest[name] = {"bytes": len(data), "sha256": digest(data)}

    def dump(name, value):
        emit(name, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())

    sources = {}

    def tree(relative):
        path = args.source / relative
        raw = path.read_bytes()
        sources[relative] = digest(raw)
        if sources[relative] != SOURCE_HASHES[relative]:
            raise RuntimeError(f"source differs from pinned {SOURCE_COMMIT}: {path}")
        return ast.parse(raw), path

    architecture, arch_path = tree("architecture/deepseek_v41.py")
    embedding, embed_path = tree("modules/ngram_embedding.py")
    namespace = {"torch": torch, "np": np, "lru_cache": lru_cache}
    token_map_fn = next(n for n in architecture.body
                        if isinstance(n, ast.FunctionDef) and n.name == "engram_token_map")
    execute([token_map_fn], namespace, arch_path)
    tokenizer_path = args.checkpoint / "tokenizer.json"
    config_path = args.checkpoint / "config.json"
    checkpoint_hashes = {p.name: digest(p.read_bytes()) for p in (config_path, tokenizer_path)}
    token_map = namespace["engram_token_map"](str(tokenizer_path))
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    config = json.loads(config_path.read_text())["text_config"]
    vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    assert token_map.shape == (vocab,)
    assert int(token_map.max()) + 1 == config["engram_compressed_vocab_size"]
    emit("token-map.i64le", token_map.numpy().astype("<i8").tobytes())

    # Evaluate the source's normalizer assignment itself, preserving its order
    # and regex behavior. Record every decoded string, fallback and final key.
    normalizer_node = next(n for n in token_map_fn.body if isinstance(n, ast.Assign)
                           and isinstance(n.targets[0], ast.Name)
                           and n.targets[0].id == "normalizer")
    normalizer_ns = {"normalizers": tokenizers.normalizers, "Regex": tokenizers.Regex}
    execute([normalizer_node], normalizer_ns, arch_path)
    normalizer = normalizer_ns["normalizer"]
    keys, rows = {}, []
    mismatches = {"nonspacing_marks_only": [], "all_mark_categories": []}
    fallback_count = empty_count = 0
    for token in range(vocab):
        text = tokenizer.decode([token], skip_special_tokens=False)
        raw_fallback = "\ufffd" in text
        normalized = normalizer.normalize_str(text)
        key = tokenizer.id_to_token(token) if raw_fallback else normalized or text
        value = keys.setdefault(key, len(keys))
        assert value == int(token_map[token]), token
        fallback_count += raw_fallback
        empty_count += not normalized and not raw_fallback
        rows.append(json.dumps([token, text, key, raw_fallback], ensure_ascii=False))

        # Diagnostic only: Python Unicode is not the oracle. This quantifies
        # whether a tempting generic normalization replacement is sufficient.
        decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
        for mode in mismatches:
            candidate = "".join(c for c in decomposed if
                                (unicodedata.category(c) != "Mn" if mode == "nonspacing_marks_only"
                                 else not unicodedata.category(c).startswith("M"))).lower()
            candidate = re.sub(r"[ \t\r\n]+", " ", candidate)
            candidate = re.sub(r"^ $", "\ue000", candidate).strip().replace("\ue000", " ")
            candidate_key = tokenizer.id_to_token(token) if raw_fallback else candidate or text
            if candidate_key != key:
                mismatches[mode].append([token, text, key, candidate_key])
    emit("token-keys.jsonl", ("\n".join(rows) + "\n").encode())
    dump("unicode-diagnostic.json", {"unicode_version": unicodedata.unidata_version,
                                    "mismatch_counts": {k: len(v) for k, v in mismatches.items()},
                                    "mismatches": mismatches})

    prime_fn = next(n for n in embedding.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_find_nth_prime_after")
    execute([prime_fn], namespace, embed_path)
    config_class = next(n for n in architecture.body if isinstance(n, ast.ClassDef)
                        and n.name == "DeepseekV41Config")
    constructor = next(n for n in config_class.body if isinstance(n, ast.FunctionDef)
                       and n.name == "__init__")
    start = next(i for i, n in enumerate(constructor.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Attribute)
                 and n.targets[0].attr == "engram_layer_ids")
    stop = next(i for i in range(start, len(constructor.body))
                if isinstance(constructor.body[i], ast.For)) + 1

    class Config:
        @staticmethod
        def read_cfg(kind, name, default):
            value = config.get(name.removeprefix("text_config->"), default)
            assert isinstance(value, kind)
            return value

    resolved = Config()
    namespace.update(self=resolved, p="text_config->")
    execute(constructor.body[start:stop], namespace, arch_path)
    # Prime enumeration is deliberately a single source execution across layers.
    layout = {}
    for layer, (sizes, multipliers) in resolved.engram_hash_params.items():
        offsets = np.cumsum([0] + sizes[:-1]).tolist()
        layout[layer] = {"sizes": sizes, "offsets": offsets, "multipliers": multipliers,
                         "rows": sum(sizes)}
    dump("layout.json", layout)

    embedding_class = next(n for n in embedding.body if isinstance(n, ast.ClassDef)
                           and n.name == "NGramEmbedding")
    methods = [n for n in embedding_class.body if isinstance(n, ast.FunctionDef)
               and n.name in {"_shift_right_ignore_eos", "compute_ngram_ids"}]
    assert len(methods) == 2
    reference_class = ast.ClassDef(name="Reference", bases=[], keywords=[], body=methods,
                                  decorator_list=[])
    execute([reference_class], namespace, embed_path)
    ng = resolved.engram_ngram_size
    context = ng - 1
    pad = int(token_map[resolved.engram_pad_token_id])
    cases = []
    for layer, params in layout.items():
        obj = namespace["Reference"]()
        obj.heads_per_ngram = resolved.engram_n_heads
        obj.ngram_size = ng
        obj.eos_token_id = pad
        obj.layer_multipliers = torch.tensor(params["multipliers"], dtype=torch.int64)
        obj.head_vocab_sizes = torch.tensor(params["sizes"], dtype=torch.int64)
        obj.head_offsets = torch.tensor(params["offsets"], dtype=torch.int64)
        for batch in (1, 2):
            for length in (1, 2, 3, 4, 7, 17, 64, 129, 513):
                for profile in ("text", "pad-boundaries", "repeated"):
                    ids = (torch.arange(batch * length).reshape(batch, length) * 7919 + layer) % vocab
                    if profile == "pad-boundaries":
                        ids[:, ::5] = resolved.engram_pad_token_id
                    elif profile == "repeated":
                        ids.fill_(37)
                    mapped = token_map[ids]
                    history = torch.cat((torch.full((batch, context), pad), mapped), dim=1)
                    output = obj.compute_ngram_ids(history, length)
                    # Source-vs-source chunk equivalence validates the captured
                    # context contract, including boundaries inside carried IDs.
                    pieces = []
                    prior = history[:, :context]
                    for offset in range(0, length, 7):
                        chunk = mapped[:, offset:offset + 7]
                        extended = torch.cat((prior, chunk), dim=1)
                        pieces.append(obj.compute_ngram_ids(extended, chunk.shape[1]))
                        prior = extended[:, -context:]
                    assert torch.equal(output, torch.cat(pieces, dim=1))
                    prefix = f"hash-{layer}-{batch}-{length}-{profile}"
                    for suffix, tensor in (("tokens", ids), ("history", history), ("rows", output)):
                        emit(f"{prefix}-{suffix}.i64le", tensor.numpy().astype("<i8").tobytes())
                    cases.append({"prefix": prefix, "layer": layer, "batch": batch,
                                  "length": length, "heads": output.shape[2]})
    dump("cases.json", cases)
    assert checkpoint_hashes == {p.name: digest(p.read_bytes())
                                 for p in (config_path, tokenizer_path)}, "checkpoint changed during capture"
    dump("provenance.json", {"source_commit": SOURCE_COMMIT, "source_files": sources,
                             "checkpoint_files": checkpoint_hashes,
                             "torch": torch.__version__, "numpy": np.__version__,
                             "tokenizers": tokenizers.__version__, "vocabulary": vocab,
                             "compressed_vocabulary": len(keys), "mapped_pad": pad,
                             "raw_token_fallbacks": fallback_count,
                             "empty_normalization_fallbacks": empty_count,
                             "cases": len(cases), "scope": "CPU token map/hash only; no native runtime qualification"})
    dump("manifest.json", dict(manifest))
    print(json.dumps({"vocabulary": vocab, "compressed": len(keys), "hash_cases": len(cases),
                      "unicode_diagnostic_mismatches": {k: len(v) for k, v in mismatches.items()},
                      "files": len(manifest)}), flush=True)


if __name__ == "__main__":
    main()
