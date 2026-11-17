"""Extract a payload-free GLM GGUF directory fixture; never read weight values."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import gguf

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("--header", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
args = parser.parse_args()
root = gguf.GGUFReader(str(args.model))
count = int(root.fields["split.count"].contents())
match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", str(args.model))
assert match and int(match[2]) == count
metadata = {k: v for k, v in root.fields.items()
            if k == "general.architecture" or k.startswith("glm5next.")}
lines = ["/* Original GGUF metadata/directory only; no model payload required. */",
         "#ifndef BN_GLM_ANATOMY_REFERENCE_H", "#define BN_GLM_ANATOMY_REFERENCE_H"]
kvs = []
for index, (name, field) in enumerate(metadata.items()):
    value = field.contents()
    kind = int(field.types[0])
    if kind == 9:
        elem = int(field.types[1])
        typename = {4: "uint32_t", 5: "int32_t", 6: "float"}[elem]
        values = ",".join(float(x).hex() + "f" if elem == 6 else str(x) for x in value)
        lines.append(f"static {typename} glm_anatomy_array_{index}[] = {{{values}}};")
        body = ".arr={.elem_type=%d,.n=%d,.data=glm_anatomy_array_%d}" % (elem, len(value), index)
    elif kind == 8:
        body = ".str={.len=%d,.str=%s}" % (len(value), json.dumps(value))
    elif kind == 6:
        body = ".f32=" + float(value).hex() + "f"
    elif kind == 7:
        body = ".b=" + str(int(value))
    else:
        assert kind in [4, 5]
        body = (".u32=" if kind == 4 else ".i32=") + str(value)
    kvs.append(" {.key=%s,.type=%d,.value={%s}}," % (json.dumps(name), kind, body))
lines += ["static BnGGUFKeyValue glm_anatomy_reference_kvs[] = {", *kvs, "};",
          "static BnGGUFTensorInfo glm_anatomy_reference_tensors[] = {"]
tensors = []
shards = []
for shard in range(count):
    path = Path(f"{match[1]}-{shard+1:05d}-of-{count:05d}.gguf")
    reader = root if shard == 0 else gguf.GGUFReader(str(path))
    shards.append(dict(path=str(path), tensors=len(reader.tensors), bytes=path.stat().st_size))
    for tensor in reader.tensors:
        shape = [int(d) for d in tensor.shape]
        kind = int(tensor.tensor_type)
        tensors.append(dict(name=tensor.name, shape=shape, type=kind, shard=shard))
        lines.append(" {.name=%s,.n_dims=%d,.dims={%s},.type=%d,.shard_idx=%d}," %
                     (json.dumps(tensor.name), len(shape), ",".join(str(d) for d in shape), kind, shard))
lines += ["};", "#endif"]
args.header.write_text("\n".join(lines) + "\n")
canonical = dict(metadata={k: v.contents() for k, v in metadata.items()}, tensors=tensors)
args.manifest.write_text(json.dumps(dict(
    **canonical, shards=shards,
    directory_sha256=hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
    header_sha256=hashlib.sha256(args.header.read_bytes()).hexdigest()), indent=2) + "\n")
print("Extracted", len(metadata), "metadata entries and", len(tensors), "tensor descriptions; zero weight payload values")
