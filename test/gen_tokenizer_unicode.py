#!/usr/bin/env python3
"""Generate tokenizer character classes from the local llama.cpp reference.

The output contains no runtime dependency on llama.cpp or the host locale.
"""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_cpp_root", type=Path)
    parser.add_argument("output_header", type=Path)
    args = parser.parse_args()
    reference = args.llama_cpp_root
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=reference, text=True
    ).strip()
    source = (reference / "src/unicode-data.cpp").read_text()
    category_data = source.split("unicode_ranges_flags =", 1)[1].split("};", 1)[0]
    whitespace_data = source.split("unicode_set_whitespace =", 1)[1].split("};", 1)[0]
    ranges = [
        (int(start, 16), int(flags, 16) & 0x16)
        for start, flags in re.findall(
            r"\{(0x[0-9A-Fa-f]+), (0x[0-9A-Fa-f]+)\}", category_data
        )
    ]
    whitespace = {int(value, 16) for value in re.findall(r"0x[0-9A-Fa-f]+", whitespace_data)}
    if not ranges or ranges[0][0] != 0 or ranges[-1][0] != 0x110000 or not whitespace:
        raise ValueError("Unrecognized reference Unicode table")
    # Include both edges of whitespace overrides, then coalesce equal flags.
    points = sorted({start for start, _ in ranges} | whitespace | {cp + 1 for cp in whitespace})
    rows = []
    index = 0
    for cp in points:
        while index + 1 < len(ranges) and ranges[index + 1][0] <= cp:
            index += 1
        flags = ranges[index][1] | (0x100 if cp in whitespace else 0)
        if not rows or rows[-1][1] != flags:
            rows.append((cp, flags))
    license_text = (reference / "LICENSE").read_text()
    output = (
        "/* Unicode category data derived from llama.cpp unicode-data.cpp,\n"
        f" * commit {revision}.\n"
        " * Retains only letter, mark, number and whitespace flags.\n"
        " * Regenerate with test/gen_tokenizer_unicode.py LLAMA_CPP_ROOT OUTPUT.\n *\n"
        + "".join((" * " + line).rstrip() + "\n" for line in license_text.splitlines())
        + " */\n#ifndef BN_TOKENIZER_UNICODE_H\n#define BN_TOKENIZER_UNICODE_H\n\n"
        "static const struct { uint32_t start; unsigned flags; } tokenizer_unicode_ranges[] = {\n"
        + "".join(f"    {{0x{cp:06x}, 0x{flags:03x}}},\n" for cp, flags in rows)
        + "};\n\n#endif /* BN_TOKENIZER_UNICODE_H */\n"
    )
    args.output_header.write_text(output)
    print(f"Wrote {len(rows)} ranges to {args.output_header}")


if __name__ == "__main__":
    main()
