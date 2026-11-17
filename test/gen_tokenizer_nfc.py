#!/usr/bin/env python3
"""Generate pinned Unicode 9 NFC data and independent conformance fixtures.

Usage: gen_tokenizer_nfc.py UCD_DIRECTORY
Inputs are the unmodified official Unicode data, not candidate normalizer output.
No Unicode library or network access is needed at runtime or during ordinary tests.
"""
import hashlib
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1])
    expected = {
        'UnicodeData-9.0.0.txt': '68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f',
        'DerivedNormalizationProps-9.0.0.txt': '02d8e12cea7c61a9f3cc5fbf8eaceadf8da23fe83d60c64cf647088830b810cc',
        'LICENSE-UNICODE.txt': 'e7a93b009565cfce55919a381437ac4db883e9da2126fa28b91d12732bc53d96',
        'NormalizationTest-9.0.0.txt': '2d48d848656b3cf889df59980ab13551988950c4ca8c5190a17c691a17f8b9bb',
    }
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Unrecognized Unicode input: {name}")
    records = {}
    for line in (root / "UnicodeData-9.0.0.txt").read_text().splitlines():
        fields = line.split(";")
        records[int(fields[0], 16)] = fields
    excluded = set()
    for line in (root / "DerivedNormalizationProps-9.0.0.txt").read_text().splitlines():
        fields = line.split("#")[0].strip().split(";")
        if len(fields) < 2 or fields[1].strip() != "Full_Composition_Exclusion":
            continue
        ends = [int(x, 16) for x in fields[0].strip().split("..")]
        excluded.update(range(ends[0], ends[-1] + 1))
    direct = {cp: [int(x, 16) for x in f[5].split()]
              for cp, f in records.items() if f[5] and not f[5].startswith("<")}

    def decompose(cp):
        return [d for x in direct[cp] for d in decompose(x)] if cp in direct else [cp]

    rows = [(cp, decompose(cp)) for cp in sorted(direct)]
    assert max(len(x) for _, x in rows) == 4
    compositions = sorted((ds[0], ds[1], cp) for cp, ds in direct.items()
                          if len(ds) == 2 and cp not in excluded)
    license_text = (root / "LICENSE-UNICODE.txt").read_text()
    notice = ("/* Generated from official Unicode 9.0.0 UCD; matches the pinned\n"
              " * tokenizers unicode-normalization-alignments 0.1.12 data version.\n"
              " * Regenerate with test/gen_tokenizer_nfc.py UCD_DIRECTORY.\n"
              " * Source: https://www.unicode.org/Public/9.0.0/ucd/\n"
              + "".join(f" * {n} SHA256 {h}\n" for n, h in expected.items())
              + " *\n" + "".join(" * " + x + "\n" for x in license_text.splitlines())
              + " */\n")
    data = notice + "#ifndef BN_TOKENIZER_NFC_DATA_H\n#define BN_TOKENIZER_NFC_DATA_H\n\n"
    data += "static const struct { uint32_t cp; unsigned char ccc; } tokenizer_nfc_classes[] = {\n"
    data += "".join(f"    {{0x{cp:x}, {f[3]}}},\n" for cp, f in sorted(records.items()) if int(f[3]))
    data += "};\n\nstatic const struct { uint32_t cp; unsigned char n; uint32_t values[4]; } tokenizer_nfc_decompositions[] = {\n"
    data += "".join(f"    {{0x{cp:x}, {len(ds)}, {{{', '.join(hex(d) for d in ds)}}}}},\n" for cp, ds in rows)
    data += "};\n\nstatic const struct { uint32_t a, b, composed; } tokenizer_nfc_compositions[] = {\n"
    data += "".join(f"    {{0x{a:x}, 0x{b:x}, 0x{cp:x}}},\n" for a, b, cp in compositions)
    data += "};\n\n#endif /* BN_TOKENIZER_NFC_DATA_H */\n"
    Path("src/tokenizer_nfc_data.h").write_text(data)
    fixtures = notice + "#ifndef BN_TOKENIZER_NFC_CASES_H\n#define BN_TOKENIZER_NFC_CASES_H\n\n"
    fixtures += "static const char *const tokenizer_nfc_cases[][5] = {\n"
    count = 0
    for line in (root / "NormalizationTest-9.0.0.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("@"):
            continue
        cols = ["".join(chr(int(x, 16)) for x in c.split()).encode("utf-8")
                for c in line.split(";")[:5]]
        strings = ['"' + "".join(f"\\x{x:02x}" for x in c) + '"' for c in cols]
        fixtures += "    {" + ", ".join(strings) + "},\n"
        count += 1
    fixtures += "};\n\n#endif /* BN_TOKENIZER_NFC_CASES_H */\n"
    Path("test/tokenizer_nfc_cases.h").write_text(fixtures)
    print(f"Generated {len(rows)} decompositions, {len(compositions)} compositions, {count * 5} conformance cases")


if __name__ == "__main__":
    main()
