#!/usr/bin/env python3
"""Protocol tests: incomplete or misaligned independent oracles must fail."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT = Path(__file__).with_name("compare_prefill_reference.py")


class ReferenceHarnessTest(unittest.TestCase):
    def check_case(self, mode, backend="cpu", pin_reference=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").write_bytes(b"synthetic model")
            (root / "corpus").write_text("synthetic corpus")
            marker = root / "runtime-ran"
            inputs_marker = root / "inputs-ran"
            runtime = root / "runtime"
            runtime.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(f"""
                import os, sys
                from pathlib import Path
                if '--print-input-ids' in sys.argv:
                    Path({str(inputs_marker)!r}).touch()
                    n = int(sys.argv[3]) - ({mode!r} == 'bad-input-count')
                    print('prompt_token_ids=' + ','.join(['1']*n))
                    print('resume_token_ids=' + ','.join(['2']*64))
                else:
                    Path({str(marker)!r}).touch()
                    if '--cuda' in sys.argv: assert os.environ['BN_CUDA_DEVICE'] == '0'
                    ref = Path(sys.argv[5]).read_text().split()
                    assert ref == list(map(str, list(range(8)) + list(range(10,18))))
                    if {mode!r} == 'runtime-error': sys.exit(7)
                    if {mode!r} != 'missing-runtime-marker':
                        print('Independent prefill reference PASSED: 16/16 token IDs match')
                """))
            runtime.chmod(0o755)
            probe = root / "probe"
            probe.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(f"""
                import sys
                if '--reference-info' in sys.argv:
                    if {mode!r} != 'missing-revision':
                        commit = '012345678' if {mode!r} == 'wrong-revision' else 'df18bee11'
                        print('ggml_commit=' + commit)
                        if {mode!r} == 'duplicate-revision': print('ggml_commit=' + commit)
                    sys.exit(0)
                n = len(sys.argv[sys.argv.index('--prompt-token-ids')+1].split(','))
                chunk = sys.argv[sys.argv.index('--prefill-chunk')+1]
                for token in range(8): print('llama_token_id=' + str(token))
                pos = n + (7 if {mode!r} == 'bad-resume-position' else 8)
                print('llama_resume_prefill pos=%d tokens=64 chunk=%s' % (pos,chunk))
                count = 7 if {mode!r} == 'truncated-oracle' else 8
                for token in range(10,10+count): print('llama_resumed_token_id=' + str(token))
                """))
            probe.chmod(0o755)
            out = root / "out"
            command = [sys.executable, str(SCRIPT),
                "--model", str(root / "model"), "--corpus", str(root / "corpus"),
                "--probe", str(probe), "--runtime", str(runtime),
                "--backend", backend, "--tokens", "33", "--chunk", "9",
                "--output-dir", str(out)]
            if pin_reference:
                command += ["--reference-commit", "df18bee11e96557b07aa7b12db0976f3c7ac9165"]
            result = subprocess.run(command, capture_output=True, text=True)
            report = json.loads((out / "result.json").read_text())
            if mode == "success":
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(report["passed"])
                self.assertEqual(report["matched_tokens"], 16)
                self.assertEqual(report["kv"], "f16" if backend == "cuda" else "f32")
                self.assertEqual("--kv-f16" in report["oracle_command"], backend == "cuda")
                self.assertEqual((out / "corpus.txt").read_text(), "synthetic corpus")
                if pin_reference:
                    self.assertEqual(report["reference_commit"], "df18bee11")
            else:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(report["passed"])
                self.assertEqual(report["status"], "failed")
                if mode in ("truncated-oracle", "bad-resume-position", "bad-input-count"):
                    self.assertFalse(marker.exists())
                if mode in ("wrong-revision", "missing-revision", "duplicate-revision"):
                    self.assertFalse(marker.exists())
                    self.assertFalse(inputs_marker.exists())

    def test_cpu_success(self): self.check_case("success")
    def test_cuda_protocol(self): self.check_case("success", "cuda")
    def test_truncated_oracle(self): self.check_case("truncated-oracle")
    def test_resume_position(self): self.check_case("bad-resume-position")
    def test_input_count(self): self.check_case("bad-input-count")
    def test_runtime_error(self): self.check_case("runtime-error")
    def test_runtime_marker(self): self.check_case("missing-runtime-marker")
    def test_pinned_reference(self): self.check_case("success", pin_reference=True)
    def test_wrong_revision(self): self.check_case("wrong-revision", pin_reference=True)
    def test_missing_revision(self): self.check_case("missing-revision", pin_reference=True)
    def test_duplicate_revision(self): self.check_case("duplicate-revision", pin_reference=True)


if __name__ == "__main__":
    unittest.main()
