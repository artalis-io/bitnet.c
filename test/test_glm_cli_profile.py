"""CLI profile admission and unsupported combinations; synthetic GGUF metadata only."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = str(Path(sys.argv[1]).resolve())

def string(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def stub(path, architecture):
    data = b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
    data += string('general.architecture') + struct.pack('<I', 8) + string(architecture)
    path.write_bytes(data + bytes((-len(data)) % 32))

def reject(path, args, expected):
    env = dict(os.environ)
    env.pop('BN_CPU_PREPARE_ALL_EXPERTS', None)
    result = subprocess.run([binary, str(path), *args], env=env, capture_output=True, text=True)
    assert result.returncode != 0, (args, result.stdout, result.stderr)
    assert expected in result.stderr, (args, expected, result.stderr)
    return result.stderr

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp)
    glm = p / 'glm.gguf'
    other = p / 'other.gguf'
    stub(glm, 'glm5next')
    stub(other, 'qwen3next')
    reject(glm, ['--cuda-bf16-profile', 'glm-recurrent'], 'requires --cuda-bf16')
    base = ['--cuda-bf16', '--cuda-bf16-profile', 'glm-recurrent']
    for flags in [['--chat'], ['--prefault-moe'], ['--gpu-cache-mb', '32'], ['--gpu-preload-experts', '--gpu-cache-mb', '32'], ['--cache-mb', '-1']]:
        reject(glm, base + flags, 'glm-recurrent requires single-shot prompts')
    # Admission occurs before topology payload binding or GPU construction.
    reject(glm, [], 'GLM requires --cuda-bf16 --cuda-bf16-profile glm-recurrent')
    error = reject(other, base, '')
    if '--cuda-bf16 requires BN_ENABLE_CUDA=1' not in error and '--cuda-bf16 requires BN_ENABLE_MKL_VML=1' not in error:
        assert 'glm-recurrent requires a supported GLM topology checkpoint' in error, error
        reject(glm, ['--cuda-bf16'], 'GLM requires --cuda-bf16 --cuda-bf16-profile glm-recurrent')
        reject(glm, base + ['--cuda-bf16-profile', 'eager'], 'GLM requires --cuda-bf16 --cuda-bf16-profile glm-recurrent')
print('GLM CLI profile admission PASSED (synthetic metadata, no GPU execution)')
