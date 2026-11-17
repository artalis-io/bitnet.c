"""Check legacy wire compatibility and clock workloads without expected outputs."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
source = root / 'test/bf16_cuda_staggered12_reference.json'
r = json.loads(source.read_text())
script = root / 'bench/prepare_bf16_arrivals.py'
with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary) / 'workload.bin'
    base = [sys.executable, str(script), str(source), str(output)]
    subprocess.run(base, check=True)
    # Frozen BGA1 from the independently qualified twelve-request workload.
    assert hashlib.sha256(output.read_bytes()).hexdigest() == '849f3c3370444307c8357afd186b6a9ccea87dfc901d77e613b8b8e369554a31'
    legacy = output.read_bytes()
    subprocess.run(base + ['--mixed-budget', '8'], check=True)
    controlled = output.read_bytes()
    assert controlled[:4] == b'BGA4' and struct.unpack_from('<i', controlled, 36)[0] == 8
    assert controlled[4:36] == legacy[4:36] and controlled[40:] == legacy[36:]
    arrivals = [0] * 4 + [1000] * 4 + [4000] * 4
    subprocess.run(base + ['--arrivals-ms', *map(str, arrivals)], check=True)
    data = output.read_bytes()
    assert data[:4] == b'BGA2'
    fields = struct.unpack_from('<8i', data, 4)
    assert fields == (12, 64, 8, 4, 64, 256, 8192, 4096)
    offset = 36
    for deadline, stream in zip(arrivals, r['streams']):
        arrival, count = struct.unpack_from('<2i', data, offset)
        offset += 8
        assert arrival == deadline and count == len(stream['input_ids'])
        assert list(struct.unpack_from(f'<{count}i', data, offset)) == stream['input_ids']
        offset += count * 4
    # BGA2 has no teacher-forced continuation or output-expectation payload.
    assert offset == len(data) == 4340
    subprocess.run(base + ['--arrivals-ms', *map(str, arrivals), '--mixed-budget', '8'], check=True)
    mixed = output.read_bytes()
    assert mixed[:4] == b'BGA3' and struct.unpack_from('<i', mixed, 36)[0] == 8
    assert mixed[4:36] == data[4:36] and mixed[40:] == data[36:]
    specification = dict(schema=1, streams=[dict(input_ids=s['input_ids']) for s in r['streams']],
                         outputs_each=64, max_live=8, capacity=8192, token_budget=256,
                         engine_request=r['engine_request'], offered_arrival_ms=arrivals, mixed_token_budget=8)
    spec = Path(temporary) / 'spec.json'
    spec.write_text(json.dumps(specification))
    spec_command = [sys.executable, str(script), str(spec), str(output)]
    subprocess.run(spec_command, check=True)
    assert output.read_bytes() == mixed
    subprocess.run(spec_command + ['--mixed-budget', '0'], check=True)
    assert output.read_bytes() == data
    specification['streams'][0]['tokens'] = [0] * 64
    spec.write_text(json.dumps(specification))
    assert subprocess.run(spec_command, capture_output=True).returncode != 0
    for options in [['--mixed-budget', '7'], ['--arrivals-ms', *map(str, arrivals), '--mixed-budget', '7'],
                    ['--arrivals-ms', *map(str, arrivals), '--mixed-budget', '257']]:
        assert subprocess.run(base + options, capture_output=True).returncode != 0
    for invalid in [[0], [-1] * 12, [1, 0] + [1000] * 10, [3600001] * 12]:
        result = subprocess.run(base + ['--arrivals-ms', *map(str, invalid)], capture_output=True)
        assert result.returncode != 0
print('BF16 arrival workload clock protocol and BGA1 compatibility PASSED')
