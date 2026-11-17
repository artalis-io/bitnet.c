#!/usr/bin/env python3
"""CPU-only wire-format checks; runtime capability admission has its own tests."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy = dict(arithmetic=3, ffn_arithmetic=2,
                  input_norm_reductions=[2] * 65,
                  post_norm_reductions=[2] * 64)
    policy['input_norm_reductions'][64] = 6
    policy['post_norm_reductions'][60] = 5
    with tempfile.TemporaryDirectory() as directory:
        d = Path(directory)
        source = d / 'probe.c'
        source.write_text('''#include "bench/bf16_policy_io.h"
#include <assert.h>
int main(int argc,char **argv) {
    assert(argc==2);
    BnTransformerBF16Policy p={0},before;
    memset(&p,0x5a,sizeof(p)); before=p;
    if(bench_policy_read(argv[1],&p)) {
        assert(!memcmp(&p,&before,sizeof(p))); return 2;
    }
    return bench_policy_write(stdout,&p)?1:0;
}
''')
        binary = d / 'probe'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-Wall',
                        '-Wextra', '-Werror', '-I'+str(ROOT),
                        '-I'+str(ROOT / 'include'), str(source), '-o', str(binary)], check=True)
        path = d / 'policy.json'

        def run(text, valid):
            path.write_bytes(text.encode() if isinstance(text, str) else text)
            result = subprocess.run([str(binary), str(path)], capture_output=True)
            assert result.returncode == (0 if valid else 2), (text, result)
            if valid:
                expected = json.loads(text)
                for key in ['attention_split_policy', 'gdn_prefill_selection']:
                    if expected.get(key) == 0:
                        del expected[key]
                assert json.loads(result.stdout) == expected
            else:
                assert not result.stdout

        valid = json.dumps(policy)
        run(valid, True)
        run(json.dumps(dict(reversed(list(policy.items()))), indent=2), True)
        defaults = dict(arithmetic=0, ffn_arithmetic=0,
                        input_norm_reductions=[0]*65, post_norm_reductions=[0]*64)
        run(json.dumps(defaults), True)
        run(json.dumps(dict(policy, arithmetic=5)), True)
        run(json.dumps(dict(policy, attention_split_policy=0)), True)
        run(json.dumps(dict(policy, attention_split_policy=1)), True)
        batch_policy = dict(policy, attention_split_policy=1)
        run(json.dumps(dict(reversed(list(batch_policy.items())))), True)
        for value in [0, 1]:
            selected = dict(batch_policy, arithmetic=5, gdn_prefill_selection=value)
            run(json.dumps(selected), True)
            run(json.dumps(dict(reversed(list(selected.items())))), True)
        run(json.dumps(selected)[:-1]+', "gdn_prefill_selection":1}', False)
        for missing in policy:
            run(json.dumps({k: v for k, v in batch_policy.items() if k != missing}), False)
        repeated = json.dumps(batch_policy)[:-1] + ', "attention_split_policy":1}'
        run(repeated, False)
        for key, limit in [('arithmetic', 5), ('ffn_arithmetic', 2), ('attention_split_policy', 1), ('gdn_prefill_selection', 1)]:
            for value in [-1, limit+1, 10, 1.0, True, None, '2']:
                run(json.dumps(dict(policy, **{key: value})), False)
        for key, count in [('input_norm_reductions', 65), ('post_norm_reductions', 64)]:
            for values in [[], [2]*(count-1), [2]*(count+1), [7]+[2]*(count-1),
                           [-1]+[2]*(count-1), [False]+[2]*(count-1)]:
                run(json.dumps(dict(policy, **{key: values})), False)
        for key in policy:
            run(json.dumps({k: v for k, v in policy.items() if k != key}), False)
            run(valid[:-1]+', '+json.dumps(key)+':'+json.dumps(policy[key])+'}', False)
        for text in ['', '{}', valid+'{}', valid+',', valid[:-1]+',}',
                     valid.replace('"arithmetic": 3', '"arithmetic": 03'),
                     valid.replace('"arithmetic"', '"unknown"'),
                     valid.replace('"arithmetic"', '"arith\\u006detic"'),
                     valid+'\0', ' '*16384+valid]:
            run(text, False)
        for end in range(len(valid)):
            run(valid[:end], False)
        print('BF16 benchmark policy protocol PASSED')


if __name__ == '__main__':
    main()
