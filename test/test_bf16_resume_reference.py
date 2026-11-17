#!/usr/bin/env python3
"""Protocol regressions for complete, aligned independent resume comparisons."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('compare_bf16_resume_reference.py')
spec = importlib.util.spec_from_file_location('resume_reference',SCRIPT)
comparison = importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class ResumeReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=self.root/'model';self.model.write_bytes(b'independent-checkpoint')
        self.ref={'schema':1,'scope':'synthetic protocol only','model_sha256':hashlib.sha256(self.model.read_bytes()).hexdigest(),
                  'architecture':{'n_layers':2,'layers':['gdn','attention'],'state':[1,2,2],'history':[2,2],'kv':[1,2],'vocab':3,'hidden':2},
                  'inputs':{'initial':[1,2],'resume':[2,1]},'tokens':[[1,2],[2,1]],'positions':{'before':5,'after':6},'payloads':{}}
        self.records=[{'kind':'inputs',**self.ref['inputs']},*({'kind':'phase','phase':i,'tokens':tokens}for i,tokens in enumerate(self.ref['tokens']))]
        self.log=self.root/'log';self.write_log()
        for index,(name,(dtype,shape)) in enumerate(comparison.required_payloads(self.ref).items()):
            import math
            data=bytes((i+index)%251 for i in range(math.prod(shape)*(4 if dtype=='float32'else 2)))
            (self.root/name).write_bytes(data);self.ref['payloads'][name]={'dtype':dtype,'shape':shape,'sha256':hashlib.sha256(data).hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def write_log(self):
        self.log.write_text('\n'.join(json.dumps(r)for r in self.records)+'\n')

    def validate(self):
        return comparison.validate(self.ref,self.root,self.log,self.model)

    def test_complete(self):
        result=self.validate();self.assertTrue(result['passed']);self.assertEqual(result['payloads'],11)

    def test_missing_layer(self):
        self.ref['architecture']['layers'].pop()
        with self.assertRaises(ValueError):self.validate()

    def test_missing_reference_payload(self):
        self.ref['payloads'].pop('before-l0-state.bin')
        with self.assertRaises(ValueError):self.validate()

    def test_extra_reference_payload(self):
        self.ref['payloads']['extra.bin']=copy.deepcopy(next(iter(self.ref['payloads'].values())))
        with self.assertRaises(ValueError):self.validate()

    def test_missing_native_payload(self):
        (self.root/'after-l1-value.bin').unlink()
        with self.assertRaises(OSError):self.validate()

    def test_corrupted_state(self):
        path=self.root/'before-l0-state.bin';data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data)
        with self.assertRaises(ValueError):self.validate()

    def test_truncated_history(self):
        path=self.root/'after-l0-history.bin';path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):self.validate()

    def test_bad_tensor_shape(self):
        self.ref['payloads']['before-l0-state.bin']['shape']=[4]
        with self.assertRaises(ValueError):self.validate()

    def test_bad_checkpoint(self):
        self.model.write_bytes(b'another-checkpoint')
        with self.assertRaises(ValueError):self.validate()

    def test_changed_native_inputs(self):
        self.records[0]=copy.deepcopy(self.records[0]);self.records[0]['initial'][0]=0;self.write_log()
        with self.assertRaises(ValueError):self.validate()

    def test_missing_inputs(self):
        self.records.pop(0);self.write_log()
        with self.assertRaises(ValueError):self.validate()

    def test_misaligned_resume(self):
        self.ref['positions']['before']-=1
        with self.assertRaises(ValueError):self.validate()

    def test_phase_order(self):
        self.records[1],self.records[2]=self.records[2],self.records[1];self.write_log()
        with self.assertRaises(ValueError):self.validate()

    def test_duplicate_phase(self):
        self.records.append(self.records[-1]);self.write_log()
        with self.assertRaises(ValueError):self.validate()

    def test_changed_native_outputs(self):
        self.records[2]=copy.deepcopy(self.records[2]);self.records[2]['tokens'][0]=0;self.write_log()
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':
    unittest.main()
