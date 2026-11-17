"""Synthetic lifetime/byte checks detect hidden eviction and bad charges."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from verify_bf16_capacity_pool import validate_capacity_pool


class CapacityPoolTests(unittest.TestCase):
    def records(self):
        unit = 65536
        rows = [dict(kind='kv_admission', mode='request_capacity'), dict(kind='summary', peak_live=2)]
        for i, (required, actual) in enumerate([(1, 64), (65, 128), (193, 256), (1, 64)]):
            rows.append(dict(kind='kv_lease', id=i, required_capacity=required, lease_capacity=actual))
        states = [
            (0, 'acquire', [64, 0], [1, 0], 1, 0, 0),
            (1, 'acquire', [64, 128], [1, 1], 2, 0, 0),
            (0, 'release', [64, 128], [0, 1], 2, 0, 0),
            (1, 'release', [64, 128], [0, 0], 2, 0, 0),
            (2, 'acquire', [64, 256], [0, 1], 3, 0, 1),
            (2, 'release', [64, 256], [0, 0], 3, 0, 1),
            (3, 'acquire', [64, 256], [1, 0], 3, 1, 1),
            (3, 'release', [64, 256], [0, 0], 3, 1, 1)]
        peak = 0
        for index, (request, action, caps, used, created, reused, evicted) in enumerate(states):
            reserved = sum(caps) * unit;peak = max(peak, reserved)
            rows.append(dict(kind='kv_state', event=index, request=request, action=action,
                             capacity=caps + [0] * 14, leased=used + [0] * 14,
                             created=created, reused=reused, evicted=evicted, discarded=0,
                             reserved_kv_bytes=reserved, peak_kv_bytes=peak))
        rows.append(dict(kind='kv_pool', scope='kv_payload_only', kv_bytes_per_slot=512*unit,
                         budget_bytes=512*unit, created=3, reused=1, discarded=0, leased_slots=0,
                         allocated_slots=2, reserved_kv_bytes=320*unit, peak_kv_bytes=320*unit))
        return rows

    def verify(self, rows):
        return validate_capacity_pool(rows, [1, 65, 193, 1], 1, 2, 512, 64, 512*65536)

    def test_reclaim_reuse_and_retained_bytes(self):
        result = self.verify(self.records())
        self.assertEqual(result['created'], 3)
        self.assertEqual(result['evicted'], 1)
        self.assertEqual(result['retained_kv_bytes'], 320*65536)

    def test_reject_corrupted_lifetimes_and_totals(self):
        for change in ['active_resize', 'lost_release', 'fake_eviction', 'bad_charge', 'bad_peak',
                       'undersized', 'missing_event', 'leak', 'bad_required', 'unaccounted_slot']:
            with self.subTest(change=change):
                rows = self.records();states = [r for r in rows if r['kind']=='kv_state']
                if change=='active_resize':states[1]['capacity'][0] = 128
                elif change=='lost_release':states[2]['leased'][0] = 1
                elif change=='fake_eviction':states[4]['evicted'] = 0
                elif change=='bad_charge':states[4]['reserved_kv_bytes'] -= 65536
                elif change=='bad_peak':states[5]['peak_kv_bytes'] = 0
                elif change=='undersized':rows[4]['lease_capacity'] = 128
                elif change=='missing_event':rows.remove(states[2])
                elif change=='leak':states[-1]['leased'][0] = 1
                elif change=='bad_required':rows[2]['required_capacity'] = 2
                else:states[0]['capacity'][2] = 64
                with self.assertRaises(AssertionError):self.verify(rows)

    def test_idle_inventory_can_exceed_peak_live(self):
        rows = self.records();rows[1]['peak_live'] = 1
        states = [r for r in rows if r['kind']=='kv_state']
        states[1].update(request=0, action='release', capacity=[64]+[0]*15,
                         leased=[0]*16, created=1, reserved_kv_bytes=64*65536, peak_kv_bytes=64*65536)
        states[2].update(request=1, action='acquire', leased=[0,1]+[0]*14)
        result = self.verify(rows)
        self.assertEqual(result['retained_kv_bytes'], 320*65536)

    def test_resize_reuses_idle_owner(self):
        rows = self.records();states = [r for r in rows if r['kind']=='kv_state']
        # The same idle128 owner grows to256, rather than being destroyed.
        for state in states[4:]:
            state['created'] = 2;state['reused'] += 1;state['evicted'] = 0;state['resized'] = 1
        pool = rows[-1];pool['created'] = 2;pool['reused'] = 2
        result = self.verify(rows)
        self.assertEqual(result['resized'], 1)
        self.assertEqual(result['evicted'], 0)
        states[4]['resized'] = 0
        with self.assertRaises(AssertionError):self.verify(rows)


if __name__ == '__main__':
    unittest.main()
