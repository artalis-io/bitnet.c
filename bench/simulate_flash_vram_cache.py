#!/usr/bin/env python3
"""Replay frozen Flash routing through a proposed bounded raw-expert cache.

This is a placement simulation, not implemented GPU residency or a speedup
measurement. Compare largest-payload slots with compact size-class pools.
Pool quotas allocate equal expert counts per layer, then spend the remainder
in layer order. Cache LRU is global for fixed slots, separate per size pool.
Prefill either borrows without admission or admits the last token's experts;
decode promotes/adopts LRU. Tail admission models a bank-to-cache device copy,
not host upload or a latency prediction; direct-to-cache upload is not modeled.
All layer hits are pinned before admitting misses; pins expire after the layer.
Actual GPU execution still needs event-protected leases and cached-weight kernels.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import struct


def load_trace(reference, inventory):
    manifest = json.loads((reference / 'manifest.json').read_bytes())
    tensors = {x['name']: x for x in json.loads(inventory.read_bytes())['tensors']}
    sizes = []
    for layer in range(48):
        projections = [tensors[f'blk.{layer}.ffn_{role}_exps.weight'] for role in ['gate', 'up', 'down']]
        assert all(p['type'] == manifest['weights'][p['name']]['type'] and p['shape'] == manifest['weights'][p['name']]['shape'] for p in projections)
        assert all(p['shape'][-1] == 512 and p['bytes'] % 512 == 0 for p in projections)
        sizes.append(sum(p['bytes'] // 512 for p in projections))
    phases = []
    for phase in manifest['phases']:
        blocks = []
        for layer in range(48):
            name = f"phase{phase['phase']}-layer{layer}-ffn-ids.bin"
            data = (reference / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
            assert len(data) == len(phase['tokens']) * 10 * 4
            ids = [v[0] for v in struct.iter_unpack('<i', data)]
            assert all(0 <= v < 512 for v in ids)
            for i in range(0, len(ids), 10):
                assert len(set(ids[i:i + 10])) == 10
            blocks.append({'experts': [(layer, expert, sizes[layer]) for expert in sorted(set(ids))],
                           'tail_experts': set(ids[-10:])})
        phases.append({'phase': phase['phase'], 'prefill': phase['prefill'], 'blocks': blocks})
    return phases, sizes, hashlib.sha256((reference / 'manifest.json').read_bytes()).hexdigest()


def pool_capacities(sizes, budget, placement):
    if placement == 'fixed':
        return {0: budget // max(sizes)}
    per_layer_set = sum(sizes)
    count = min(512, budget // per_layer_set)
    capacities = {size: sizes.count(size) * count for size in set(sizes)}
    remaining = budget - count * per_layer_set
    for size in sizes:
        if count < 512 and remaining >= size:
            capacities[size] += 1
            remaining -= size
    assert all(capacities[size] <= sizes.count(size) * 512 for size in capacities)
    assert sum(size * n for size, n in capacities.items()) <= budget
    return capacities


def simulate(phases, sizes, budget, repeats, prefill_policy, placement):
    capacities = pool_capacities(sizes, budget, placement)
    caches = {key: OrderedDict() for key in capacities}
    reserved = sum((key or max(sizes)) * n for key, n in capacities.items())
    output = []
    for repeat in range(repeats):
        for phase in phases:
            before = {key: tuple(cache.items()) for key, cache in caches.items()}
            counts = dict(hits=0, misses=0, hit_bytes=0, upload_bytes=0, evictions=0, bypassed_admissions=0, admitted_device_copy_bytes=0)
            for block in phase['blocks']:
                pool = 0 if placement == 'fixed' else block['experts'][0][2]
                cache = caches[pool]
                capacity = capacities[pool]
                pinned = {(layer, expert) for layer, expert, _ in block['experts'] if (layer, expert) in cache}
                for layer, expert, size in block['experts']:
                    admit = not phase['prefill'] or (prefill_policy == 'tail' and expert in block['tail_experts'])
                    key = (layer, expert)
                    if key in pinned:
                        counts['hits'] += 1
                        counts['hit_bytes'] += size
                        if admit:
                            cache.move_to_end(key)
                        continue
                    counts['misses'] += 1
                    counts['upload_bytes'] += size
                    if not admit:
                        continue
                    if len(cache) >= capacity:
                        victim = next((key for key in cache if key not in pinned), None)
                        if victim is None:
                            counts['bypassed_admissions'] += 1
                            continue
                        del cache[victim]
                        counts['evictions'] += 1
                    cache[key] = size
                    if phase['prefill']:
                        counts['admitted_device_copy_bytes'] += size
                    pinned.add(key)
                assert len(cache) <= capacity
            if phase['prefill'] and prefill_policy == 'borrow':
                assert {key: tuple(cache.items()) for key, cache in caches.items()} == before
            baseline = sum(size for block in phase['blocks'] for _, _, size in block['experts'])
            assert counts['hit_bytes'] + counts['upload_bytes'] == baseline
            assert sum(sum(cache.values()) for cache in caches.values()) <= reserved <= budget
            output.append({'repeat': repeat, 'phase': phase['phase'], 'prefill': phase['prefill'],
                           **counts, 'baseline_upload_bytes': baseline, 'resident_slots': sum(map(len, caches.values())),
                           'resident_payload_bytes': sum(sum(cache.values()) for cache in caches.values())})
    return {'placement': placement, 'prefill_policy': prefill_policy, 'budget_bytes': budget, 'slots': sum(capacities.values()),
            'pool_capacities': capacities, 'reserved_slot_payload_bytes': reserved,
            'phases': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--budget-mib', nargs='+', type=int, default=[0, 64, 256, 1024, 1536, 2048, 4096, 8192])
    parser.add_argument('--placement', nargs='+', choices=['fixed', 'pools'], default=['fixed', 'pools'])
    parser.add_argument('--prefill-policy', nargs='+', choices=['borrow', 'tail'], default=['borrow', 'tail'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats > 0 and all(x >= 0 for x in args.budget_mib)
    phases, sizes, digest = load_trace(args.reference, args.inventory)
    result = {'scope': __doc__, 'reference_manifest_sha256': digest, 'inventory_sha256': hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
              'slot_bytes': max(sizes), 'full_expert_raw_bytes': sum(sizes) * 512,
              'full_fixed_slot_bytes': max(sizes) * 48 * 512,
              'cases': [simulate(phases, sizes, x * 1024**2, args.repeats, policy, placement)
                        for placement in args.placement for policy in args.prefill_policy for x in args.budget_mib]}
    args.output.write_text(json.dumps(result, indent=2))
    print(f'{len(phases)} reference phases; maximum slot payload {max(sizes)} bytes; {len(result["cases"])} simulated budgets')


if __name__ == '__main__':
    main()
