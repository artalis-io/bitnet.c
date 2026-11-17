#!/usr/bin/env python3
"""Validate retained KV payload accounting for the qualified dense64 layout."""


def validate_pool(config, records):
    pools = [r for r in records if r.get('kind') == 'kv_pool']
    if not config.get('kv_pool_enabled', False):
        assert not pools, 'unexpected pool accounting'
        return None
    assert len(pools) == 1
    p = pools[0]
    assert p['scope'] == 'kv_payload_only'
    # Qualified dense64 anatomy: sixteen attention layers, two BF16 vectors
    # of 1024 elements per token. The actual allocation count is checked by
    # the loaded-model runtime test; SSM/workspace/model memory is excluded.
    per = config['maxseq'] * 16 * 2 * 1024 * 2
    assert p['kv_bytes_per_slot'] == per
    assert p['discarded'] == p['leased_slots'] == 0
    assert p['allocated_slots'] == p['created'] == config['peak_live']
    assert 1 <= p['allocated_slots'] <= config['max_live']
    assert p['created'] + p['reused'] == config['requests']
    assert p['reserved_kv_bytes'] == p['peak_kv_bytes'] == per * p['allocated_slots']
    assert p['peak_kv_bytes'] <= p['budget_bytes']
    return p
