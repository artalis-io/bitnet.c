"""Check request capacities and every retained-slot transition, not just totals."""


def validate_capacity_pool(records, prompt_lengths, outputs, limit, capacity, max_tokens, budget):
    assert 1 <= len(prompt_lengths) <= 64 and 1 <= outputs <= 256
    assert 1 <= limit <= 16 and 1 <= max_tokens <= capacity <= 8192
    modes = [r for r in records if r['kind'] == 'kv_admission']
    assert modes == [dict(kind='kv_admission', mode='request_capacity')]
    leases = [r for r in records if r['kind'] == 'kv_lease']
    assert [r['id'] for r in leases] == list(range(len(prompt_lengths)))
    for i, r in enumerate(leases):
        assert r['required_capacity'] == prompt_lengths[i] + outputs - 1
        assert 1 <= r['required_capacity'] <= capacity
        needed = max(r['required_capacity'], max_tokens)
        rounded = min(1 << (needed - 1).bit_length(), capacity)
        assert rounded <= r['lease_capacity'] <= capacity
    pool, = [r for r in records if r['kind'] == 'kv_pool']
    unit = 65536  # Qualified dense hybrid: 16 attention layers, 2 BF16[1024] vectors.
    assert pool['scope'] == 'kv_payload_only'
    assert pool['kv_bytes_per_slot'] == capacity * unit and pool['budget_bytes'] == budget
    assert budget >= capacity * unit
    caps, used, active, seen = [0] * 16, [0] * 16, {}, set()
    created = reused = evicted = resized = peak = peak_live = 0
    states = [r for r in records if r['kind'] == 'kv_state']
    assert len(states) == 2 * len(leases)
    for index, r in enumerate(states):
        assert r['event'] == index and 0 <= r['request'] < len(leases)
        new, flags = r['capacity'], r['leased']
        assert len(new) == len(flags) == 16
        assert all(type(n) is int and 0 <= n <= capacity for n in new)
        assert all(type(n) is int and n in (0, 1) for n in flags)
        assert all(not flags[j] or new[j] for j in range(16))
        assert not any(new[limit:]) and not any(flags[limit:])
        request = r['request']
        if r['action'] == 'acquire':
            assert request not in seen and request not in active
            fresh = [j for j in range(16) if flags[j] and not used[j]]
            assert len(fresh) == 1
            slot, = fresh
            assert new[slot] == leases[request]['lease_capacity']
            for j in range(16):
                if used[j]:
                    assert flags[j] and new[j] == caps[j], 'active lease changed'
            new_count, reuse_count = r['created'] - created, r['reused'] - reused
            resize_count = r.get('resized', 0) - resized
            assert resize_count in (0, 1)
            assert (new_count, reuse_count) in ((1, 0), (0, 1))
            if resize_count:
                assert reuse_count == 1 and 0 < caps[slot] < new[slot]
            elif reuse_count:
                assert caps[slot] == new[slot]
            removed = 0
            for j in range(16):
                if j != slot:
                    assert new[j] in (0, caps[j]) and flags[j] == used[j]
                if caps[j] and not used[j] and not (j == slot and resize_count) and (new[j] != caps[j] or j == slot and new_count):
                    removed += 1
            assert r['evicted'] - evicted == removed
            active[request] = slot
            seen.add(request)
        else:
            assert r['action'] == 'release' and request in active
            slot = active.pop(request)
            expected = list(used);expected[slot] = 0
            assert new == caps and flags == expected
            assert r['created'] == created and r['reused'] == reused and r['evicted'] == evicted
            assert r.get('resized', 0) == resized
        assert r['discarded'] == 0
        reserved = sum(new) * unit
        assert reserved == r['reserved_kv_bytes'] <= budget
        peak = max(peak, reserved)
        assert r['peak_kv_bytes'] == peak
        assert sum(flags) == len(active) <= limit
        peak_live = max(peak_live, len(active))
        caps, used = list(new), list(flags)
        created, reused, evicted = r['created'], r['reused'], r['evicted']
        resized = r.get('resized', 0)
    assert seen == set(range(len(leases))) and not active and not any(used)
    assert pool['created'] == created and pool['reused'] == reused
    assert created + reused == len(leases) and pool['discarded'] == pool['leased_slots'] == 0
    assert pool['allocated_slots'] == sum(n != 0 for n in caps)
    assert pool['reserved_kv_bytes'] == sum(caps) * unit and pool['peak_kv_bytes'] == peak
    summary, = [r for r in records if r['kind'] == 'summary']
    assert summary['peak_live'] == peak_live
    return dict(capacity_pool_valid=True, created=created, reused=reused, evicted=evicted, resized=resized,
                retained_kv_bytes=sum(caps) * unit, peak_kv_bytes=peak)
