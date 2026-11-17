#!/usr/bin/env python3
"""Summarize a Flash native BF16 Nsight Systems SQLite export.

Windows start at each Q8 embedding's host launch and stop at the next launch;
the final window stops at the second stream fence after its logits readback
(readback fence, then transaction-completion fence). Thus windows include caller
work and between-run restart, and are NOT isolated request latencies.
Preparation is excluded. Allocation counts cover captured runtime API calls;
uncaptured driver/internal allocation is not inferred. Activity unions include kernels, copies and memset.
Bank copy streams are identified across the capture by raw expert uploads
(>=3,072,000 bytes), then weight and descriptor traffic is counted separately.
The explicit expert-cache mode also separates pointer tables and pool fills;
it permits phases with no expert upload. Copy timings are GPU activity durations,
not independently measured critical-path stalls or latency savings.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    end = None
    total = 0
    for start, stop in sorted(intervals):
        if end is None or start >= end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def summarize(database, phases_per_run, prefill_batches, expert_cache=False, bank_stream_ids=None):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    embeddings = list(connection.execute("""
        SELECT r.start host_start, k.gridX, k.blockX
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        WHERE s.value LIKE '%embedding_q8_kernel%' ORDER BY r.start
    """))
    assert embeddings and len(embeddings) % phases_per_run == 0
    final_head = connection.execute("""
        SELECT r.end FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId
        WHERE m.start>? AND m.copyKind=2 AND m.bytes=496640 ORDER BY m.start LIMIT 1
    """, (embeddings[-1]['host_start'],)).fetchone()
    assert final_head is not None, 'final phase must return Flash vocabulary logits'
    fences = list(connection.execute("""
        SELECT r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        WHERE r.start>=? AND s.value LIKE 'cudaStreamSynchronize_%' ORDER BY r.start LIMIT 2
    """, (final_head[0],)))
    assert len(fences) == 2
    finish = fences[1][0]
    # Discover streams over the complete capture: a fully cached phase may
    # have no raw uploads from which to identify its bank streams locally.
    bank_streams = {r[0] for r in connection.execute('''
        SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start>=? AND end<=? AND copyKind=1 AND bytes>=3072000
    ''', (embeddings[0]['host_start'], finish))}
    if bank_stream_ids is not None:
        assert expert_cache, 'explicit streams require cached-bank parsing'
        bank_streams = set(bank_stream_ids)
    assert bank_streams
    rows = []
    for index, embedding in enumerate(embeddings):
        begin = embedding['host_start']
        end = embeddings[index + 1]['host_start'] if index + 1 < len(embeddings) else finish
        args = (begin, end)
        copies = list(connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=?', args))
        kernels = list(connection.execute('SELECT k.*,s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE k.start>=? AND k.end<=?', args))
        memsets = list(connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=? AND end<=?', args))
        apis = list(connection.execute('SELECT r.*,s.value name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE r.start>=? AND r.end<=?', args))
        weights = [r for r in copies if r['copyKind'] == 1 and r['streamId'] in bank_streams and r['bytes'] > 96]
        small = [r for r in copies if r['copyKind'] == 1 and r['streamId'] in bank_streams and r['bytes'] <= 96]
        pointers, descriptors = [], []
        if expert_cache:
            # Each cached bank uploads a pointer table, then its descriptors.
            # Byte sizes alone are ambiguous (e.g. 48 bytes can be either).
            for stream in bank_streams:
                ordered = sorted((r for r in small if r['streamId'] == stream), key=lambda r: r['start'])
                assert len(ordered) % 2 == 0
                for pointer, descriptor in zip(ordered[::2], ordered[1::2]):
                    assert pointer['bytes'] % 8 == 0
                    assert pointer['bytes'] // 8 == descriptor['bytes'] // 12
                    pointers.append(pointer)
                    descriptors.append(descriptor)
        else:
            descriptors = small
            # Uncached banks upload one contiguous raw batch each.
            assert len(weights) == len(descriptors)
        assert all(r['bytes'] % 12 == 0 for r in descriptors)
        fills = [r for r in copies if r['copyKind'] == 8 and r['streamId'] in bank_streams]
        assert expert_cache or not fills
        metadata = [r for r in copies if r['copyKind'] == 2 and r['bytes'] == 2080]
        assert len(metadata) == 48, (index, len(metadata))
        assert all(r['returnValue'] in (0, 600) for r in apis if r['name'].startswith('cudaEventQuery_'))
        allocs = [r for r in apis if any(word in r['name'] for word in ['Alloc', 'Free', 'MemCreate', 'MemRelease', 'MemAddressReserve'])]
        nt, remainder = divmod(embedding['gridX'] * embedding['blockX'], 2560)
        assert not remainder and nt > 0
        if index % phases_per_run >= prefill_batches:
            assert nt == 1, 'decode phase must contain one token'
        per_kernel = {}
        for r in kernels:
            item = per_kernel.setdefault(r['name'], {'calls': 0, 'time_ns': 0})
            item['calls'] += 1
            item['time_ns'] += r['end'] - r['start']
        activity = [(r['start'], r['end']) for r in [*kernels, *copies, *memsets]]
        span = end - begin
        rows.append({
            'run': index // phases_per_run, 'phase': index % phases_per_run,
            'window_start_ns': begin, 'window_end_ns': end,
            'prefill': index % phases_per_run < prefill_batches, 'tokens': nt,
            'window_ms': span / 1e6, 'gpu_activity_union_ms': union_ns(activity) / 1e6,
            'gpu_activity_fraction': union_ns(activity) / span,
            'expert_upload_bytes': sum(r['bytes'] for r in weights),
            'expert_uploads': len(weights), 'expert_items': sum(r['bytes'] // 12 for r in descriptors),
            'expert_copy_gpu_ms': sum(r['end'] - r['start'] for r in weights) / 1e6,
            'descriptor_upload_bytes': sum(r['bytes'] for r in descriptors),
            'pointer_upload_bytes': sum(r['bytes'] for r in pointers),
            'cache_fill_copy_bytes': sum(r['bytes'] for r in fills),
            'cache_fill_copy_gpu_ms': sum(r['end'] - r['start'] for r in fills) / 1e6,
            'h2d_bytes': sum(r['bytes'] for r in copies if r['copyKind'] == 1),
            'd2h_bytes': sum(r['bytes'] for r in copies if r['copyKind'] == 2),
            'routing_metadata_reads': len(metadata), 'routing_metadata_bytes': 48 * 2080,
            'stream_synchronizations': sum(r['name'].startswith('cudaStreamSynchronize_') for r in apis),
            'event_query_calls': sum(r['name'].startswith('cudaEventQuery_') for r in apis),
            'event_not_ready': sum(r['name'].startswith('cudaEventQuery_') and r['returnValue'] != 0 for r in apis),
            'inference_runtime_allocation_calls': len(allocs),
            'allocation_api_names': sorted({r['name'] for r in allocs}),
            'kernel_calls': len(kernels), 'kernel_gpu_ms': sum(r['end'] - r['start'] for r in kernels) / 1e6,
            'top_kernels': sorted(per_kernel.items(), key=lambda x: x[1]['time_ns'], reverse=True)[:12],
        })
    connection.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('--phases-per-run', type=int, required=True)
    parser.add_argument('--prefill-batches', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expert-cache', action='store_true', help='parse cached bank pointer tables and per-expert uploads')
    parser.add_argument('--bank-stream-id', type=int, action='append', help='explicit bank copy stream for fully resident captures with no raw uploads; repeat for each stream')
    parser.add_argument('--dependencies', action='store_true',
                        help='report conservative exposed bank dependencies; requires event activity')
    args = parser.parse_args()
    assert 0 < args.prefill_batches <= args.phases_per_run
    phases = summarize(args.database, args.phases_per_run, args.prefill_batches, args.expert_cache, args.bank_stream_id)
    result = {'database_sha256': hashlib.sha256(args.database.read_bytes()).hexdigest(),
              'scope': __doc__, 'profiler_overhead_present': True,
              'cpu_fallback_entrypoints_instrumented': False,
              'driver_internal_allocations_independently_instrumented': False,
              'expert_cache': args.expert_cache,
              'explicit_bank_stream_ids': args.bank_stream_id,
              'phases': phases}
    if args.dependencies:
        from cuda_dependencies import extract_dependencies
        dependencies = extract_dependencies(args.database)
        starts = [phase['window_start_ns'] for phase in phases]
        selected = [[] for _ in phases]
        bank_streams = set(args.bank_stream_id or [])
        if not bank_streams:
            connection = sqlite3.connect(args.database)
            try:
                bank_streams = {row[0] for row in connection.execute("""
                    SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY
                    WHERE copyKind=1 AND bytes>=3072000 AND start>=? AND end<=?
                """, (starts[0], phases[-1]['window_end_ns']))}
            finally:
                connection.close()
        assert bank_streams, 'bank streams must be identified'
        for dependency in dependencies:
            if dependency['producer_stream'] not in bank_streams:
                continue
            index = bisect.bisect_right(starts, dependency['wait_host_ns']) - 1
            if index >= 0 and dependency['wait_host_ns'] < phases[index]['window_end_ns']:
                assert dependency['consumer_kind'] == 'kernel'
                assert 'bank_project_grouped' in dependency['consumer_name'], \
                    'dependency reporting requires grouped expert banks'
                selected[index].append(dependency)
        for phase, events in zip(phases, selected):
            assert events, 'no matched bank dependencies in phase'
            intervals = [(event['interval_begin_ns'], event['interval_end_ns'])
                         for event in events if event['exposed_dependency_ns']]
            # Union prevents double counting if several waits precede a consumer.
            phase['bank_wait_events'] = len(events)
            phase['bank_positive_dependency_events'] = len(intervals)
            phase['bank_dependency_lower_bound_ms'] = union_ns(intervals) / 1e6
        result['dependency_scope'] = (
            'Union of max(0, preceding producer GPU activity end - max(prior '
            'consumer-stream activity end, first consumer API return)). '
            'Single process/device/context. Excludes host submission gaps; '
            'event readiness is bounded by preceding producer work. This is '
            'not full source stall time, transfer duration, isolated latency '
            'savings or a counterfactual critical path. Profiling overhead remains.')
    args.output.write_text(json.dumps(result, indent=2))
    print(f'{len(phases)} phase windows; {sum(p["expert_upload_bytes"] for p in phases)} streamed expert bytes; {sum(p["inference_runtime_allocation_calls"] for p in phases)} captured allocation calls')


if __name__ == '__main__':
    main()
