"""Conservative exposed CUDA stream dependencies from an Nsight SQLite export.

Supports one captured process/device/context. Returned intervals are lower bounds
on unsatisfied producer dependencies, not counterfactual latency savings. Event
GPU timestamps may be absent: preceding producer activity bounds readiness.
Kernel, copy, memset, event and runtime submission records must be captured.
"""
import bisect
import sqlite3


def extract_dependencies(database):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        identities = list(connection.execute('''
            SELECT DISTINCT deviceId, contextId, globalPid
            FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT
        '''))
        assert len(identities) == 1, 'requires one captured process/device/context'
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        work = []
        for table, kind in [('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                            ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'),
                            ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')]:
            if table not in tables:
                continue  # Nsight omits tables for activity kinds with no rows.
            activity_identities = list(connection.execute(f'''
                SELECT DISTINCT deviceId, contextId, globalPid FROM {table}
            '''))
            assert all(tuple(row) == tuple(identities[0])
                       for row in activity_identities), 'mixed capture identities'
            name = 's.value' if kind == 'kernel' else "''"
            join = 'JOIN StringIds s ON s.id=a.demangledName' if kind == 'kernel' else ''
            work.extend(dict(row, kind=kind) for row in connection.execute(f'''
                SELECT r.start host_start, r.end host_end, a.start, a.end,
                       a.streamId, a.correlationId, {name} name
                FROM {table} a JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                  ON r.correlationId=a.correlationId {join}
            '''))
        streams = {}
        for stream in {row['streamId'] for row in work}:
            rows = sorted((row for row in work if row['streamId'] == stream),
                          key=lambda row: row['host_start'])
            prefix, end = [], 0
            for row in rows:
                end = max(end, row['end'])
                prefix.append(end)
            streams[stream] = rows, [row['host_start'] for row in rows], prefix
        waits = list(connection.execute('''
            SELECT s.start wait_start, s.end wait_end, s.streamId,
                   e.streamId producer_stream, e.eventId, e.eventSyncId,
                   r.start record_start, r.end record_end
            FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION s
            JOIN CUPTI_ACTIVITY_KIND_CUDA_EVENT e
              ON s.eventId=e.eventId AND s.eventSyncId=e.eventSyncId
             AND s.contextId=e.contextId AND s.deviceId=e.deviceId
             AND s.globalPid=e.globalPid
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=e.correlationId
            WHERE s.syncType=2 ORDER BY s.start
        '''))
        expected = connection.execute('''
            SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
            WHERE syncType=2
        ''').fetchone()[0]
        assert len(waits) == expected, 'missing or ambiguous event association'
        result = []
        for wait in waits:
            rows, starts, prefix = streams[wait['streamId']]
            _, producer_starts, producer_prefix = streams[wait['producer_stream']]
            index = bisect.bisect_left(starts, wait['wait_start'])
            producer_index = bisect.bisect_left(producer_starts, wait['record_start'])
            assert index < len(rows) and producer_index, 'activity missing'
            prior_end = prefix[index - 1] if index else 0
            producer_end = producer_prefix[producer_index - 1]
            consumer = rows[index]
            assert consumer['start'] >= producer_end, 'event dependency violated'
            assert wait['record_end'] <= wait['wait_end'] <= consumer['host_start'], \
                'host submission order is ambiguous'
            # API return bounds when the consumer was submitted. Exclude any
            # prior compute and host submission delay from the dependency span.
            begin = max(prior_end, consumer['host_end'])
            result.append(dict(
                wait_host_ns=wait['wait_start'],
                producer_stream=wait['producer_stream'],
                consumer_stream=wait['streamId'], event=wait['eventId'],
                event_sync=wait['eventSyncId'],
                producer_ready_lower_bound_ns=producer_end,
                prior_compute_end_ns=prior_end,
                consumer_enqueue_complete_ns=consumer['host_end'],
                consumer_gpu_start_ns=consumer['start'],
                consumer_kind=consumer['kind'], consumer_name=consumer['name'],
                consumer_correlation=consumer['correlationId'],
                interval_begin_ns=begin, interval_end_ns=max(begin, producer_end),
                exposed_dependency_ns=max(0, producer_end - begin)))
        return result
    finally:
        connection.close()
