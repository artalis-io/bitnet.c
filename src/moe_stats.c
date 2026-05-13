#include "moe_internal.h"

void bn_moe_print_stats(const BnMoEState *ms, int n_tokens) {
    if (!ms || n_tokens <= 0) return;

    double io_per_tok = (double)ms->stats.io_bytes / (1024.0 * 1024.0) / n_tokens;

    char iot_s[32], bw_s[32], rss_s[32];
    char norm_s[32], rt_s[32], gu_s[32], sw_s[32], dn_s[32], ac_s[32], sh_s[32], ct_s[32];

    snprintf(iot_s, sizeof(iot_s), "%.1f", io_per_tok);

    if (ms->stats.io_time_ms > 0.1)
        snprintf(bw_s, sizeof(bw_s), "%.0f",
                 (double)ms->stats.io_bytes / (1024.0 * 1024.0) / (ms->stats.io_time_ms / 1000.0));
    else
        snprintf(bw_s, sizeof(bw_s), "mmap");

    snprintf(norm_s, sizeof(norm_s), "%.1f", ms->stats.norm_time_ms);
    snprintf(rt_s, sizeof(rt_s), "%.1f", ms->stats.route_time_ms);
    snprintf(gu_s, sizeof(gu_s), "%.1f", ms->stats.gate_up_time_ms);
    snprintf(sw_s, sizeof(sw_s), "%.1f", ms->stats.swiglu_time_ms);
    snprintf(dn_s, sizeof(dn_s), "%.1f", ms->stats.down_time_ms);
    snprintf(ac_s, sizeof(ac_s), "%.1f", ms->stats.accum_time_ms);
    snprintf(sh_s, sizeof(sh_s), "%.1f", ms->stats.shared_time_ms);
    snprintf(ct_s, sizeof(ct_s), "%.1f", ms->stats.compute_time_ms);

    char pw_s[32], ma_s[32];
    snprintf(pw_s, sizeof(pw_s), "%.1f", ms->stats.prefetch_wait_ms);
    snprintf(ma_s, sizeof(ma_s), "%.1f", ms->stats.madvise_time_ms);

    size_t rss = bn_platform_rss_bytes();
    snprintf(rss_s, sizeof(rss_s), "%.2f", (double)rss / (1024.0 * 1024.0 * 1024.0));

    SH_LOG_INFO("MoE stats",
                "MB/tok", iot_s,
                "stream_MB/s", bw_s,
                "rss_GB", rss_s);
    SH_LOG_INFO("MoE breakdown (ms)",
                "norm", norm_s,
                "route", rt_s,
                "gate+up", gu_s,
                "swiglu", sw_s,
                "down", dn_s,
                "accum", ac_s,
                "shared", sh_s,
                "pf_wait", pw_s,
                "madvise", ma_s,
                "total", ct_s);

    bn_moe_cache_print_stats(ms);
}

void bn_moe_reset_stats(BnMoEState *ms) {
    if (!ms) return;
    ms->stats.io_bytes = 0;
    ms->stats.io_time_ms = 0;
    ms->stats.route_time_ms = 0;
    ms->stats.compute_time_ms = 0;
    ms->stats.gate_up_time_ms = 0;
    ms->stats.swiglu_time_ms = 0;
    ms->stats.down_time_ms = 0;
    ms->stats.accum_time_ms = 0;
    ms->stats.shared_time_ms = 0;
    ms->stats.norm_time_ms = 0;
    ms->stats.io_count = 0;
    ms->stats.prefetch_wait_ms = 0;
    ms->stats.madvise_time_ms = 0;
    ms->stats.cache_hits = 0;
    ms->stats.cache_misses = 0;
}

