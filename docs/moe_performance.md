# MoE Performance Notes

MoE performance is mostly a function of expert access locality, dispatch
batching, and I/O policy.

## Current Design

- Router work is vectorized and parallelized across experts.
- Active expert gate/up projections are batched where the inputs match.
- Down projections are dispatched as a multi-task batch when each expert has its
  own post-activation input.
- mmap is the fast path when the expert working set fits in RAM and the page
  cache is warm.
- pread plus the expert LRU cache lowers RSS for larger sparse models.

## Practical Modes

| Mode | Use when |
|---|---|
| mmap | Model and hot expert set fit comfortably in RAM. |
| `--pread --cache-mb N` | You need lower RSS or more predictable memory limits. |
| `--pread --cache-mb 0` | Memory is constrained and lower throughput is acceptable. |

## Optimization History

The major completed optimization was reducing per-layer expert dispatch count by
batching gate/up and down work across active experts. This changed MoE execution
from many small thread-pool dispatches into a few larger dispatches per layer.

Keep new MoE benchmark claims tied to:

- exact model file
- quant format
- thread count
- generated token count
- cache mode and cache size
- cold versus warm page-cache state
- backend placement
