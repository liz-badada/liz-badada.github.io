// Auto-apply a cluster class to <body> based on URL path, so topic pages
// pick up their cluster accent color from extra.css.

(function () {
  const CLUSTERS = {
    'in-inference':    ['engines', 'kv-cache', 'attention', 'scheduling', 'quantization'],
    'in-parallelism':  ['parallelism', 'large-scale', 'communication', 'moe', 'long-context'],
    'in-training':     ['training', 'post-training', 'agentic-rl'],
    'in-applications': ['agents', 'multimodal', 'diffusion'],
    'in-foundations':  ['hardware', 'perf-modeling', 'benchmarks'],
  };

  function apply() {
    const path = window.location.pathname;
    // Remove any stale cluster classes first
    for (const cls of Object.keys(CLUSTERS)) {
      document.body.classList.remove(cls);
    }
    for (const [cls, segments] of Object.entries(CLUSTERS)) {
      if (segments.some(seg => path.includes('/' + seg + '/') || path.endsWith('/' + seg))) {
        document.body.classList.add(cls);
        break;
      }
    }
  }

  // Run on first load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', apply);
  } else {
    apply();
  }
  // Re-run on Material's instant-navigation page changes
  if (typeof document$ !== 'undefined') {
    document$.subscribe(apply);
  }
})();
