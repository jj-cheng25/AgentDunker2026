# Contributing

Thanks for helping! A few rules keep the project healthy:

1. **No footage, no weights, no keys in git.** NBA clips and films made from them are copyrighted; weights are
   downloaded by `agentdunker download-models`; credentials live in environment variables. CI rejects large files.
2. **Always a film.** A change must never turn an uncertain input into a crash: degrade the element, record the
   reason in `report.json`, keep going.
3. **Numbers, not anecdotes.** Quality changes need a measurement on several clips (and a visible example).
4. **Module code is synced.** `agentdunker/{tracking,court,identity,replay,events,narration,film}` come from the
   module branches via `tools/sync_modules.py`. Fix them upstream, or add a rule to the script; do not edit them
   in place (`python tools/sync_modules.py --check` must pass).

## Development

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,ui]'
ruff check agentdunker tests tools
pytest -q                      # unit tests + synced module tests + synthetic smoke test (no weights, no network)
```

Commit messages: `type(scope): summary` (feat, fix, docs, test, perf, chore). Licence of contributions: AGPL-3.0.
