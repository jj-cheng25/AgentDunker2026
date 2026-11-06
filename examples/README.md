# Examples

- `agentdunker.toml` — every configuration key, commented.
- `make_synthetic_clip.py` — writes a 6 s generated clip (no NBA footage) for trying the pipeline end to end:

```bash
python examples/make_synthetic_clip.py demo.mp4
agentdunker run demo.mp4 -o out/ --offline --skip tracks,court,identity,events,replay,narration
```

Real use needs a real broadcast clip (5–20 s, one offensive possession, mostly the main wide camera). NBA footage
is copyrighted: do not commit clips or films made from them.
