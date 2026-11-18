# RoboDojo adapter

The adapter uses the RoboDojo environment through RLinf's interactive
`CollectEnv` integration. Each environment owns a scene and returns official
success and progress. The Pi0.5 route accepts and executes the benchmark's
14-value joint actions.

## Resources

- Build the GPU runtime following [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
  and the [upstream integration guide](../../../third_party/README.md).
- Set `ROBODOJO_ASSETS_ROOT` to the directory containing `Assets/`. Prepare the
  provided cuRobo configuration by substituting the absolute assets path for
  `${ASSETS_PATH}` in `curobo_tmp.yml`.
- Set `PI05_CHECKPOINT_PATH` to the matching RoboDojo policy and
  `SAM3_CHECKPOINT_PATH` to the segmentation weights. Set
  `PI05_NORM_STATS_PATH` when the policy has separate normalization statistics.
- Use `--sim-python` and `--pi05-python` for separate interpreters when needed.
  `--source-root` and `--xpolicylab-root` select explicit upstream checkouts.
  Compatible running services can be passed as `--vla-endpoint` and
  `--sam3-endpoint`.

The environment step returns per-action acknowledgement, proprioception and a
reference to the server-owned camera snapshot. Image and perception requests
load that exact snapshot. The interactive bridge must use the upstream
zero-delay capture setup and call `render_for_capture()` before camera reads.

## Run

```bash
reweave --robot robodojo --task put_bottles_into_dustbin --layout-set 1 \
  --layout 0 --cuda-device 0 --max-episode-steps 700 --wait-steps 0 \
  --output-dir runs/robodojo
```

This starts a simulator episode. For a CPU-only interface check, run the
[scripted demo](../../../examples/robodojo_demo.py).
