# Closed-loop evaluation (AlpaSim)

150 scenes x 2 rollouts, same runtime for every arm (Teacher 10B, Pisco-4B, DART50, DART75). Results: `docs/results.md`.

## Contents

| Path | What |
|---|---|
| `driver_plugin/` | pip package `pisco_alpasim`: registers `model_type: pisco` in AlpaSim's `alpasim.models` entry-point group (thin adapter over `Alpamayo15Model` + `pisco.load_pisco`) |
| `driver_plugin/smoke_predict.py` | simulator-free check: one saved sample through AlpaSim's driver `predict()` |
| `driver_configs/` | driver configs for Pisco-4B / DART50 / DART75 |
| `topology/remote_driver_resize576.yaml` | wizard topology: simulation host + remote driver host |
| `scenes/scenes150_20261101.txt` | the 150 scene IDs in run order (holdout58 + new32 + ext60) |
| `scenes/holdout58.txt`, `scenes/closedloop_ext60_20261030.json` | group membership (frozen selection, `training_use_forbidden: true`) |
| `scenes/scenes_collrerun9_20261104.txt` | 9-scene collision rerun set |
| `scenes/sim_scenes_ext60pin_20261102.csv` | scene catalog pinned to the NRE 26.01 artifacts used for ext60 |
| `score_closedloop.py` | re-aggregation: per-group score, failures, paired scene bootstrap, invalid-scene exclusion |

## Invalid scenes

Three scenes have a recorded GT path that is itself off-road in the reconstructed map and fail for every arm:
`clipgt-a9e8dfc7-...`, `clipgt-cc8cbc45-...`, `clipgt-29815599-...` (matched by clip-id prefix in `score_closedloop.py`).
They stay in the scene list; `score_closedloop.py` reports `all150` (included) and `all147` (excluded for all arms).

## Scoring

AlpaSim `score_criteria`: a rollout fails (score 0) on at-fault collision or off-road, otherwise
`min(clamp(progress_clipped_rel, 0, 1) / 0.8, 1)`; rear collisions are not at-fault. Verified: the formula reproduces
AlpaSim's per-rollout `score` for all 300 rollouts of each arm. Statistics are paired scene bootstraps (scene = unit).

```
python eval/closedloop/score_closedloop.py \
  --arm teacher="runs/teacher/*/aggregate/results-summary.json" \
  --arm pisco4b="runs/pisco4b/*/aggregate/results-summary.json" \
  --arm dart50="runs/dart50/*/aggregate/results-summary.json" --baseline teacher
```

## Running

1. Install AlpaSim (https://github.com/NVlabs/alpasim) on the simulation host and the plugin on the driver host:
   `pip install -e . && pip install -e eval/closedloop/driver_plugin` (the driver host needs `alpasim_driver`).
2. Download a Pisco checkpoint on the driver host and edit `checkpoint_path` in `driver_configs/driver_pisco4b*.yaml`.
   DART variants read `pisco_dart_keep` from the checkpoint `config.json`, or set `PISCO_DART_KEEP=0.5` in the driver env.
3. Start the driver (`alpasim_driver_main --config-name driver_pisco4b port=6789`), then on the simulation host:

```
uv run alpasim_wizard deploy=local topology=remote_driver_resize576 driver=<driver cfg> \
  "wizard.external_services.driver=[<DRIVER_HOST>:6789]" scenes.scenes_csv=[eval/closedloop/scenes/sim_scenes_ext60pin_20261102.csv] \
  "scenes.scene_ids=[<ids from scenes150>]" runtime.simulation_config.n_rollouts=2
```

## Remote-driver topology

Renderer, physics, traffic, controller and runtime run on one GPU host; the driver runs on another GPU host and is
called over gRPC, one rollout at a time. Reasons: the NRE renderer peaks at ~18 GB (a 24 GB card cannot also hold the
driver) and the driver keeps per-rollout state. Camera frames are downscaled to 576x320 on the simulation host before being sent.

## Differences from the published numbers (read before comparing)

* The reported runs used the BF16 compiled runtime (static KV cache + CUDA graphs, 16-frame window, AE uniform Euler 10,
  1 path, no navigation) inside our AlpaSim fork. `driver_plugin` uses the eager PyTorch rollout of this repository:
  same model, prompt, sampling rule and Euler steps, but not the compiled kernels, so rollouts are not bit-identical
  and the driver is much slower. [UNVERIFIED: closed-loop scores with the eager Pisco plugin were not re-run. Verified:
  `smoke_predict.py` through AlpaSim's driver `predict()` (aarch64 driver env) reproduces `scripts/infer.py` exactly with the
  same x0; the entry-point registration and a full simulator run were not exercised.]
* Our fork carries runtime patches that are not upstream: driver RPC keepalive / `wait_for_ready` / retries over a LAN,
  an idempotent driver server, and the optional `ALPASIM_DRIVER_IMAGE_WH` frame downscale. These must be published or
  re-applied for the topology file to work as is.
* Closed-loop runs are not deterministic (scene physics, traffic and CUDA nondeterminism); 2 rollouts per scene.
