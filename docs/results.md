# Results

Source: closed-loop FACTS note of 2026-11-02..06 (verified; re-aggregated here with `eval/closedloop/score_closedloop.py`,
which reproduces every number below from the AlpaSim `results-summary.json` files).

## Closed loop (AlpaSim, 150 scenes x 2 rollouts)

Setup: simulation GPU host (renderer, runtime) + separate driver GPU host (BF16 compiled runtime, full 16-frame window,
AE uniform Euler 10, 1 path, no navigation). All four arms on the SAME runtime. Scene set 150 = holdout58 + new32 + ext60
(ext60 pinned to NRE 26.01 artifacts). Score = 0 on at-fault collision or off-road, else min(progress / 0.8, 1).

Arms: Teacher = Alpamayo-1.5 10B; Ishinca-4B = Cosmos-Reason2-2B backbone + relation KD + AE28 action expert;
DART75 / DART50 = Ishinca-4B + DART vision-token pruning (keep 75% / 50%), retrained.

| group | Teacher 10B | Ishinca-4B | DART50 | DART75 |
|---|---|---|---|---|
| holdout58 | 0.724 | 0.767 | 0.807 | 0.734 |
| new32 | 0.719 | 0.773 | 0.772 | 0.744 |
| ext60 | 0.737 | 0.724 | 0.748 | 0.722 |
| all150 | 0.728 | 0.751 | 0.776 | 0.731 |
| all147 (3 invalid-map scenes excluded) | 0.743 | 0.767 | 0.792 | 0.746 |
| failed rollouts / 300 | 40 | 37 | 37 | 45 |
| off-road / at-fault collision | 32 / 8 | 34 / 3 | 30 / 7 | 35 / 10 |

Paired vs Teacher (all150, 95% scene bootstrap): Ishinca-4B +0.023 [-0.030, +0.078]; DART50 +0.048 [-0.007, +0.103];
DART75 +0.003 [-0.050, +0.058]; all not significant. (`score_closedloop.py` prints the same intervals to within +-0.001;
the bootstrap RNG differs.) DART50 - DART75: +0.045 [+0.016, +0.075], the only significant pair, likely noise at 2 rollouts.

Reading: Student is statistically indistinguishable from the Teacher in closed loop, and DART keeps the overall score.
An earlier runtime showed a larger student advantage on holdout58 (+0.096); with the Teacher re-run on the same
runtime it is +0.043 (n.s.).

Failure analysis (all arms): 3 invalid-map scenes (GT path itself off-road: a9e8dfc7, cc8cbc45, 29815599); the rest are
junction/turn-route errors (no navigation input -> route ambiguity), lane-keep drift and a few at-fault collisions.
The plan, not the controller, is the cause (tracking error <= 0.01 m). 14 scenes fail for all four arms.

Collision rerun (9 scenes where collisions differed, 6 new rollouts per arm): at-fault collisions / 54: Teacher 10,
Ishinca-4B 11, DART50 19, DART75 21; scene sign-flip p vs Teacher 0.33 (DART50), 0.19 (DART75). Trend (DART causes more at-fault
collisions in both experiments), not significant in either.

## Open loop

gold644 (644 clips, vs GT, protocol in `eval/openloop/PROTOCOL.md`, official-protocol benchmark scripts of the research
repository, AE mode, no navigation):

| model | ADE (rollout=1) | minADE6@6.4s (rollout=6) |
|---|---|---|
| Teacher 10B (FP16, FlashAttention-2) | 2.241 | 0.831 |
| Ishinca-4B (research bundle of 2026-10-07, bf16; mean of the two 322-clip shards) | 2.381 | 1.045 |

The Ishinca-4B row comes from the research-repo bundle of 2026-10-07; its weights were not hash-compared with the released
checkpoint, but the per-clip ADE of the released checkpoint matches it within 0.03 m on the 8 clips checked below.
gold644 is the 644-example public set of Table 10 (§6.4) in the Alpamayo-R1 paper (arXiv:2511.00088); the paper
reports minADE6@6.4s 0.849 m for Alpamayo-R1-10B and 0.913 m for Alpamayo-R1-0.5B. DART variants were not run on gold644.

Validation set (4,000 held-out PhysicalAI-AV samples built by us, not used for training; same x0 for all models,
AE uniform Euler 10; vs-Teacher reference = Teacher action-expert path 0 with its own CoC):

| model | image tokens | vs GT ADE | vs GT minADE6 | vs Teacher ADE | vs Teacher FDE | vs Teacher minADE6 |
|---|---|---|---|---|---|---|
| Teacher 10B | 2,880 | 1.750 | 0.826 | - | - | - |
| Ishinca-4B | 2,880 | 1.957 | 0.986 | 1.041 | 3.172 | 0.670 |
| Ishinca-4B-DART75 (retrained) | 2,160 | 1.957 | 0.994 | 1.043 | 3.173 | 0.686 |
| Ishinca-4B-DART50 (retrained) | 1,440 | 1.972 | 1.003 | 1.068 | 3.250 | 0.698 |

Without retraining, applying DART to Ishinca-4B costs vs-Teacher ADE +0.127 (keep 75%) and +0.466 (keep 50%); retraining
the backbone (1 epoch) and then the action expert (3 epochs) on the pruned input recovers it to +0.001 / +0.026.

Smoke check on the 8 gold644 clips that were available offline (x86, RTX 3090, bf16, seed 42; n = 8, **not** a benchmark):

| model | ADE (m) | FDE (m) | minADE6 (m) |
|---|---|---|---|
| Teacher 10B (same 8 clips, from the gold644 reference run) | 1.685 | n/a | 0.655 |
| Ishinca-4B | 2.420 | 6.352 | 1.069 |
| Ishinca-4B-DART50 | 2.399 | 7.572 | 1.069 |
| Ishinca-4B-DART75 | 2.245 | 6.679 | 1.016 |

See `docs/validation_x86.md` for the x86 run details.
