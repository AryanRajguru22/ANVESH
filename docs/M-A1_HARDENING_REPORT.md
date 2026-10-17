# M-A.1 — Critical correctness hardening: report

Scope: the defects found by the post-milestone review of M-A (`docs/M-A_FOUNDATION_REPORT.md`, not modified). This is **not** M-B and **nothing here makes ANVESH production-ready**. Nothing was committed. Environment: RTX 3050 6 GB laptop, Python 3.11.9, project `.venv`.

Evidence labels used throughout: **[synthetic]** = hermetic test/fixture/probe on constructed data; **[real]** = run on real CityFlow / `demo.mp4` video and the real YOLOv8n+ByteTrack or its cached output; **[read]** = code inspection only.

## 1. Baseline

| | |
|---|---|
| Branch / HEAD | `main` / `042ea4bfb22a9f69e820b14e7e2c45f8a4af7be3` |
| Working tree at start | 71 `git status` entries (27 modified + untracked M-A work: `anvesh/streaming/`, `anvesh/evaluation/*`, M-A scripts/tests/reports, `outputs/`). All **uncommitted**; preserved. Snapshot kept in the session scratchpad (`baseline_status.txt`). |
| Tests at start | **692 passed** |
| Review used | the M-A post-milestone review delivered earlier in the same session (no review file existed in the repository) |

## 2. Findings independently reproduced (before any fix)

| # | Review finding | Reproduction (pre-fix) | Evidence |
|---|---|---|---|
| S1 | Validated geometry with no measured speeds still ranks | 18 validated-geometry tracks, all `speed_available=False`, congested → H1 `supports/full`, H2 `contradicts/full`, **`ranked`, H1 belief 0.946, tier `high`** | **[synthetic]** probe `probe_h1_validated.py` |
| S2 | `WORLD` track silently becomes `WORLD_VALIDATED` when validity is omitted | `VehicleTrack(motion_space=WORLD)` accepted; `export_ui_data` and demo scripts built tracks this way | **[read]** + test |
| E1 | Outage reads as free flow | 35 s reconnect gap: windows w1–w3 `frames=0, dropped=0, degraded=False`, congestion `free_flow` | **[synthetic]** probe `probe_stream.py` #1 |
| E2 | Drops charged to the wrong window | frames 95–104 lost: w0 `dropped=0`, w1 `dropped=10`; outage: all 349 lost frames charged to the recovery window | **[synthetic]** probe #1, #2 |
| I1 | Session id reusable | `TrackManager.begin_session` accepted `s000→s001→s000` (duplicate key); same through the **engine** with epochs 0→1→0 | **[synthetic]** probes #3, `probe_dwell_session.py` #7 |
| R1 | Boundary-spanning stop missed in both windows | stop 7.5–12.5 s, 10 s windows: no safety event in w0 or w1; unclipped track is `STOPPED` | **[synthetic]** probe `probe_dwell_session.py` #6 |
| D | Abstention rationale | `ranking.py` blames Dempster normalisation; `single_camera_result` never combines or discounts — the belief was the raw H1 score | **[read]** |

Findings discovered **while doing this work** (not in the review): `tick()` acting during normal inter-frame idle time (introduced by me, caught in review of my own work, fixed); a half-searched validated window still ranking H1 `high` (belief 0.913); the new `stop_episodes` result field not being populated (caught by the real c006 run); the image-space stop heuristic being jitter-fragile on real data (§9).

## 3. Files changed and why

**Production code**

| File | Why |
|---|---|
| `anvesh/storage/schemas.py` | `VehicleTrack`: validity mandatory for `WORLD`, position-unit consistency check; `TrafficState.mean_speed_available` (schema 1.1.0); documented `TrafficState` validity gap |
| `anvesh/storage/db.py` | persist/restore `mean_speed_available` (old rows default `True`) |
| `anvesh/perception/speed_estimation.py` | a km/h `measurement` no longer implies `WORLD_VALIDATED` |
| `anvesh/perception/traffic_state.py` | `mean_speed_available`; `CongestionStateTracker.reset()` |
| `anvesh/evidence/evidence_builder.py` | central `ObstructionDetection` availability from real speed coverage; H1 unavailable when no obstruction search was possible; H1 absence-evidence scaled by coverage; PARTIAL completeness |
| `anvesh/evidence/reliability.py` | calibration factor reflects speed coverage; documented frame-loss gap |
| `anvesh/hypotheses/cause_model.py`, `ranking.py` | `COMPETITOR_PREREQUISITES`; eligibility derived from evidence (completeness **and** direction, H1 needs an obstruction test); stale rationale corrected; single-camera semantics documented |
| `anvesh/fusion/ds_fusion.py`, `anvesh/pipeline.py` | `single_camera_discounted_result` (Baseline A `single_camera_result` untouched); `run_corridor_window(stop_episodes=)`; result carries `stop_episodes` |
| `anvesh/perception/track_manager.py` | sessions never reused; identity-scope documentation |
| `anvesh/streaming/engine.py` | run-scoped ids by default; unique session names; drop attribution by estimated lost-frame time; time-based observed coverage; `tick()`; `StreamHealth`; stop-episode tracking |
| `anvesh/streaming/windows.py` | `WindowAvailability`, `silent_s`, `coverage`, `expected_frames`, `stop_episodes` on the snapshot |
| `anvesh/streaming/reasoning.py` | unavailable/degraded handling; hysteresis protection; stop events |
| `anvesh/streaming/pipeline.py`, `rtsp.py` | idle polling (`idle_poll_s`); `RtspFrameSource.stream_time()` |
| `anvesh/perception/pipeline.py` | `run_m1_pipeline` deterministic per-recording run id |
| `anvesh/safety/stop_episodes.py` (new) | stop episodes across windows |
| `anvesh/safety/stopped_vehicle.py` | docstring cross-reference only |

**Scripts / UI**: `scripts/export_ui_data.py` (placeholder speeds exported as `null`, episodes via the shared tracker, synthetic fixtures declare validity), `scripts/c006_safety_report.py` (M-A.1 column, tolerates withheld speeds, writes to `outputs/m_a1/`), `scripts/stream_parity.py` (+`--out-dir`, now compares availability and stop episodes), `scripts/_stream_common.py`, `scripts/stream_bench.py`, `scripts/stream_soak.py` (+`--out-dir`), three synthetic demo scripts (explicit validity), `ui/anvesh_dashboard.template.html` (shows "speed unavailable"). **Built dashboards `ui/anvesh_dashboard.html` and `ui/live/*.html` were not regenerated** (stale since M-A; they read `speed_estimate.value` unguarded and must be rebuilt before use, never pointed at M-A.1 exports).

**New tests** (80; suite 692 → 772): `tests/test_ma1_missing_speed_and_geometry.py` (27), `tests/streaming/test_ma1_identity.py` (11), `tests/streaming/test_ma1_outage.py` (25), `tests/streaming/test_ma1_stop_episodes.py` (17).

### Pre-existing tests whose expectations I changed (none weakened or deleted)

| Test(s) | Change | Why |
|---|---|---|
| 13 `VehicleTrack` fixtures in 12 test files + 7 constructions in 4 scripts (3 demos + the exporter's synthetic fixtures) | `VehicleTrack(WORLD)` now declares `geometry_validity=WORLD_VALIDATED` (conditional where the space is a parameter) | the implicit default is gone. **This label is the fixtures' declaration; none of these cameras is validated** |
| `test_speed_safety_dataflow::test_a_world_track_cannot_be_declared_without_validated_geometry` | assertion **flipped**: omitted validity used to be derived; now must raise | deliberate contract change (Fix 2) |
| `test_evidence_builder::test_image_space_h1_is_only_partial_completeness` → `..._h1_is_unavailable_because_no_obstruction_search_was_possible` | `PARTIAL` → `NONE`/`INSUFFICIENT` | H1 asserts nothing without an obstruction search (Fix 1/6) |
| `test_ranking::test_evaluable_hypothesis_ids_counts_across_cameras` | counts H2/H3 instead of a lone H1 | H1 alone is no longer an eligible competitor (Fix 6) |
| 9 `VehicleSpeedEstimate` fixtures (`test_rendering`, `test_speed_estimation`, `test_speed_evidence_safety`, `test_export_ui_data`) | declare `WORLD_VALIDATED` | a km/h value no longer implies it |
| `test_engine_and_windows` (2), `test_sources_queues_pipeline` (3) | `run_id=None` where bare `s000` ids are asserted; shared `run_id="parity"` where two engines are compared | run-scoped ids are now the default |
| `test_pipeline_end_to_end` (3) | expected ids carry the deterministic per-recording prefix | same |
| `test_export_ui_data` result stub | + `stop_episodes = ()` | mirrors the new `AnveshWindowResult` field |

## 4. Design changes

**Fix 1 — missing-speed evidence.** The ability to look for stopped vehicles is derived **once**, from the tracks present (`_obstruction_detection`): the space must be `WORLD` **and** at least `min_speed_coverage` (0.5) of the window's vehicles must carry a real measurement. H1, H2, H3 and the stopped branch of H5 share it, so they become unavailable together (`NONE`/`INSUFFICIENT`, score 0, never `CONTRADICTS`). Between 50% and 100% coverage evidence is `PARTIAL`, measured stopped vehicles still count, and H1 — an *absence* claim — is scaled by the searched fraction. `TrafficState.mean_speed_available` carries "no real mean" to storage, the exporter (`null`) and the dashboard ("speed unavailable"). A genuinely measured 0 m/s stays distinguishable (`speed_available=True`) and is a stopped vehicle.

**Fix 2 — explicit validity.** A `WORLD` `VehicleTrack` must declare `WORLD_VALIDATED` (omitting raises); `IMAGE` may omit (can only derive `IMAGE`). `WORLD` requires `WorldPosition` entries, `IMAGE` forbids them. `VehicleSpeedEstimate` likewise. An AST test asserts every `VehicleTrack(...)` call under `anvesh/` passes `geometry_validity`. The only production assignment of `WORLD_VALIDATED` is `CalibrationProfile.geometry_validity` (validated **and** metres) and the two builders that branch on it.

**Fix 3 — outages and drops.** Each window now reports `availability` (`OBSERVED` / `DEGRADED` / `UNAVAILABLE`), time-based `coverage` and `silent_s` (silence = a gap > `max(max_gap_s, 2.5 frame intervals)`), `expected_frames`. Lost frames are spread over the interval between the previous and the current frame and charged to the windows that interval overlaps. `WindowReasoner`: `UNAVAILABLE` (coverage < `min_coverage` = 0.5, or zero observed frames) ⇒ **nothing is reasoned**, hysteresis is reset; `DEGRADED` ⇒ reasoned, flagged ("counts are lower bounds"), and the hysteresis advances only if coverage ≥ `min_update_coverage` (0.9), otherwise the window is analysed against a **copy** of the tracker. A healthy empty scene stays `OBSERVED` free-flow. Liveness: `engine.tick(stream_time, source_state)` (closes windows, ages tracks, updates `StreamHealth` NO_SIGNAL/LIVE/STALLED/DOWN) is acted on **only after a genuine silence**; `StreamingPipeline(idle_poll_s=)` calls it on queue timeouts; `RtspFrameSource.stream_time()`.

**Fix 4 — identity.** `TrackManager` remembers every session id and refuses reuse. The engine never reuses a session name: a repeated epoch gets `s000e1`, `s000e2`… (monotonic epochs keep plain names). **Run scoping is now the default**: a random 8-hex run prefix per engine instance; `run_id=None` is the explicit session-only opt-out; `run_m1_pipeline` defaults to a **deterministic** per-recording prefix (`derive_run_id(camera, file name, size)`). Consequence: **streaming ids are no longer reproducible across runs** unless the caller passes `derive_run_id(...)`. Documented scope: an identity is a *tracklet*; there is no re-ID, stitching or cross-camera claim.

**Fix 5 — stop episodes.** `StopEpisodeTracker` follows each tracklet's continuous low-speed run across windows (same step definition as `_longest_low_speed_run`; confirmed at `dwell_seconds`; closed by a moving step, a silence, or the tracklet ending). Event id = `(track, run start)`, stable. Exactly one window (the one containing `confirmed_at`) reports the episode `OPENED` and yields the `SafetyEvent`; later windows see `CONTINUING`, the ending window `CLOSED`. Image-space episodes are labelled `image_px_per_s_heuristic` (a pixel heuristic, **not** a real-world speed); only validated metric space uses `metric_m_per_s`. The recorded-file exporter feeds the same tracker from final states. The per-window classifier remains the path for whole-lifetime tracks (`stop_episodes=None`).

**Fix 6 — evaluability and abstention.** Eligibility = completeness ≠ `NONE` **and** direction ≠ `INSUFFICIENT`, plus prerequisites (H1 needs H2 or H3 evaluable on the same camera). "Tested and found nothing" stays eligible; "could not test" does not. The production single-camera path now applies the camera's reliability discount (`single_camera_discounted_v1`); Baseline A is unchanged. `ranking.py` documents what a single-camera ranking does and does not establish and corrects the earlier (wrong) Dempster-normalisation rationale.

## 5. Regression tests added (80)

Fix 1: validated geometry + all speeds unavailable → abstains, tiers low, no resolved candidate; missing ≠ zero ≠ stopped ≠ contradicting; mixed coverage → `PARTIAL`; insufficient coverage → unavailable; threshold is a parameter; measured zero is a stopped vehicle and distinct from unavailable; full-coverage still ranks, discounted by reliability; **partial coverage never reaches HIGH and belief falls monotonically**; storage round-trip incl. legacy rows; reliability factor; exporter nulls.
Fix 2: omitted validity rejected (string and enum forms); contradictory validity; unit-inconsistent / mixed positions; IMAGE may omit; explicit valid accepted; km/h estimate never implies validity; AST audit of production constructors.
Fix 3: 35 s gap (windows unavailable, no traffic state, drops charged correctly, recovery window observed, new session); total silence via `tick` (health transitions LIVE→STALLED→DOWN, windows closed without a frame); tracks aged out; source-reported `reconnecting`; NO_SIGNAL; non-advancing/backwards clock; healthy empty road stays observed free-flow; boundary-spanning drops split 5/5; source-reported gap attributed by time; EOF/complete file fully observed; brief gaps; partial silence coverage; reasoner unavailable / zero-frame / lightly degraded / heavily degraded (hysteresis protected) / observed unchanged; hysteresis reset after an outage; threaded pipeline idle polling (fake silent source); RTSP stream clock; **idle ticks between ordinary frames change nothing** (equality with a no-tick run; verified to fail when the guard is removed).
Fix 4: manager refuses reuse; engine epochs 0→1→0, flapping epochs, monotonic reconnects, unchanged epoch; repeated raw ids across a reset; two default engines never collide; `run_id=None` opt-out collides (why it is not the default); explicit run id reproducible; scope reported; batch pipeline ids deterministic per recording and distinct across recordings.
Fix 5: boundary-spanning stop detected exactly once (was 0) and equals the unclipped classification; stop persisting across 4 windows is one incident with OPENED/CONTINUING/CLOSED; stop ending after a window; sub-threshold stops ignored; two stops by one tracklet; silence breaks a run; image vs metric thresholds and labels; confidence grows with dwell; only the opening window yields an event; streaming = post-hoc; legacy path unchanged; a stop confirmed in an unavailable window is not lost; result carries episodes; **new state stays bounded over 10,000 frames**.
Fix 6: H1 needs an obstruction test; INSUFFICIENT evidence never eligible whatever its flag; evaluated-but-unsupported stays eligible; missing evidence contributes no mass; discounting moves mass to ignorance and leaves Baseline A raw.

## 6. Before / after

| Defect | Before | After |
|---|---|---|
| S1 validated, 0 speeds measured | `ranked`, H1 **0.946 high** | `insufficient_evidence`, H1/H2/H3 `none`, all tiers low |
| Half-searched window (9/18 measured) | H1 **0.913 high** | H1 0.456, tier medium; 100% coverage unchanged (0.974 high) |
| S2 omitted validity | accepted as `WORLD_VALIDATED` | `ValueError` |
| E1 35 s outage | w1–w3 `free_flow`, `degraded=False` | w0–w3 `UNAVAILABLE`, no traffic state, hysteresis reset |
| E2 drop attribution | w0 0 / w1 10; outage: all 349 in recovery window | w0 5 / w1 5; outage 50/100/100/99/0 (sum 349) |
| I1 `0→1→0` | duplicate key `(cam,s000,0)` | `s000, s001, s000e1`, 0 duplicates |
| R1 stop 7.5–12.5 s | **0 events** | **1 event** in w1, window (7.5, 12.5), stable id |
| Dead camera | silence forever | `tick()` ⇒ windows close unavailable, health DOWN |

## 7. Test results

Commands (project `.venv`, run from `D:\ANVESH`; `-B -p no:cacheprovider` so nothing is written to the repo):

```
.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q                         # 772 passed (was 692)
.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests/streaming tests/perception/test_tracking_reset.py   # 91 passed, run 5x consecutively: stable
.venv\Scripts\python.exe scripts\run_s02_c006_showcase.py --sample-seconds 30 --output outputs\m_a1\s02_c006_data_AFTER_ma1_30s.json
.venv\Scripts\python.exe scripts\c006_safety_report.py
.venv\Scripts\python.exe scripts\stream_parity.py --out-dir outputs\m_a1
.venv\Scripts\python.exe scripts\stream_soak.py --max-frames 12000 --warmup-s 60 --tag ma1_soak_8min --out-dir outputs\m_a1
```

Also executed OK: `run_stopped_vehicle_demo`, `run_end_to_end_demo`, `run_corridor_state_demo`, `run_world_alignment_demo`; every `scripts/*.py` parses.

## 8. c006 safety results **[real]** (`outputs/m_a1/`)

30 s of S02/c006, real YOLOv8n+ByteTrack, 107 tracks, 3 windows, unvalidated CityFlow homography (`image` space):

| | motion space | median "speed" | H1/H2/H3 completeness | outcome | H1 tier |
|---|---|---|---|---|---|
| M-A "AFTER" | image | 71/55/51 px/s | partial/none/none | insufficient ×3 | low (belief 0.87/0.89 still displayed) |
| **M-A.1** | image | same | **none/none/none** | insufficient ×3 | low (belief **0.00**) |

The only stop event (track `t00018`, 10.0–13.1 s) is the same one M-A found; it now has a stable id and an `image_px_per_s_heuristic` label. Full cached 211 s run **[real]**: legacy per-window classifier and episodes both report 2 stops; the second shows the lifecycle (opened in w18, **closed** in w19, one event, not two).

**Batch/stream parity [real, cached tracker output, 22 windows]**: sync == threaded == paced exactly, **now including availability (all `observed`), coverage, drops and stop-episode phases**; the post-hoc batch agrees on the 2 opened episodes; the known post-hoc difference is unchanged (`occlusion_state` only, 349 tracks, max reliability delta 0.0789). *(Correction: my first parity statement covered track content only; the script now checks the new fields and this is the verified result.)* Part B on the **final code** (real YOLOv8n+ByteTrack on the GPU, first 60 s): run-to-run and sync-vs-threaded both identical, outcomes identical.

**Soak [real, `demo.mp4`, 12,000 frames = 480 s of video, unpaced, BLOCK queue]**: 0 dropped, 48 windows (all abstained: image space), **2,238 identities = 2,238 unique**, RSS 540 → 1,362 MiB (model load) → one 135 MiB step down → 1,229 MiB flat; retained tracks 13–62 after warm-up; `frame_times` ≤ 249 samples. `verdict_rss_flat=false` again because of a *decrease*, not growth. Throughput 29 fps is **not comparable** with M-A's 38 fps (I ran GPU probes and test suites concurrently). Isolated engine overhead with 40 tracks: 0.34 ms/frame.

## 9. Unresolved issues

1. **Image-space stop heuristic is jitter-fragile (pre-existing).** On real c006 the vehicle behind the 10.0–13.1 s event is parked for the whole clip; bbox jitter spikes (5.8 px/s at 9.0 s, 9.2 px/s at 10.0 s) exceeded the 5 px/s per-step threshold and fragmented one long stop into runs, so the episode "starts" at the window boundary only by chance. Episode logic inherits this; fixing it needs smoothing/displacement-over-time, not a threshold tweak. Blocks any P3 stopped-vehicle claim.
2. **Reliability ignores frame loss** (documented in `reliability.py`); degraded windows are flagged but not down-weighted.
3. **`TrafficState(motion_space=WORLD)` has no validity guard** (documented): exit criterion 2 is structural for `VehicleTrack` and `VehicleSpeedEstimate` only.
4. Drop attribution assumes lost frames were spread evenly between the neighbouring frames; windows closed by a tick before the reconnect frame arrives carry **lower-bound** drop counts.
5. `stale_raw_id_restarts` (2 in this soak, 10 in M-A's) is still unexplained (fragmentation, not duplication).
6. ID-switch 42-vs-40 vs `motmetrics` remains unverified (motmetrics not in `.venv`).
7. New **untuned policy parameters** (defaults, not derived from data): `min_speed_coverage=0.5`, `WindowSpec.max_gap_s=1.0` (effective ≥ 2.5 frame intervals), `WindowSpec.min_coverage=0.5`, `WindowReasoner.min_update_coverage=0.9`, `stalled_after_s=2.0` / `down_after_s=10.0` (≥ 5 frame intervals), H1 coverage scaling (linear), `COMPETITOR_PREREQUISITES`. Stop thresholds are unchanged (3 s dwell, 1 s gap, 5 px/s, 0.5 m/s).
8. Streaming ids are not reproducible across runs by default (use `derive_run_id`); identity is a tracklet, never a vehicle.
9. Built dashboards under `ui/` are stale and unsafe with new exports (see §3).

### Incidents (disclosed)
- I **overwrote `outputs/m_a/c006_safety_report.json` twice** by running `c006_safety_report.py` while its edit had not applied. I restored it by recomputing it from its three unchanged input JSONs and stripping the added keys; the values match what I read before the incident (e.g. median 381.264…, H1 0.851). It is byte-identical to **my restored copy**; the original bytes cannot be verified. The script now writes M-A.1 results only to `outputs/m_a1/`.
- Running `run_corridor_state_demo.py` regenerated `outputs/corridor_state_demo.sqlite3` (a synthetic, gitignored demo artifact).
- `outputs/m_a/stream_parity.json` was backed up and verified untouched.

## 10. Implemented vs tested synthetically vs validated on real data

| Capability | Implemented | Synthetic tests | Real data |
|---|---|---|---|
| Missing-speed evidence safety, coverage scaling, eligibility | yes | **yes** (the only place it runs) | c006 is image-space ⇒ abstains, but **no `WORLD_VALIDATED` camera exists**, so the validated-geometry path has never run on real data |
| Explicit geometry validity | yes | yes | n/a |
| Outage / drop / degraded handling | yes | yes (incl. threaded fake source) | file mode only: complete recordings are `observed`; **no real outage ever occurred** |
| RTSP source + `stream_time` | yes | yes (fake captures) | **never run against a camera or server** |
| Session / run identity | yes | yes | c006 (107 tracks) and 8-min soak: 0 duplicates |
| Stop episodes | yes | yes | c006: 2 stops, lifecycle seen once; image-space only; no metric episode on real data |
| Single-camera discounted ranking | yes | yes | all real windows abstain (image space) ⇒ discount untested on a real *ranked* window |

## 11. Exit criteria

| Criterion | Status | Evidence |
|---|---|---|
| Missing speed cannot create unsupported high-confidence evidence | **Met** for tested scenarios | S1 and half-coverage tests **[synthetic]**; thresholds are untuned policy |
| Metric world space requires explicit valid metadata | **Partially met** | `VehicleTrack`, `VehicleSpeedEstimate`: structural. `TrafficState`: no guard (§9.3) |
| Outages cannot masquerade as free flow | **Met** in the engine/reasoner path | 35 s gap, silence, threaded idle tests **[synthetic]**; RTSP unvalidated |
| Drop attribution and degraded handling correct | **Met** under the even-spread assumption | boundary and outage tests **[synthetic]** |
| Session/run identities cannot collide in tested lifecycles | **Met** | epoch/reset/reconnect/cross-run tests **[synthetic]**, 0 duplicates in c006 and 8-min soak **[real]** |
| Boundary-spanning stops detected without duplicate incidents | **Met** | tests **[synthetic]**; one real open→close across windows; jitter caveat (§9.1) |
| Evaluability grounded in actual evidence | **Met** | Fix 6 tests **[synthetic]** |
| Full suite passes without weakened tests | **Met** | 772 passed; changed expectations itemised in §3 |

## 12. Remaining blockers before M-B

None of the M-A.1 defects blocks starting M-B (earning `WORLD_VALIDATED`: a surveyed calibration with hold-out reprojection checks and reference speeds, `fit_planar_homography` still asserts VALIDATED "by construction"). Recommended before or alongside it:

1. **Decide on a checkpoint** of M-A + M-A.1 (still entirely uncommitted; `demo.mp4`/`traffic.mp4` at the repo root are not gitignored).
2. **The first real `WORLD_VALIDATED` camera will be the first real run of the Fix 1/6 path** — expect the policy parameters in §9.7 to need calibration against labelled data, and add a `geometry_validity` field (or guard) to `TrafficState`.
3. Rebuild the dashboards before any demo.
4. If live/RTSP is in scope: a real-camera outage/reconnect test is still owed.
5. Stop-heuristic jitter (§9.1) before any stopped-vehicle claim.
