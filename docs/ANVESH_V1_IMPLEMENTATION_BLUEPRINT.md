# ANVESH V1 — Implementation Blueprint

**Status:** Design blueprint only. No code, no repository, no dependencies installed. V0 and all prior ANVESH/TRINETRA documents are unmodified.
**Inputs:** V0_AUDIT.md, ANVESH_MIGRATION_ARCHITECTURE.md, ANVESH_V1_MASTER_SPEC.md, ANVESH_V1_RED_TEAM.md, ANVESH_ARCHITECTURE_PROPOSAL.md, TRINETRA_Phase0_Research_Report.md, TRINETRA_Event_Representation_PriorArt_Addendum.md, ANVESH_RESEARCH_GAP_RECONSTRUCTION.md, and the four-agent multi-camera deep-dive completed immediately before this document (traffic/causal literature, shockwave/propagation literature, cross-camera uncertainty-fusion literature, patents + deployed-systems search). No new literature search was performed to produce this document, per instruction — every prior-art claim below traces to one of those existing sources.
**Purpose:** a concrete, execution-ready engineering plan for ANVESH V1. This is the build plan; the research question it tests is frozen in Part 1 and must not drift during implementation.

---

## PART 1 — Freeze the V1 Research Question

### Primary research question

> For a 2-camera, single-corridor, fixed-infrastructure deployment: does **uncertainty-aware cross-camera evidence fusion**, combined with a **propagation-based feedback loop** that revises candidate-cause hypothesis confidence when observed corridor propagation diverges from prediction, produce corridor-level congestion-cause hypotheses that are more accurate and better calibrated than (a) independent single-camera reasoning and (b) naive multi-camera confidence averaging — and does the feedback loop itself add measurable value beyond fusion alone?

This sharpens the user's proposed formulation in one respect, directly justified by the four-agent research: the deep-dive found that "multi-camera fusion" alone (Baseline B/C territory) is the *less* defensible novelty claim — CN118097571A (Dahua), the 1993 Hitachi patent US 5,554,983, ECoMS, and TTC-X all already do some form of multi-camera aggregation for traffic congestion. The one mechanism that came back as a **clean, repeatedly-confirmed null result** across both the literature agent (six independently-phrased queries) and the patent agent (claims-field search) is a **feedback loop where propagation observations revise an earlier cause hypothesis**. That is why the frozen RQ centers the feedback loop as a first-class, independently-tested claim — not folded silently into "fusion."

### Primary hypothesis

**H-primary:** ANVESH (fusion + feedback) achieves higher top-1 candidate-cause accuracy and lower false-confidence rate than Baselines A, B, and C (Part 10), on both the SUMO-simulated and real-recorded evaluation sets, at a pre-registered minimum effect size (Part 11).

### Secondary hypotheses

- **SH1 (shared-uncertainty discount):** explicit shared-vs-independent uncertainty discounting (Part 6) prevents false-confidence inflation specifically under *correlated* (shared-condition) camera degradation, compared to naive averaging (Baseline B) or undiscounted Dempster-Shafer combination — tested via the "independent vs. correlated degradation" ablation (Part 11). This directly targets the specific structural blind spot the uncertainty-fusion agent found: every existing conflict-based Dempster-Shafer method keys on *disagreement*, and does nothing when two co-located cameras *agree* under a shared failure (e.g. both fogged in).
- **SH2 (feedback earns its keep where it matters):** the feedback loop measurably improves top-1 accuracy specifically on scenarios where the *pre-feedback* fusion ranking is wrong — i.e., feedback provides positive value where it is needed, not generic noise.
- **SH3 (evidence-completeness, not just confidence):** weighting a camera's contribution by whether it actually observed the causally-relevant region (evidence-completeness), not merely by its raw detection confidence, improves fusion accuracy on scenarios where one camera is confident about something irrelevant. The patent agent found **zero** patents weighting by observation-relevance as distinct from confidence — this is presently unclaimed anywhere searched.
- **SH4 (honest limitation, not a positive claim):** the kinematic-wave-theory propagation model's location/time predictions degrade measurably on heterogeneous-speed, mixed-vehicle segments (motorcycle/bicycle-heavy) relative to homogeneous-vehicle segments. This is expected to be *confirmed* as a limitation, not refuted — it is carried from the earlier Research Gap Reconstruction's finding that a single fixed speed threshold across vehicle classes is methodologically naive, and it must be measured and reported, not assumed away.

### Independent variables
Fusion method (Baseline A / B / C / ANVESH — the four-level ladder in Part 10); camera degradation condition (clean / independently-degraded / shared-degraded); vehicle-mix homogeneity (for SH4); presence/absence of the feedback loop.

### Dependent variables
Top-1 / top-k candidate-cause accuracy; internal calibration (Brier score / ECE, computed but not necessarily displayed numerically — see Part 6, Part 11); false-confidence rate; propagation-location error; propagation-time error; hypothesis-revision accuracy; congestion-detection precision/recall/F1.

### Baselines
Defined precisely in Part 10 — Baseline A (single-camera), Baseline B (naive averaging), Baseline C (fusion, no feedback), ANVESH (fusion + feedback).

### Success criteria
Adopting the same pre-registration discipline already established in the (frozen, unmodified) Master Spec's kill-criteria section — **defined before any run, not adjusted after seeing results**:
- H-primary succeeds only if ANVESH beats Baseline C on top-1 accuracy by a **pre-registered minimum absolute margin (proposed: ≥10 percentage points)** on the SUMO set, *and* the direction (not necessarily the same margin) replicates on the real-recorded set.
- If ANVESH does not beat Baseline C by the pre-registered margin, the feedback loop's added complexity is **not justified by this project's own evidence** and must be reported as such — this is a legitimate, plannable negative result, not a failure of the blueprint.
- SH1–SH3 each have their own pass/fail condition, defined in Part 11's ablation table, evaluated and reported together regardless of individual outcome.
- SH4 has no "pass" condition — it is a measurement, reported honestly either way.

---

## PART 2 — V1 Scope

| Tier | Item |
|---|---|
| **MUST HAVE** | 2 fixed cameras, 1 corridor stretch (non-overlapping or minimally-overlapping FOV, known spatial ordering); recorded/replayed video (not live feeds) as the primary input mode; SUMO-simulated corridor as the first, fastest ground-truth track; the 7-hypothesis cause set (H1–H7, Part 7); Dempster-Shafer fusion with reliability + shared-uncertainty discounting (Part 6); kinematic-wave/shockwave propagation model (Part 8); the feedback/update engine (Part 9); Baseline A/B/C/ANVESH evaluation harness (Part 10–11); SQLite storage; a minimal read-mostly dashboard (3 views, matching the frozen Master Spec's Part 13); the schemas in Part 5, frozen on Day 1 |
| **SHOULD HAVE** | A small set of real recorded + staged-scenario clips (beyond the SUMO track) for external validity; a simple config-driven ExperimentRun tracker; Docker packaging for reproducibility across team machines; a lightweight React/Next.js frontend if a team member has the skill, otherwise Jinja2/HTML is sufficient |
| **LATER (V1.5/V2, not built now)** | 3+ cameras / a real road-segment graph; live camera feeds; a learned/ML propagation model (GNN/DBGCN-style) as an alternative to the kinematic-wave model; full multi-camera correlated-error modeling via copula-based decision fusion (the mathematically "correct" discrete analogue the uncertainty-fusion agent identified, but too heavy for V1 — see Part 6); PostgreSQL if SQLite's concurrency limits are actually hit; RBAC beyond authentication; automatic re-generation of new hypotheses when all existing ones are contradicted (V1 just surfaces H7 honestly instead) |
| **OUT OF SCOPE** | Live government deployment of any kind; safety-event / accident detection as a validated capability (schema-reserved only, per the frozen Master Spec's Part 11); any autonomous action; license plate recognition; government interoperability export; mobile apps; auto-scaling infrastructure; city-scale or multi-corridor systems |

The first working milestone (M1–M2) uses exactly 2 cameras, 1 corridor, a small number of traffic states (free-flow / building / congested / dissipating), and the 7-hypothesis set — never a city-scale ambition.

---

## PART 3 — System Architecture

```
                        ┌────────────────────────────────────────────┐
                        │              CAMERA 1 (upstream)             │
                        │              CAMERA 2 (downstream)           │
                        └───────────────────┬──────────────────────────┘
                                            │ recorded / replayed video
                     ┌──────────────────────┴──────────────────────┐
                     │  1. VIDEO INGESTION  (per camera)             │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  2. CAMERA CALIBRATION / METADATA             │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  3. VEHICLE/OBJECT DETECTION                  │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  4. TRACKING                                  │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  5. TRAFFIC-STATE EXTRACTION (per camera)     │
                     └──────────────────────┬──────────────────────┘
                                            │  per-camera streams (x2)
                     ┌──────────────────────┴──────────────────────┐
                     │  6. TEMPORAL ALIGNMENT (across cameras)       │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  7. SPATIAL / CORRIDOR REPRESENTATION         │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  8. PER-CAMERA EVIDENCE GENERATION            │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │  9. RELIABILITY ESTIMATION                    │
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │ 10. CROSS-CAMERA FUSION  (DS + discounting)   │◄─────┐
                     └──────────────────────┬──────────────────────┘      │
                     ┌──────────────────────┴──────────────────────┐      │
                     │ 11. CORRIDOR-STATE ESTIMATION                 │      │
                     └──────────────────────┬──────────────────────┘      │
                     ┌──────────────────────┴──────────────────────┐      │
                     │ 12. CAUSE HYPOTHESIS ENGINE  (H1–H7)          │      │
                     └──────────────────────┬──────────────────────┘      │
                     ┌──────────────────────┴──────────────────────┐      │
                     │ 13. PROPAGATION MODEL (shockwave/kinematic)   │      │
                     └──────────────────────┬──────────────────────┘      │
                                            │ PropagationPrediction        │
                                            ▼                              │
                                     (wait Δt for new observations)        │
                                            │                              │
                                            ▼                              │
                     ┌───────────────────────────────────────────────┐    │
                     │ 14. FEEDBACK / UPDATE ENGINE                   │────┘
                     │     predicted vs. observed → HypothesisUpdate  │
                     └──────────────────────┬───────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │ 15. EXPLANATION / OUTPUT LAYER (dashboard/API)│
                     └──────────────────────┬──────────────────────┘
                     ┌──────────────────────┴──────────────────────┐
                     │ 16. EVALUATION / LOGGING (cross-cutting)      │
                     └───────────────────────────────────────────────┘
```

Modules 1–5 run **once per camera, independently**. Modules 6 onward operate on the *combined* two-camera stream. Module 16 (evaluation/logging) is cross-cutting — it observes every other module's output, it is not a pipeline stage.

| # | Module | Responsibility | Inputs | Outputs | Recommended implementation | V1 / Later |
|---|---|---|---|---|---|---|
| 1 | Video ingestion | Read frames from a recorded file per camera, timestamped | video file path, camera_id | timestamped frames | OpenCV `VideoCapture`, explicit open-check (fixes V0's silent failure, V0_AUDIT §13) | **V1** |
| 2 | Camera calibration/metadata | Hold the documented calibration procedure's output per camera | one-time survey/calibration run | `CalibrationProfile` (Part 5) | Manual 4-point homography + measured error bound, versioned config file | **V1** |
| 3 | Vehicle/object detection | Per-frame detection with class + confidence | frame | `Detection[]` | Ultralytics YOLO, pretrained (fine-tuning is later) | **V1** |
| 4 | Tracking | Frame-to-frame association, per-camera track IDs | `Detection[]` | `VehicleTrack` updates | ByteTrack (Ultralytics built-in) | **V1** |
| 5 | Traffic-state extraction | Per-camera occupancy/speed/count over a segment window | `VehicleTrack` stream, `CalibrationProfile` | `TrafficState` (per camera) | World-projected speed via calibrated homography, wall-clock windowing (fixes V0_AUDIT §6) | **V1** |
| 6 | Temporal alignment | Synchronize the two cameras' timestamps to a shared clock | two `TrafficState` streams | aligned, jointly-timestamped state pairs | Recording-time NTP/device-clock sync + a one-time manual landmark-event check (Part 16 risk mitigation) | **V1** |
| 7 | Spatial/corridor representation | Define the corridor's segment ordering and known inter-camera distance | operator-defined topology | `CorridorState` topology (static config, not learned) | A single ordered list of 2 segments with a known distance/expected free-flow travel time between them | **V1** |
| 8 | Per-camera evidence generation | Turn `TrafficState` into evidence supporting/contradicting each H1–H7 | `TrafficState`, `VehicleTrack` | `Evidence` records | Rule-based signature scoring against Part 7's per-hypothesis signatures | **V1** |
| 9 | Reliability estimation | Score each camera's trustworthiness for this observation window | calibration error, occlusion state, evidence-completeness | per-camera reliability `r_i` | A documented weighted combination of calibration staleness + occlusion fraction + evidence-completeness flag (Part 6) | **V1** |
| 10 | Cross-camera fusion | Combine per-camera `Evidence`/hypothesis mass into one joint mass | `Evidence` × 2, reliability | fused Dempster-Shafer mass function | Reliability-discounted Dempster's rule + shared-uncertainty discount (Part 6, equations given) | **V1** |
| 11 | Corridor-state estimation | Assemble the fused mass + raw states into one corridor-level congestion state | fused mass, `TrafficState` × 2 | `CorridorState` | Direct assembly — no new modeling, mostly bookkeeping | **V1** |
| 12 | Cause hypothesis engine | Rank H1–H7 by fused belief/plausibility | fused mass, `CorridorState` | `CandidateCauseHypothesis[]` ranked | Bel/Pl extraction from the DS mass, mapped to categorical confidence tiers (Part 6, Part 10 honesty rule) | **V1** |
| 13 | Propagation model | Predict where/when the disturbance's effect will next be observable | top hypothesis, `CorridorState` flow/density | `PropagationPrediction` | Rankine-Hugoniot shockwave-speed estimate from kinematic wave theory (Part 8) | **V1** |
| 14 | Feedback/update engine | Compare predicted vs. observed propagation; revise hypothesis ranking | `PropagationPrediction`, next-window `CorridorState` | `HypothesisUpdate`, revised ranking | Explicit 4-branch logic: match / partial / contradict / missing-evidence (Part 9 pseudocode) | **V1** |
| 15 | Explanation/output layer | Present the ranking, evidence, and confidence to a human | all above | dashboard views, API responses | FastAPI + minimal templates (Part 4) | **V1** (3-view minimum, per frozen Master Spec) |
| 16 | Evaluation/logging | Compute Part 11's metrics; run the A/B/C/ANVESH ablation; log `ExperimentRun`s | all module outputs + ground truth | metric reports, `EvaluationRecord` | A dedicated `evaluation/` package, run offline against stored data | **V1** |

Everything above is deliberately a single Python process with in-process function calls between modules — no event bus, no microservices, no distributed deployment, matching the architectural discipline already frozen in the (unmodified) Master Spec.

---

## PART 4 — Tech Stack

| Choice | One-sentence justification |
|---|---|
| **Python 3.11+** (single language, whole project) | One language for perception, fusion, propagation, storage, and API removes context-switching for a small student team and matches every dependency below. |
| **OpenCV** | Standard, well-documented video I/O and frame handling; no reason to use anything else for reading recorded video. |
| **Ultralytics YOLO** (v8 or later stable) | Same family V0 used, well-documented, easy for students, and keeps the V0-vs-ANVESH comparison honest since detection isn't where the contribution lives (per the frozen Master Spec's own finding that detection comparisons to V0 aren't meaningful). |
| **ByteTrack** (via Ultralytics' built-in tracker) | Simplest tracker that's good enough for V1; BoT-SORT/appearance-based re-ID is explicitly deferred (Part 2, LATER) since V1 doesn't need cross-camera *object* re-identification, only cross-camera *evidence* fusion. |
| **NumPy / Pandas** | Standard, minimal-friction tools for the evidence/state feature engineering and the shockwave-equation arithmetic in Part 8. |
| **A hand-rolled Dempster-Shafer module (≤7-hypothesis frame, no external DS library required)** | With only 7 singleton hypotheses plus ignorance, the combination rule is a few dozen lines of arithmetic — a full belief-function library is unnecessary complexity for this scale. |
| **PyTorch** — present only as Ultralytics' own dependency | Not a V1 project dependency in its own right; V1's propagation model is classical (Part 8), not learned, so no direct PyTorch usage is required beyond what YOLO already pulls in. |
| **SQLite** | One process, one small dataset, no concurrent-writer problem at V1 scale — a real database server is unjustified overhead (matches the frozen Master Spec's "one database" collapse decision). |
| **FastAPI** | Lightweight, Python-native, minimal boilerplate for the read-mostly API the dashboard needs. |
| **Jinja2/HTML templates as the MUST-HAVE frontend; React/Next.js as a SHOULD-HAVE upgrade** | The 3-view dashboard (Part 13 of the frozen Master Spec) doesn't need a SPA to be useful; a templated page is buildable by anyone on the team, while React is a nice-to-have only if someone already has the skill and time. |
| **TypeScript** | Only relevant if the React upgrade is pursued; skipped entirely otherwise. |
| **Docker** — SHOULD HAVE, not MUST | Valuable for reproducibility across team machines and for the eventual demo, but a pinned `requirements.txt`/`pyproject.toml` is sufficient to start; don't let container plumbing delay M1. |
| **SUMO** (Eclipse SUMO) | The standard open-source microscopic traffic simulator, already used by the closest comparable prior-art systems found in research (IncidentNet, TTC-X) for exactly this kind of controlled-ground-truth validation. |
| **pytest** | Standard Python test runner, used for every module's contract tests per Part 3's module table. |

Deliberately excluded from V1: any message broker/event bus, Kubernetes or other orchestration, a second programming language for the frontend unless the React upgrade is actually pursued, and any deep-learning propagation model (GNN/DBGCN-class) — all explicitly LATER per Part 2.

---

## PART 5 — Data Model

Conceptual, implementation-ready schemas. Every record includes `schema_version` and relevant timestamps (omitted from the tables below for brevity, but required on every entity, per the discipline already frozen in the Master Spec).

**Camera**
| Field | Type | Meaning |
|---|---|---|
| camera_id | string | stable identifier |
| position_description | string | human-readable site description |
| role | enum{upstream, downstream} | this camera's position in the 2-camera corridor |
| calibration_profile_id | string (FK) | active `CalibrationProfile` |

**CameraObservation**
| Field | Type | Meaning |
|---|---|---|
| observation_id | string | |
| camera_id | string (FK) | |
| frame_timestamp | float (unix, UTC) | |
| detections | Detection[] | camera-space detections for this frame |
| model_id, model_version | string | provenance |

**VehicleTrack**
| Field | Type | Meaning |
|---|---|---|
| track_id | string | camera_id-scoped |
| camera_id | string (FK) | |
| first_seen, last_seen | float | |
| class | enum | vehicle class (car/motorcycle/auto-rickshaw/bus/truck/bicycle) |
| occlusion_state | enum{visible, partial, lost} | explicit, never silently interpolated |
| position_history | (world_x, world_y, timestamp)[] | via `CalibrationProfile` projection |
| speed_estimate | float ± error | wall-clock windowed (fixes V0_AUDIT §6) |

**TrafficState** (per camera, per time window)
| Field | Type | Meaning |
|---|---|---|
| camera_id | string (FK) | |
| window_start, window_end | float | wall-clock window |
| occupancy | float [0,1] | fraction of segment occupied |
| mean_speed | float ± error | |
| vehicle_count | int | |
| flow_rate | float | vehicles/time |
| density | float | vehicles/length |
| congestion_level | enum{free_flow, building, congested, dissipating} | derived, hysteresis-damped |

**Evidence** (per camera, per hypothesis, per window)
| Field | Type | Meaning |
|---|---|---|
| evidence_id | string | |
| camera_id | string (FK) | |
| window | (start, end) | |
| hypothesis_id | string (FK → H1–H7) | which hypothesis this evidence bears on |
| signature_match_score | float [0,1] | how well this window's observations match the hypothesis's expected signature (Part 7) |
| supporting_track_refs | track_id[] | traceable, never inline-only |
| contradicting | bool | does this evidence contradict the hypothesis |
| evidence_completeness | enum{full, partial, none} | did this camera actually observe the causally-relevant region — distinct from confidence (SH3) |

**CauseHypothesis** (definition record, static per H1–H7, not per-incident)
| Field | Type | Meaning |
|---|---|---|
| hypothesis_id | string | H1..H7 |
| name | string | e.g. "downstream_bottleneck" |
| expected_signature | structured (Part 7) | onset pattern, propagation direction, persistence expectation |
| contradiction_rules | structured (Part 7) | what evidence rules this hypothesis out |

**CandidateCauseHypothesis** (per-incident ranked output — not to be confused with the static `CauseHypothesis` definitions above)
| Field | Type | Meaning |
|---|---|---|
| ranking_id | string | |
| corridor_state_id | string (FK) | |
| outcome | enum{ranked, insufficient_evidence} | never forced |
| ranked_list | [{hypothesis_id, belief, plausibility, confidence_tier, supporting_evidence_refs[], contradicting_evidence_refs[]}] | Bel/Pl interval + categorical tier (Part 6, Part 10) |
| engine_model_id, version | string | provenance |

**PropagationPrediction**
| Field | Type | Meaning |
|---|---|---|
| prediction_id | string | |
| based_on_ranking_id | string (FK) | which `CandidateCauseHypothesis` ranking this predicts from |
| predicted_shockwave_speed | float ± error | m/s, signed (upstream negative, downstream positive) |
| predicted_arrival_camera | camera_id | which camera should next observe the effect |
| predicted_arrival_time | float | |
| predicted_queue_growth_rate | float | vehicles/min |
| method | string | "kinematic_wave_v1" |

**PropagationObservation**
| Field | Type | Meaning |
|---|---|---|
| observation_id | string | |
| prediction_id | string (FK) | which prediction this observation is being compared against |
| actual_arrival_camera | camera_id | |
| actual_arrival_time | float | |
| actual_queue_growth_rate | float | |
| data_completeness | enum{full, partial, missing} | |

**HypothesisUpdate**
| Field | Type | Meaning |
|---|---|---|
| update_id | string | |
| prior_ranking_id | string (FK) | |
| propagation_observation_id | string (FK) | |
| outcome | enum{confirmed, partial, contradicted, evidence_missing} | Part 9's four branches |
| revised_ranking_id | string (FK) | the new `CandidateCauseHypothesis` produced |
| discrepancy | {location_error, time_error, rate_error} | |

**CorridorState**
| Field | Type | Meaning |
|---|---|---|
| corridor_state_id | string | |
| window | (start, end) | |
| segment_states | TrafficState[] | one per camera/segment |
| fused_congestion_level | enum | corridor-wide, from fusion |
| active_ranking_id | string (FK) | current `CandidateCauseHypothesis` |

**EvaluationRecord**
| Field | Type | Meaning |
|---|---|---|
| record_id | string | |
| experiment_run_id | string (FK) | |
| baseline | enum{A, B, C, ANVESH} | |
| scenario_id | string | which SUMO/real scenario |
| metrics | {top1_accuracy, topk_accuracy, brier_score, ece, false_confidence_rate, propagation_location_error, propagation_time_error, hypothesis_revision_accuracy, congestion_p, congestion_r, congestion_f1} | Part 11 |
| ground_truth_ref | string | scenario's known injected/adjudicated cause |

---

## PART 6 — Fusion Model

### Options considered

| Method | Handles conflicting sources? | Handles per-source reliability? | Handles correlated/shared errors? | Buildable by a student team in V1? |
|---|---|---|---|---|
| Weighted confidence averaging | No — averages away conflict rather than resolving it | Yes, trivially (weighted mean) | No | Yes — this is exactly Baseline B, kept as a control, not the ANVESH method |
| Classical Bayesian fusion (independent likelihoods) | Partially | Yes | No — assumes independence by construction | Yes, but inherits the same "agreeing-under-shared-fog" blind spot the uncertainty-fusion agent found in every reviewed method |
| Dempster-Shafer combination (with reliability discounting) | Yes — conflict mass is explicit and inspectable | Yes, via classical Shafer discounting | Not by default, but *contextual discounting* (Mercier/Quost/Denoeux 2008) gives a documented, implementable path to add it | **Yes — chosen for V1** |
| Split/Inverse Covariance Intersection (SCI/ICI) | Conservative under unknown correlation | Yes | Yes — this is literally what SCI/ICI were built for | No — every application found (Cros et al. 2025, Noack et al. 2017) operates on continuous Gaussian state estimates, not discrete confidence-scored hypotheses; porting it requires a target-type bridge the research found nobody has built |
| Copula-based decision fusion (Sundaresan & Varshney 2011; vine copulas) | Yes, and is the mathematically *correct* discrete analogue of SCI/ICI | Yes | Yes, explicitly | Not for V1 — vine-copula fitting is a research-grade undertaking, not a week-one implementation task |

### V1 choice: reliability-discounted Dempster-Shafer combination with an explicit shared-uncertainty discount

This is **not presented as new mathematics**. Classical Dempster's rule and Shafer discounting are textbook; contextual discounting (source reliability conditional on the hypothesis) is Mercier/Quost/Denoeux (2008); the shared-vs-independent split is directly inspired by SCI/ICI's decomposition, adapted here as a pragmatic discrete-domain approximation rather than a rigorous port. What's specific to ANVESH is the instantiation: applying this combination to fixed-camera candidate-cause evidence with an explicit, logged shared-condition flag.

**Frame of discernment:** Θ = {H1, H2, H3, H4, H5, H6} (the six substantive causes). H7 ("insufficient evidence") is represented as the ignorance mass on Θ itself, not as a seventh singleton — this maps the abstention discipline already frozen in the Master Spec directly onto the DS formalism.

**Step 1 — per-camera basic probability assignment (BPA).** Camera *i*'s evidence generation module (Part 3, module 8) produces `m_i(Hk)` for k=1..6 and `m_i(Θ)` (residual ignorance), summing to 1.

**Step 2 — reliability discounting (classical Shafer discounting).** Let `r_i ∈ [0,1]` be camera *i*'s reliability for this window (Part 3, module 9 — derived from calibration staleness, occlusion fraction, and evidence-completeness):

```
m_i'(Hk) = r_i * m_i(Hk)                    for k = 1..6
m_i'(Θ)  = 1 - r_i * (1 - m_i(Θ))
```

**Step 3 — shared-uncertainty discount (before combination, when a shared-condition flag is active).** If cameras *i* and *j* share a logged site-level condition (same-timestamp adverse weather/lighting flag), combine their discounted BPAs two ways and blend:

```
m_DS      = DempsterCombine(m_i', m_j')                     # standard Dempster's rule, below
m_conserv = ElementwiseMax(m_i', m_j')  (per hypothesis)     # conservative fallback: never award more
                                                              # combined confidence than the more confident
                                                              # single reliable camera already had

λ_shared = estimated_independent_fraction ∈ [0,1]            # 1.0 if no shared-condition flag is active;
                                                              # lower (e.g. 0.3–0.5) when the flag is active,
                                                              # set as a documented, versioned config constant
                                                              # in V1 — not learned — pending SH1's ablation

m_final = λ_shared * m_DS + (1 - λ_shared) * m_conserv
```

This is the concrete mechanism SH1 tests: does this blend correctly prevent false-confidence inflation under shared degradation, compared to Baseline B (plain averaging, no discount at all) and an undiscounted DS combination (λ_shared fixed at 1.0)?

**Dempster's combination rule** (standard, applied to the two reliability-discounted BPAs):

```
K = Σ_{Hp ∩ Hq = ∅} m_i'(Hp) * m_j'(Hq)          # conflict mass (here, all distinct singletons conflict)
m_DS(Hk) = [ m_i'(Hk)*m_j'(Hk) + m_i'(Hk)*m_j'(Θ) + m_i'(Θ)*m_j'(Hk) ] / (1 - K)
m_DS(Θ)  = [ m_i'(Θ) * m_j'(Θ) ] / (1 - K)
```

**Known limitation, documented rather than hidden:** vanilla Dempster's rule is known to misbehave when K (conflict) is very high — this is a documented, decades-old critique of the classical rule. V1 logs K explicitly per fusion event; if K exceeds a pre-registered threshold, the fusion output is flagged as "high camera disagreement" and surfaced to the reviewer distinctly from low confidence (per Part 9's explicit handling of the "cameras disagree" case), rather than silently trusting a renormalized-but-unreliable result. Switching to Yager's rule or an unnormalized combination is a documented, deferred extension point if this proves to matter in practice — not built speculatively now.

**Output:** for each hypothesis, `Bel(Hk) = m_final(Hk)` and `Pl(Hk) = m_final(Hk) + m_final(Θ)` — reported as an interval, mapped to a categorical confidence tier (low/medium/high) by interval width and Bel value, per the frozen Master Spec's rule that a number is only displayed numerically once calibration-justified (Part 6/Part 11 track this internally as continuous values for metric computation; the **displayed** value stays categorical in V1).

---

## PART 7 — Cause Model

Frame of discernment: H1–H6 substantive causes, H7 the explicit abstention outcome (mapped to DS ignorance mass, Part 6).

| Hypothesis | Observable evidence | Expected propagation signature | Confidence update logic | Contradictions | What cameras should observe if true |
|---|---|---|---|---|---|
| **H1 — Downstream bottleneck** | Sustained high occupancy at a *fixed* downstream point; queue tail growing upstream; no stationary/blocking object within the queue | Queue forms backward (upstream) from a fixed capacity-limited point at a roughly constant shockwave speed; onset gradual, correlated with rising demand | Rises if downstream camera shows an unchanging, capacity-limited flow point over repeated windows; rises further if dissipation matches predicted shockwave behavior as demand falls | A moving origin point, or a stationary blocking object found in the queue | Downstream camera: consistent capacity-limited flow at a fixed point. Upstream camera: backward-growing queue |
| **H2 — Stopped/disabled vehicle** | A specific tracked vehicle at near-zero speed, sustained, *not* at a marked stop point | Queue originates from a fixed *point* coincident with the stopped vehicle; abrupt onset; dissipates from that point once the vehicle moves again | Rises when the stopped vehicle is confirmed at the queue head in a later frame/camera, and again if dissipation coincides with that vehicle resuming motion | The "stopped" vehicle is actually moving slowly (misclassification), or is at a legitimate signal stop | Origin camera: a stationary tracked object. Downstream of it: free flow |
| **H3 — Collision/incident-like obstruction** | Multiple vehicles abnormally clustered/stopped, possibly abnormal orientation; abrupt onset; possible pedestrian-on-carriageway signal | Similar to H2 but higher severity, longer queue-growth rate, longer persistence, possible full (not partial) lane blockage | Rises with visual clustering cues and abrupt onset relative to the demand trend | Orderly single-vehicle stop, or flow continuing around the location | Origin camera: abnormal multi-vehicle cluster. Up/downstream: stronger version of H2's signature |
| **H4 — Lane blockage (non-incident)** | A fixed-position obstruction persisting far longer than a plausible disabled-vehicle clearance time, or a non-vehicle-classified static object; repeated lane-change/swerve behavior at that point | Partial-capacity reduction at a fixed point, lower severity than H2/H3, much longer persistence | Rises with repeated swerve/lane-change detections clustered at one fixed point and abnormally long persistence | The object moves/disappears quickly (favors H2/H3), or all lanes are equally affected (favors H1) | Camera at the obstruction: consistent lane-change clustering at a fixed point |
| **H5 — Upstream demand surge** | No stopped/obstructing object anywhere in the corridor; congestion explained by elevated inflow from upstream/a side entry | Queue grows from the downstream-most capacity point, but growth rate correlates with a *measured* upstream inflow-rate increase, not a new obstruction | Rises if the upstream camera shows a measurable count/inflow increase preceding congestion onset, with no obstruction evidence anywhere | Any stopped vehicle or lane blockage found in the corridor; no consistent slow-down under normal demand at the downstream point | Upstream camera is most informative: elevated counts/density before downstream congestion appears |
| **H6 — Signal/intersection restriction (abnormal)** | Congestion originates at/upstream of a known signalized point; queue length oscillates with signal phase in a pattern that deviates from the historical baseline for that signal | Cyclical (not monotonic) growth/dissipation tied to signal phase; abnormal if amplitude/period deviates from expectation | Rises when oscillation deviates from the historical baseline pattern (requires stored `CorridorState` history — a direct reuse of the "post-incident analysis" government-value item from the earlier research documents) | Oscillation matches the known-normal signal timing (not flaggable), or queue persists through green phases (favors a downstream cause) | Camera covering the intersection approach; V1 infers phase purely from cyclical queue behavior, no direct signal-controller feed (documented limitation) |
| **H7 — Insufficient evidence** | None of H1–H6's signatures are sufficiently supported, or the corridor's 2 cameras do not cover the plausibly-relevant area | N/A by definition | Default/floor state — other hypotheses' mass must clear an evidentiary bar before H7's (ignorance) mass shrinks | Any of H1–H6 gaining strong support | N/A — this is the honest "we don't/can't know" state, directly implementing the frozen Master Spec's `insufficient_evidence` contract |

---

## PART 8 — Propagation

"Propagation," in ANVESH V1, means a small set of measurable quantities computed directly from `CorridorState` history — never left as a vague notion:

- **Congestion onset time:** first window timestamp at which a segment's `congestion_level` crosses from `free_flow`/`building` into `congested`.
- **Congestion location:** the segment/camera id, plus estimated position within that camera's FOV via the calibrated world projection.
- **Queue growth rate:** `d(queue_length)/dt`, in vehicles or meters per second, over consecutive windows.
- **Upstream/downstream movement:** the sign of `d(queue_front_position)/dt` — negative (moving toward the upstream camera) = growing; positive (moving toward the downstream camera) = dissipating.
- **Propagation velocity (shockwave speed):** the classical Rankine-Hugoniot kinematic-wave estimate,
  ```
  v_shock = (q2 - q1) / (k2 - k1)
  ```
  where `q1, k1` are flow and density just upstream of the queue front and `q2, k2` just downstream — a well-established, decades-old result from LWR/kinematic-wave traffic-flow theory, not a new formula.
- **Affected camera sequence:** the ordered list of cameras that have observed the disturbance's effect over time (in V1, at most `[downstream, upstream]` or `[upstream, downstream]`).
- **Persistence:** duration the corridor's `congested` state remains above threshold.
- **Dissipation:** the time/location at which the state returns below threshold.

### V1 method: rule-based + classical traffic-flow theory, not ML

Per "prefer the simplest scientifically defensible option": V1's `PropagationPrediction` (Part 5) is generated by plugging the top-ranked hypothesis's expected signature (Part 7) and the current `CorridorState`'s flow/density values into the shockwave-speed formula above, producing a predicted arrival camera, arrival time, and queue-growth rate. This requires no training data, is fully interpretable, and is directly citable to established transportation-engineering theory rather than an unvalidated learned model. A GNN/DBGCN-style learned propagation model (matching the "saturated, competitive" literature the shockwave-research agent surveyed) is explicitly **LATER** (Part 2) — it may outperform the classical model once more corridor-scale data exists, but V1 does not need it to test the frozen RQ, and building it now would spend the team's limited time on a component that is not where the claimed contribution lives.

---

## PART 9 — Feedback Loop

```
function feedback_loop(corridor_state_t0, ranking_t0):
    prediction = predict_propagation(ranking_t0.top_hypothesis, corridor_state_t0)   # Part 8

    wait(Δt)                                             # fixed, pre-registered observation window

    corridor_state_t1 = extract_corridor_state(new_observations)
    observation = extract_propagation_observation(corridor_state_t1, prediction)     # Part 5

    discrepancy = {
        location_error: distance(prediction.arrival_camera, observation.actual_arrival_camera),
        time_error:     prediction.arrival_time - observation.actual_arrival_time,
        rate_error:     prediction.queue_growth_rate - observation.actual_queue_growth_rate,
    }

    if observation.data_completeness != "full":
        # MISSING EVIDENCE: never treated as confirmation or contradiction
        emit HypothesisUpdate(outcome = "evidence_missing", discrepancy = None)
        return ranking_t0   # unchanged; evidence_completeness flag set LOW for this cycle

    for H in ranking_t0.ranked_list:
        match_score = score_signature_match(H.expected_signature, observation)
        if match_score == STRONG_MATCH:
            H.mass = boost(H.mass)          # re-run Part 6 fusion with this window's evidence reinforcing H
        elif match_score == PARTIAL_MATCH:
            H.mass = slight_boost_or_unchanged(H.mass)
        elif match_score == STRONG_CONTRADICTION:
            H.mass = penalize(H.mass)       # increase discount toward Θ for this hypothesis's evidence source

    revised_ranking = rerank(ranking_t0)     # re-extract Bel/Pl per Part 6 after the mass updates above

    if all(H.mass ≈ 0 for H in revised_ranking) and revised_ranking.top_confidence < H7_threshold:
        revised_ranking.outcome = "insufficient_evidence"   # promote H7 honestly; V1 does NOT auto-generate
                                                              # a new hypothesis outside H1-H7 (LATER, Part 2)

    outcome = "confirmed" if all H matched strongly else
              "contradicted" if any top-hypothesis strongly contradicted else
              "partial"

    emit HypothesisUpdate(prior_ranking_id = ranking_t0.id,
                           propagation_observation_id = observation.id,
                           outcome = outcome,
                           revised_ranking_id = revised_ranking.id,
                           discrepancy = discrepancy)

    return revised_ranking
```

**Explicit behavior per case, as required:**
- **Prediction matches:** confidence boosted for the matching hypothesis; `HypothesisUpdate.outcome = "confirmed"`; ranking reinforced, not just left alone.
- **Prediction partially matches:** modest boost or no change; flagged `"partial"`; deliberately does **not** trigger a sharp re-ranking off one ambiguous window, to avoid overreacting to noise — a wider observation window is used before the next update is trusted.
- **Prediction strongly contradicts:** the corresponding hypothesis's mass is penalized sharply; if *all* hypotheses are contradicted, H7 (`insufficient_evidence`) is promoted rather than the system inventing a new, unlisted cause.
- **Evidence is missing:** no confidence change in either direction — explicitly distinct from contradiction, per the frozen Master Spec's "missing evidence ≠ low confidence" honesty rule; `evidence_completeness` is set LOW and logged for that cycle.
- **Cameras disagree:** handled upstream, at the fusion step (Part 6) — a high conflict mass `K` is logged and surfaced as "high camera disagreement," distinct from low confidence, never silently resolved by picking a side.

---

## PART 10 — Baselines

| Baseline | What it does |
|---|---|
| **A — Single-camera independent reasoning** | Runs the full per-camera pipeline (modules 1–5, 8–9, 12–13) using **only the downstream camera**, with no fusion (module 10 skipped — its own `Evidence` is passed straight through as the "fused" mass), no propagation feedback (module 14 skipped). This is the retired single-camera RQ4, rebuilt as a control — it directly and empirically re-tests why the original research question was insufficient, rather than only arguing it narratively. |
| **B — Naive multi-camera confidence averaging** | Both cameras run independently (as in A) producing per-camera hypothesis masses; combined by simple arithmetic mean per hypothesis, renormalized. No Dempster combination rule (no conflict handling), no reliability weighting, no shared-uncertainty discount, no feedback. Isolates whether "more data, naively combined" helps at all. |
| **C — Multi-camera fusion, no feedback** | Full Part 6 fusion (reliability discounting + shared-uncertainty discount + Dempster combination), producing a ranked hypothesis list — but module 14 (feedback) is skipped entirely; the initial ranking is never revised. Isolates whether the fusion sophistication alone (without feedback) is where the value is. |
| **ANVESH — full system** | Baseline C plus the Part 9 feedback loop. The complete, frozen system under test. |

This four-level ladder is designed so H-primary, SH1, and SH2 (Part 1) are each answerable from a single, consistent set of runs — A vs. B isolates fusion-vs-no-fusion, B vs. C isolates naive-vs-uncertainty-aware fusion, and C vs. ANVESH isolates feedback's marginal value.

---

## PART 11 — Evaluation

### Metrics
Congestion-detection precision/recall/F1; cause-classification accuracy/F1; top-1 and top-2 cause accuracy; calibration (Brier score and ECE, computed internally against the categorical-tier-to-Bel/Pl mapping — per the frozen Master Spec's rule, these are validation metrics, not necessarily user-displayed numbers until they justify numeric display); propagation-location error; propagation-time error; hypothesis-revision accuracy (does a `confirmed`/`contradicted` `HypothesisUpdate` correctly track the scenario's true cause persisting or changing); false-confidence rate (proportion of high-confidence-tier outputs that were wrong — the single most important honesty metric); robustness deltas under camera degradation and under camera conflict.

### Ablations (each maps directly to Part 1's hypotheses)

| Ablation | Compares | Tests |
|---|---|---|
| 1 camera vs. 2 cameras | Baseline A vs. B/C/ANVESH | Does adding a second camera help at all |
| Naive fusion vs. uncertainty-aware fusion | Baseline B vs. C | Does the DS combination + discounting add value over simple averaging |
| Without feedback vs. with feedback | Baseline C vs. ANVESH | H-primary, SH2 |
| Clean video vs. degraded video | All baselines, clean vs. synthetically degraded (blur/noise/dropout) input | General robustness |
| **Independent errors vs. correlated/shared degradation** | All baselines, one camera degraded (independent) vs. both cameras degraded simultaneously by the same injected condition (shared) | **SH1 — the core novel-mechanism test.** Expected result: Baseline B and undiscounted DS *inflate* confidence when both cameras agree under shared degradation; ANVESH's shared-uncertainty discount should suppress that inflation, measurably lowering the false-confidence rate specifically in the shared-degradation condition relative to the independent-degradation condition |
| Homogeneous vs. heterogeneous vehicle mix | Segments with mostly cars/trucks vs. motorcycle/bicycle-heavy segments | SH4 — expected to confirm degraded propagation-error, an honest limitation, not a win |

Every ablation is run against **both** the SUMO track (exact ground truth) and the real-recorded track (adjudicated ground truth, Part 12), and every run produces a `EvaluationRecord`/`ExperimentRun` pair — no number is reported without one, per the discipline already frozen in the Master Spec.

---

## PART 12 — Dataset Strategy

**Two parallel tracks, not one, so the project is never blocked on real-data logistics:**

**Track 1 — SUMO microsimulation (first, fastest, exact ground truth).** A small 2–3 segment corridor network in SUMO, with scripted congestion episodes injecting each of H1–H6 (several replicates per hypothesis), each episode logged with its exact injected cause, onset time/location, and true propagation trace. This validates the **fusion + cause + propagation + feedback reasoning core in isolation**, decoupled from real-world perception noise — directly mirroring how IncidentNet and TTC-X (both found in the prior research) validated their reasoning components primarily via SUMO. **Minimum for M1:** ~20–30 simulated episodes covering H1–H6 with several replicates each, plus a handful of H7 (ambiguous/no-cause) control episodes.

**Track 2 — real recorded + staged 2-camera footage (slower, necessary for external validity).** At least 2 fixed cameras (phones/webcams are sufficient) covering one short real corridor stretch, recording ordinary traffic plus a handful of *deliberately staged* scenarios (a parked car as a lane blockage, a slow-driving vehicle simulating a disabled vehicle, a coned-off lane) — matching how CIRS and the WTS dataset (both found in prior research) handle the same real-incident-scarcity problem. Real accidents/collisions are not waited for, staged, or engineered — H3 (collision-like obstruction) is validated primarily on the SUMO track and, if available, existing public accident-video clips, never by staging an actual collision.

**Minimum dataset for M1:** the SUMO track's 20–30 episodes (fast to produce, no recording logistics) plus, in parallel, a few hours of real 2-camera footage covering at least 2–3 staged scenario types and ordinary H1/H7 cases — the real-video minimum does not need all seven hypotheses represented on day one; it expands over subsequent milestones.

**Splits:** source-scenario-level and time-window-level (per the frozen Master Spec's leakage-prevention discipline) — no SUMO episode or real clip contributes to more than one of train/val/test/demo (V1 has no "training" in the ML sense for the reasoning core, but the split discipline still applies to any threshold-tuning done on the data).

---

## PART 13 — Repository Structure

```
anvesh/
├── README.md
├── pyproject.toml
├── configs/
│   ├── cameras/              # per-camera CalibrationProfile configs
│   ├── corridor/              # corridor topology (2-segment ordering, inter-camera distance)
│   └── experiment/            # pinned model versions, thresholds, Δt, λ_shared, etc.
├── anvesh/                    # main package — one process, modular monolith
│   ├── perception/
│   │   ├── ingestion.py           # module 1
│   │   ├── detection.py           # module 3
│   │   ├── tracking.py            # module 4
│   │   └── traffic_state.py       # module 5
│   ├── world/
│   │   ├── calibration.py         # module 2
│   │   └── corridor.py            # module 7
│   ├── evidence/
│   │   ├── evidence_builder.py    # module 8
│   │   └── reliability.py         # module 9
│   ├── fusion/
│   │   ├── alignment.py           # module 6
│   │   └── ds_fusion.py           # module 10, Part 6's equations
│   ├── hypotheses/
│   │   ├── cause_model.py         # H1-H7 definitions, module 12
│   │   └── ranking.py
│   ├── propagation/
│   │   └── shockwave.py           # module 13, Part 8
│   ├── feedback/
│   │   └── update_engine.py       # module 14, Part 9
│   ├── storage/
│   │   ├── schemas.py             # Part 5, imported by every other package — frozen Day 1
│   │   └── db.py                  # SQLite access
│   ├── api/
│   │   └── app.py                 # FastAPI, module 15
│   └── evaluation/
│       ├── metrics.py             # Part 11
│       ├── baselines.py           # Part 10's A/B/C/ANVESH harness
│       └── experiment_run.py
├── datasets/
│   ├── sumo/                  # network configs + generated episode logs (Track 1)
│   └── recorded/               # real footage, gitignored, referenced by path/checksum (Track 2)
├── scripts/
│   ├── run_pipeline.py
│   ├── run_sumo_experiment.py
│   └── run_evaluation.py
├── ui/                          # Jinja2 templates (MUST HAVE); react-app/ only if the SHOULD-HAVE upgrade is taken
├── tests/
│   ├── perception/ fusion/ hypotheses/ propagation/ feedback/   # one dir per Part 3 module cluster
└── docs/
    └── adr/
```

This deliberately departs from an `apps/`+`services/` layout: the frozen architecture (Part 3, and the unmodified Master Spec before it) is a single-process modular monolith, and an `apps/services` split would visually imply a microservice decomposition this project has repeatedly, explicitly rejected. `storage/schemas.py` is placed to be importable by every package precisely because Part 5's schemas are meant to be the one shared contract — freezing it on Day 1 is the single most important repository-structure decision for the 4-track team split in Part 15.

---

## PART 14 — Milestones

| # | Objective | Files/modules created | Acceptance criteria | Test/demo |
|---|---|---|---|---|
| **M0** | Repository bootstrap | `pyproject.toml`, `configs/`, `storage/schemas.py`, empty SQLite schema, CI/lint | Config loads and validates; DB initializes with all Part 5 tables; `pytest` runs (even trivially) | `pytest` green on an empty test suite |
| **M1** | One-camera perception | `perception/*` | Given a fixture video, produces correct `Detection`/`VehicleTrack`/`TrafficState` counts on a hand-checked short clip | Contract test passes against a fixture with known expected counts |
| **M2** | Two-camera alignment | `world/calibration.py`, `world/corridor.py`, `fusion/alignment.py` | Given two fixture videos with a known injected time offset, alignment synchronizes within a stated tolerance; corridor topology correctly orders upstream→downstream | Fixture test: known offset in, corrected alignment out |
| **M3** | Corridor state | `evidence/*` (partial — reliability only, no fusion yet), `storage/db.py` wiring | Per-camera `TrafficState` correctly assembled into a `CorridorState`; congestion-level hysteresis passes a hand-labeled fixture (the "one noisy frame shouldn't flip state" regression, carried from the frozen Master Spec) | Hand-labeled fixture regression test |
| **M4** | Fusion | `fusion/ds_fusion.py`, `evaluation/baselines.py` (A/B/C harness stood up) | Fusion output on a fixture with known conflicting/agreeing evidence matches a hand-computed reference `Bel`/`Pl`; Baselines A/B/C differ as expected on a designed fixture | Hand-computed DS combination reference test |
| **M5** | Cause hypotheses | `hypotheses/*` | On SUMO fixture episodes with known injected cause, correct top-1 hypothesis on a majority of clean (non-adversarial) episodes | Run against the Track-1 SUMO fixture set |
| **M6** | Propagation | `propagation/shockwave.py` | Predicted shockwave speed/arrival matches a hand-computed reference given known flow/density values (verifies the Rankine-Hugoniot arithmetic) | Hand-computed reference test |
| **M7** | Feedback | `feedback/update_engine.py` | Fixture scenarios for each of the 4 feedback branches (match/partial/contradict/missing) produce the correct `HypothesisUpdate`; ANVESH (C+feedback) measurably differs from C alone on a fixture designed so the initial ranking is wrong and later evidence should correct it | Four branch-coverage fixtures + one designed correction-case fixture |
| **M8** | Evaluation | `evaluation/metrics.py`, `evaluation/experiment_run.py` | Full Part 11 metric suite + all six ablations run end-to-end from one command on both SUMO and (whatever exists of) the real-video track; every run produces an `ExperimentRun` record | One command produces a complete ablation report |
| **M9** | Demo/UI | `api/app.py`, `ui/` | A human can replay a scenario end-to-end and watch the hypothesis ranking update live as propagation evidence arrives, across the 3 dashboard views | Live walkthrough of one full scenario in the UI |
| **M10** | Research packaging | write-up, finalized `datasets/`, reproducibility check | RQ/hypotheses/results reported against Part 1's pre-registered success criteria; a fresh clone + documented run reproduces the headline numbers | Fresh-environment reproduction run |

---

## PART 15 — Team Task Split

| Track | Owns | Primary milestones | Depends on |
|---|---|---|---|
| **A — Perception** | `perception/`, `world/` | M1, contributes to M2/M3 | `storage/schemas.py` (frozen Day 1, M0) |
| **B — Fusion/reasoning** | `fusion/`, `hypotheses/`, `propagation/`, `feedback/` | M4–M7 | Track A's `TrafficState`/`Evidence` schema (can build against a **mocked** stream matching the Part 5 schema before A's real CV pipeline is fully polished) |
| **C — Data/evaluation** | SUMO scenario generation, real-video recording/staging coordination, annotation tooling, `evaluation/` | Dataset creation from M0 onward (parallel), M8 | Track B's output schemas (`CandidateCauseHypothesis`, `PropagationPrediction`, `HypothesisUpdate`) for building the evaluation harness against — but SUMO scenario generation and annotation tooling can start immediately using the frozen schemas as a mock |
| **D — Backend/UI/integration** | `storage/db.py`, `api/`, `ui/`, config system, CI, cross-track schema-drift watch | M0, M9, continuous integration glue | Everyone — D is the natural integration owner |

**The single most important practice:** freeze `storage/schemas.py` (Part 5) literally on Day 1 as the one file all four tracks import and none of them modify unilaterally — this is what lets A, B, and C build in parallel against mocks of each other's real output, matching the "interfaces before implementations" principle already established throughout the earlier frozen documents.

---

## PART 16 — Risks

| Risk | Mitigation |
|---|---|
| Lack of suitable multi-camera datasets | Two-track dataset strategy (Part 12) — never block on finding a perfect public dataset; SUMO de-risks the reasoning core independently of real-data availability |
| Camera synchronization | Record with device-clock/NTP sync; validate with a one-time manual "walk-through" landmark event at recording start, checked in M2's acceptance test |
| Camera calibration | Reuse the frozen Master Spec's documented, versioned, error-bounded calibration procedure; re-survey if a camera moves |
| Tracking failures (ID switches, occlusion) | Explicit `occlusion_state` field (Part 5) feeding reliability discounting (Part 6); robustness ablations (Part 11) measure the impact directly rather than assuming it away |
| False cause inference | The H7/`insufficient_evidence` abstention contract, the conservative shared-uncertainty discount, and the false-confidence-rate metric are all specifically designed against this |
| Correlated camera errors | This *is* the SH1 research question — not an unmitigated risk but a designed-around, directly measured ablation (Part 11) |
| Propagation model mismatch under heterogeneous traffic | Explicitly tracked as SH4 — expected to surface a real limitation, reported honestly in M10, not hidden |
| Overclaiming novelty | Part 1's RQ and this document's Part 6/Part 7 explicitly cite every piece of prior art the four-agent research found (CN118097571A, the Hitachi patent, ConFormer/IGSTGNN, SCI/ICI, contextual discounting) and narrow the claim to exactly the feedback loop + shared-uncertainty discount + evidence-completeness weighting — carry this discipline unmodified into any paper or pitch |
| Insufficient ground truth | SUMO provides exact ground truth for the reasoning core; the real-video track uses multi-annotator adjudication with an explicit `indeterminate` outcome, never a forced label (per the frozen Master Spec's ground-truth methodology) |

---

## PART 17 — Final Decision

1. **FINAL V1 RESEARCH QUESTION:** does uncertainty-aware cross-camera evidence fusion, combined with a propagation-based feedback loop that revises candidate-cause hypothesis confidence against observed corridor propagation, produce more accurate and better-calibrated corridor-level cause hypotheses than independent single-camera reasoning and naive multi-camera averaging — and does the feedback loop add value beyond fusion alone? (Part 1)
2. **FINAL V1 ARCHITECTURE:** a single-process, 16-module modular monolith, no event bus, no microservices, 2 cameras / 1 corridor (Part 3).
3. **FINAL TECH STACK:** Python, OpenCV, Ultralytics YOLO + ByteTrack, NumPy/Pandas, a hand-rolled Dempster-Shafer module, SQLite, FastAPI, Jinja2 (React optional), SUMO for simulation, pytest (Part 4).
4. **FINAL FUSION METHOD:** reliability-discounted Dempster-Shafer combination with an explicit, versioned shared-uncertainty discount blend (Part 6).
5. **FINAL CAUSE SET:** H1 downstream bottleneck, H2 stopped/disabled vehicle, H3 collision-like obstruction, H4 lane blockage, H5 upstream demand surge, H6 abnormal signal restriction, H7 insufficient evidence (Part 7).
6. **FINAL PROPAGATION METHOD:** rule-based, classical kinematic-wave/Rankine-Hugoniot shockwave estimation — not learned (Part 8).
7. **FINAL FEEDBACK METHOD:** the explicit 4-branch (confirmed/partial/contradicted/evidence-missing) predicted-vs-observed comparison loop (Part 9).
8. **FIRST DATASET/EXPERIMENT:** the SUMO Track-1 corridor, 20–30 scripted episodes across H1–H6 plus H7 controls (Part 12).
9. **FIRST MILESTONE:** M0 (repository bootstrap + frozen schemas) immediately followed by M1 (one-camera perception) (Part 14).
10. **WHAT WE SHOULD BUILD FIRST TOMORROW:** `storage/schemas.py` (Part 5, frozen, imported by everything) and the SUMO corridor network + first scripted episode (Track C) — in parallel with Track A starting M1's `perception/ingestion.py` and `detection.py` against a single fixture video. Nothing else starts before the schema file exists.

**STOP. This blueprint is complete. No code has been written. Awaiting approval before implementation begins.**
