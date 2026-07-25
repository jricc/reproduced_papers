# QSVM Medical Imaging Reproduction Plan

## Goal and claim boundaries

Reproduce a small, scientifically meaningful subset of
[*Quantum Kernel Advantage over Classical Collapse in Medical Foundation Model
Embeddings*](https://arxiv.org/html/2604.24597v1), within these constraints:

- CPU only; no GPU or HPC assumption;
- no access to the original MIMIC-CXR/insurance dataset;
- integration with the shared `reproduced_papers` runtime;
- a separate MerLin 0.4 photonic adaptation after the qubit baseline is
  characterized.

Three scopes must remain visibly separate:

1. **Reference protocol**: the paper's MIMIC-CXR insurance task and frozen
   foundation-model embeddings. This can be documented and supported by config,
   but cannot currently be executed locally.
2. **Open-data surrogate**: the same comparison protocol on a different task and
   dataset. It tests the method, not the paper's medical or quantum-advantage
   claim.
3. **MerLin photonic adaptation**: a related photonic fidelity kernel evaluated
   on the same surrogate splits. It is not a faithful implementation of the
   paper's qubit BSP circuit unless an explicit mapping is established.

No local result may be described as a reproduction of the paper's `18/18` or
`7/7` wins. PneumoniaMNIST raw pixels are suitable for a pipeline smoke test,
not as a scientific substitute for frozen foundation-model embeddings.

## Preservation strategy

The authors' pipeline at the pinned upstream revision is the initial baseline.
For the first tests:

- preserve its scientific behavior, including audited differences from the
  paper text;
- limit local modifications inside upstream code to CPU/serial execution and
  alternative-dataset compatibility;
- add catalogue integration through thin wrappers rather than moving or
  rewriting the upstream implementation;
- label outputs `upstream_baseline` and use them for characterization, not as
  validated scientific claims;
- discuss the observed behavior before correcting normalization,
  preprocessing, circuit, failure handling, or aggregation;
- if a correction is approved, retain the baseline and introduce a separately
  named config/mode so results remain comparable.

The complete findings and open questions are retained in [`AUDIT.md`](AUDIT.md).

## Sources and provenance

- Paper: arXiv `2604.24597v1`, 27 April 2026.
- Authors' repository: <https://github.com/sebasmos/qml-medimage>.
- Audited upstream revision: `9e80037305d683b0e70c94b8fa7dd648e1bac82b`
  (the only upstream commit observed on `main` during the July 2026 audit).
- Target catalogue: <https://github.com/merlinquantum/reproduced_papers>.
- MerLin target: [documentation 0.4](https://merlinquantum.ai/0.4/index.html)
  and source tag `0.4.0`.

Before publishing results, record the imported upstream revision, paper version,
dataset/model identifiers, licenses, checksums, software versions, and local
deviations in machine-readable metadata. Resolve the apparent provenance
discrepancy where the paper calls DT9 "Uncertainty Coreset" while the gated
dataset filenames reportedly associate different names with DT9/DT11. Do not
guess the mapping.

## Audit snapshot (24 July 2026)

### What already exists

- The authors' Qiskit/MPI/cuQuantum scientific code is retained; obsolete HPC
  launch material was removed after approval.
- A serial MPI fallback and an exact Qiskit statevector CPU path were added.
- A PneumoniaMNIST preparation script and a bounded Table 1 adaptation launcher
  exist for `q = 2, 4, 6`, five seeds, and at most 200 samples.
- Linear `C=1` and tuned-RBF baselines plus a Table 1 summary script exist.
- README/NOTICE already acknowledge that PneumoniaMNIST is not MIMIC-CXR.
- A MerLin 0.4 fidelity-kernel smoke, curated local results, and the thin shared
  catalogue runner are present.

### Scientific observations retained for discussion

1. **Kernel normalization**: the training Gram matrix is trace-normalized, while
   rectangular validation/test blocks are left unscaled; Frobenius blocks use
   independent scales. The paper requires the training trace to normalize the
   test--train block too.
2. **Data leakage**: the default preprocessing fits MinMax scaling on training
   plus held-out data. The Table 1 launcher uses this default, and baseline
   scripts do not implement one canonical train-only pipeline.
3. **Incomplete runs look successful**: broad exception handling can produce a
   zero exit status without artifacts; aggregation accepts missing/duplicate
   seed--qubit cells.
4. **Circuit mismatch**: the paper describes one `Ry` per qubit followed by CNOT
   ring entanglement, while the audited upstream `make_bsp` adds other rotations
   and uses an open CNOT chain. The intended reference circuit must be frozen and
   named before comparing results.
5. **Protocol ambiguity**: "full train" does not currently include validation,
   preprocessing differs across scripts, and random caps are not stratified.
6. **Kernel validation**: rounded matrices are passed to SVC without systematic
   finiteness, shape, diagonal, symmetry, range, or PSD checks.

Items 1--4 are present in or interact with the preserved upstream path. They do
not need to be changed before the initial characterization tests. Results from
that path must, however, remain labelled as diagnostic baseline results and must
not support a scientific reproduction claim. See `AUDIT.md` for origin evidence
and the full discussion record.

### Catalogue and maintenance issues

- The shared runtime markers and thin MerLin runner are present. A user-run
  catalogue smoke completed successfully and wrote the standard configuration,
  log, metrics, and dataset metadata artifacts.
- The catalogue path uses `data/qsvm_medimage/` and writes raw runs under
  `outdir/run_*`; direct legacy scripts remain available separately.
- Obsolete HPC tests and artifacts were removed after an explicit inventory,
  and a fast runtime-contract test was added.
- Direct dependencies are declared consistently in `requirements.txt` and
  `pyproject.toml`.
- Small sanitized result artifacts are versioned under `results/`.

## Selected artifacts

Table 1 is the conceptual comparison framework. Its detailed evidence is in
Tables 2 and 4, so the adapted Table 1 must be derived from detailed per-seed
records rather than produced alone.

The criteria below describe the eventual auditable artifacts. The first
`upstream_baseline` tests may deliberately retain known upstream behavior and
therefore do not yet satisfy all of these scientific criteria.

### Priority 0: minimum useful reproduction

1. **Tier 1 / Table 2 analogue**: QSVM `C=1` versus linear SVM `C=1`, identical
   PCA dimension, split, preprocessing, and paired seeds. Primary metric:
   minority-class F1; also report accuracy, recall, AUC when defined, sample
   counts, and class balance.
2. **Tier 2 / Table 4 analogue**: the same QSVM versus RBF SVM with
   `C in {0.01, 0.1, 1, 10, 100}` selected on validation data and
   `gamma="scale"`.
3. **Table 1 adapted summary**: configuration wins and paired deltas derived
   only after the Tier 1/2 grids pass completeness checks. Never reuse the
   paper's `18/18` or `7/7` denominators for a smaller grid.
4. **Minority-class behavior**: one paired confusion-matrix figure/table for a
   predeclared representative configuration, not whichever run looks best.

Initial CPU scope: one surrogate embedding family, `q = 2, 4, 6`, and five
paired seeds. Label this exploratory. A confirmatory ten-seed run is optional
and must be decided from a user-run timing pilot; every artifact states its
actual denominator.

### Priority 1: mechanism and robustness

5. **Table 5 + Figure 3 analogue**: effective rank and eigenspectrum of the
   trace-normalized quantum, linear, and RBF training kernels on the same split.
6. **Figure 4 analogue**: one trace-normalized kernel heatmap using the same
   predeclared configuration as the confusion matrix.
7. **Figure 5 analogue**: F1 versus `q`, reusing the complete Tier 1 records; do
   not run a separate sweep if the same data suffice.
8. **Table 7 normalization ablation**: trace versus the explicitly defined
   alternatives, only after cross-kernel normalization is correct and tested.

### Deferred

- Table 8 circuit-degree/depth ablations and Table 10 rank-matched RBF.
- `q >= 8`, especially `q=16`, projected kernels, additional embedding models,
  or the full upstream grid.
- Hardware or computational-advantage claims. The paper reports noiseless
  simulation evidence, not a demonstrated speedup on quantum hardware.

## Candidate corrected protocol (decision deferred)

This is the protocol suggested by the audit for a possible corrected comparison.
It must not replace the upstream baseline before the initial tests and an
explicit discussion/approval.

- Define a versioned split manifest with deterministic stratification and paired
  sample IDs for every method.
- Fit `StandardScaler -> PCA(q) -> MinMaxScaler[-1, 1]` on training data only,
  once per split/seed. Reuse the fitted objects for validation and test.
- Explicitly define whether the final estimator is trained on train only or
  train+validation. If train+validation is chosen, refit preprocessing only on
  that combined partition and never on test.
- Define the positive/minority label from the training split and preserve it in
  metrics metadata.
- Use the paper-described 1-DOF `Ry` + CNOT-ring circuit as the reference target.
  If the upstream implementation is retained for comparison, give it a distinct
  `upstream_legacy` name and never mix its results with the reference circuit.
- Compute the raw training Gram matrix once. Derive the trace scale from it and
  apply that same scale to validation/test cross-kernels.
- Reject non-finite values, wrong shapes, unexpected diagonal/range violations,
  and materially non-symmetric training matrices. Record any PSD repair rather
  than applying it silently.
- Use the same `C=1`, PCA features, splits, and scoring implementation for the
  Tier 1 pair. Select RBF `C` using validation minority F1 with a deterministic
  tie rule.
- Save one resolved config, split/data fingerprint, dependency versions, raw
  per-seed metrics, aggregate metrics, and artifact manifest per run. Do not save
  controlled/raw feature vectors by default.

## Implementation phases

Current execution order agreed with the user: MerLin implementation, README,
pedagogical notebook, then an explicitly approved cleanup.

### Phase 0 -- Governance and provenance

Status: **audit complete; metadata implementation pending**.

- Merge the two local AGENTS files and adapt the useful generic rules to this
  Python/scientific project.
- Preserve the detailed audit conclusions and open questions in `AUDIT.md`.
- Pin the paper version and upstream revision in NOTICE/README metadata.
- Create an explicit terminology table for reference, surrogate, smoke, and
  MerLin results.
- Review dependency/data/model licenses and clarify their scope without making
  unsupported legal claims.
- Build a keep/archive/delete inventory for upstream HPC/docs/notebook files;
  request approval before deletion.

Acceptance: one authoritative `AGENTS.md`; no ambiguous reproduction claim; no
destructive cleanup hidden inside a refactor.

### Phase 1 -- Minimal CPU and alternative-dataset adaptation

Status: **complete for the CPU/PneumoniaMNIST smoke**.

- Preserve the original scripts and their scientific behavior.
- Keep only the local CPU statevector path, serial fallback, and support for the
  alternative dataset schema.
- Make an optional dependency non-blocking only when it prevents the CPU path
  from importing or running.
- Do not add a new framework, abstraction, inventory, or test architecture in
  this phase.
- Use the original entry point with the smallest practical CPU configuration;
  the user runs it and reports the result.

Acceptance: the original QSVM script can read the prepared alternative dataset,
run on CPU without CUDA/HPC, and write its normal outputs. The diff remains
local and minimal. After that first run, discuss the observed pipeline behavior
before considering any scientific correction.

### Phase 2 -- Integrate the shared catalogue runtime

Status: **minimal integration and user-run smoke complete**.

Keep only the minimum runtime skeleton:

- `configs/defaults.json`: intentionally small CPU-safe defaults;
- root `cli.json`: paper-specific options only;
- `lib/runner.py::train_and_evaluate(cfg, run_dir)` as a thin adapter around the
  validated MerLin entry path.

Use `data/qsvm_medimage/` for datasets, `outdir/run_*` for disposable raw runs,
and `results/` only for curated, sanitized outputs. Keep the existing upstream
scripts available and avoid duplicating their scientific logic. Do not restore
the previous synthetic framework wholesale.

Acceptance commands for the user to run from repository root:

```bash
python implementation.py --list-papers
python implementation.py --paper qsvm_medimage --help
pytest -q papers/qsvm_medimage/tests
```

The paper must be discoverable, imports must be CPU-only by default, and the
small prepared-data smoke must write the standard config snapshot, log, and
result artifacts when the user chooses to execute it.

### Phase 3 -- Establish data tiers

Status: **not started**.

- Keep raw-pixel PneumoniaMNIST only as `pneumoniamnist_smoke`.
- Select an accessible open chest-X-ray dataset plus a frozen embedding model
  for the primary surrogate only after checking access, redistribution terms,
  label suitability, download size, and CPU extraction cost.
- Cache embeddings outside the paper directory with model revision, transform,
  sample-ID, and dataset checksums. Do not redistribute inputs unless their
  license clearly permits it.
- Keep the gated MIMIC path optional. A missing credential or dataset must fail
  clearly, never trigger a silent fallback to another task.

Acceptance: a short data card defines task, source, labels, positive class,
split policy, representation, license, and all deviations. The primary surrogate
uses frozen embeddings; smoke and primary configs cannot be confused.

### Phase 4 -- Run and review the qubit baseline

Status: **blocked on Phases 1--3 and user execution**.

The agent prepares commands and validates configuration statically. The user
runs experiments in this order:

1. tiny prepared-data smoke;
2. one-configuration timing/memory pilot;
3. one small `upstream_baseline` Tier 1 comparison;
4. review the baseline and the discussion gate;
5. only after approval, the chosen exploratory Tier 1/Tier 2 grids;
6. derived Table 1 and predeclared confusion matrix;
7. optional ten-seed confirmation if the pilot budget is acceptable.

Acceptance: every expected cell is present exactly once; paired raw metrics and
aggregates agree; plots are reproducible from CSV/JSON without rerunning kernels;
outputs identify both `surrogate` and the protocol (`upstream_baseline` or an
approved corrected mode) and include dataset, representation, `q`, seeds,
samples, uncertainty, timing, and deviations. Baseline diagnostics are not
promoted to validated reproduction results.

### Phase 5 -- Explain the kernel behavior

Status: **blocked on a reviewed Phase 4 protocol/run**.

- Derive effective rank, eigenspectra, heatmap, and `q` sweep from saved kernels
  and records where possible.
- Predeclare the displayed configuration and eigenvalue tolerance.
- Compare quantum, linear, and RBF kernels on identical training samples.
- Add the normalization ablation only after the primary trace result is stable.

Acceptance: Tables/Figures 3--5/7 analogues are generated from structured,
auditable inputs; no visual cherry-picking; numerical definitions and tolerances
are documented.

### Phase 6 -- Add the MerLin 0.4 adaptation

Status: **local and shared-runtime smoke complete**.

- Implement the documented `FeatureMap` + `FidelityKernel(feature_map,
  input_state, ...)` API. Prefer an explicit `input_state` so that photon count
  is recorded; the installed 0.4.0 still accepts `n_photons=`, but
  `FidelityKernel.simple()` is deprecated.
- Define the mapping from PCA features to photonic phases and state explicitly
  whether it is a native photonic analogue or a resource-matched translation.
- Reuse the exact surrogate split manifests, fitted features, SVM protocol, and
  artifact schema from the qubit comparison.
- Add tiny checks for train/test shape, symmetry, diagonal, range, determinism,
  dtype/device, PSD policy, and sklearn precomputed-kernel compatibility.
- Keep CPU exact computation as the default. Ask the user to run a very small
  timing/memory pilot before selecting modes, photons, sample count, or a grid.
- Isolate or reconcile MerLin 0.4's Python/NumPy/scikit-learn requirements rather
  than forcing an unverified dependency upgrade into the Qiskit environment.

Acceptance: the adaptation runs through the shared runtime on a tiny user-run
CPU fixture, produces the same structured artifact contract, and is never
reported as reproducing the paper's qubit kernel without equivalence evidence.

### Phase 7 -- Notebook, documentation, and curated results

Status: **paper-local README, notebook, and curated artifacts complete**.

- Create a pedagogical `notebook.ipynb` that loads a small fixture or existing
  structured run; it must not require HPC or execute a full Gram computation.
- Rewrite README sections for attribution, original method, exact scope,
  install/run commands, configs, data, obtained results, limitations, tests,
  citation, and license.
- Update NOTICE with the upstream revision and substantive local changes.
- Curate small CSV/JSON/PNG artifacts in `results/`; strip absolute paths, raw
  embeddings, notebook outputs, and private metadata.
- Update the catalogue root README only after actual user-run metrics exist.

Acceptance: all documented commands match the runtime; every displayed number
has a curated source artifact; links are relative and valid; missing results are
called TODO rather than inferred.

### Phase 8 -- Approved cleanup and final review

Status: **approved first cleanup complete**.

- Present the keep/archive/delete inventory for explicit approval.
- Remove only approved obsolete HPC scripts, broken tests, stale generated site,
  caches, and redundant dependency entries.
- Preserve useful upstream provenance and any compatibility scripts documented
  as such.
- Before opening a pull request, create a branch with an accepted catalogue
  prefix such as `paper-qsvm-medimage`; keep the current development branch
  unchanged until then.
- Perform the final focused checks only when explicitly requested by the user.

The approved batch removed obsolete HPC launchers, generated documentation,
private-path notebooks, broken tests, and unused dependencies. Active CPU and
MerLin scripts, `qve/`, useful tests, curated artifacts, and Python PCA
utilities remain.

Acceptance: no absolute local/HPC path in the supported runtime or documentation;
retained upstream utilities with HPC defaults are explicitly unsupported; no
duplicate implementation path; requirements match imports; Git diff contains no
generated/raw data; the user has the exact commands needed for final validation.

## Execution gate

Implementation starts only after this plan is reviewed. During implementation,
the agent may inspect and edit approved files and prepare tests/commands, but it
does not download data, launch training/evaluation, generate results/figures, or
run test/build/benchmark commands unless the user explicitly requests that
execution in the same message. Upstream scientific behavior is not corrected or
refactored without a separate discussion and explicit approval.
