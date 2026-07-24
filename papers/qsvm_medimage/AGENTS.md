# AGENTS.md

## Core principles

- Tell the truth.
- Never invent facts, sources, files, commands, results, or code behavior.
- If information is missing, uncertain, or outdated, say so clearly.
- Separate facts, assumptions, and uncertainties.
- Prefer checking the repository before answering about its code.
- Do not claim that something works without evidence.

## Privacy and data safety

- Treat all user data, source code, documents, logs, and conversations as
  confidential.
- Prefer local analysis whenever possible.
- Do not share data with third parties or reuse it outside the current task.
- Do not print secrets, credentials, private data, raw controlled embeddings, or
  unrelated personal information.
- Explain why before using external access.

## File and command safety

- Read-only inspection commands are allowed when needed.
- Do not modify, move, delete, rename, reformat, stage, commit, push, reset, or
  rebase files without explicit approval.
- Do not install dependencies without explicit approval.
- Do not run tests, builds, benchmarks, or other long-running commands unless
  explicitly requested. The user normally runs them manually.
- Do not launch training, evaluation, dataset download, result generation, or
  figure generation commands. Preparing source code and launch scripts does not
  imply permission to run them.
- Do not hide failures or use broad fallbacks for required inputs. Fail clearly
  and preserve the original error context.

## Answer style

- Be short, direct, and precise.
- Use simple vocabulary and prefer clear explanations over jargon.
- Name the dataset, protocol, backend, and experiment when a statement depends
  on that context.
- Mention limitations, risks, and edge cases when useful.
- Cite sources when factual precision matters.
- Do not overstate results, scientific equivalence, or novelty.

## Working mode

Use simple mode by default:

- Understand the current code first.
- Propose the smallest useful change.
- Avoid long plans, generalization, new abstractions, and unrelated refactoring.
- Keep changes small and reversible.

Use strict quality mode only when explicitly requested. Before coding, state the
minimal solution in 2--5 lines, including the likely files and why the scope is
minimal. A more generic design may be mentioned as an alternative, but must not
be implemented unless requested.

## Code quality

- Prefer simple, readable, maintainable code and follow the existing project
  style.
- Use short functions, descriptive names, and explicit error handling.
- Prefer pure functions and immutable data when practical.
- Do not add classes, modules, helper layers, registries, callbacks, or
  frameworks unless they simplify the current task.
- A small duplicated expression is acceptable when an abstraction would make
  the code harder to understand.
- Reuse an existing function before adding a helper.
- Document important choices, limitations, and invariants. Avoid comments that
  merely restate the code.
- Do not rewrite generated artifacts, notebooks, caches, or unrelated results
  unless the task explicitly requires it.

## Python rules

- Use type hints where they clarify public interfaces and scientific array
  contracts.
- Document public functions, classes, and modules with concise NumPy-style
  docstrings. Include inputs, outputs, raised errors, defaults, and important
  invariants.
- Use `pathlib.Path` for new path handling and do not hard-code local absolute
  paths.
- Validate required configuration keys, array shapes, dtypes, labels, and finite
  values explicitly.
- Keep model, data, kernel, analysis, and orchestration logic separate only when
  that separation removes real duplication or makes scientific behavior easier
  to verify.

## Scientific reproducibility

- Keep reference reproduction, open-data surrogate experiments, and the MerLin
  photonic adaptation clearly separated. Do not claim they are equivalent.
- Preserve the upstream scientific pipeline as the initial baseline. Limit
  local changes in upstream code to CPU execution and alternative-dataset
  compatibility; document every such change.
- Record audited scientific concerns without silently correcting them. Run and
  discuss the preserved baseline first; any correction requires explicit
  approval and a separately named, comparable path.
- Record the protocol, deviations, data/model revisions, seeds, samples,
  positive label, metrics, and artifact completeness needed to trace a result.
- Do not make scientific claims from incomplete or unverifiable artifacts, and
  do not expose raw controlled embeddings or private paths.

## Repository integration

- Add the shared runtime markers (`configs/defaults.json`, root `cli.json`, and
  `lib/runner.py`) as thin wrappers around the preserved upstream pipeline.
- Keep JSON configs authoritative; use `data/qsvm_medimage/` for data,
  `outdir/run_*` for raw runs, and `results/` for small curated artifacts.
- Keep the detailed audit in `AUDIT.md` and the roadmap in `PLAN.md`.
- Inventory legacy files before cleanup and remove only an explicitly approved
  list.

## Testing and validation

- Prefer small, deterministic tests that isolate one scientific invariant.
- Cover normal, edge, error, and regression cases when relevant.
- Avoid tests that silently skip the only behavior they are meant to validate.
- CPU smoke tests must not require CUDA, MPI, HPC paths, network access, or the
  controlled original dataset.
- If checks are not run, say so. Never claim that tests passed unless their exact
  command and result are known.

## Review behavior

When reviewing code:

- Identify correctness and data-safety issues first.
- Then discuss reproducibility, design, readability, tests, and documentation.
- Distinguish blocking issues from suggestions.
- Propose minimal fixes first.

## After changes

If files were changed, summarize:

- what changed;
- which files changed;
- which commands were run;
- what remains uncertain;
- what the user should check or run manually.
