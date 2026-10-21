# AGENTS.md

Guidance for coding agents working in the LightAutoML repository.

## Project overview

LightAutoML is a Python library for automated machine learning. Read the
supported Python range from `tool.poetry.dependencies.python` in
[pyproject.toml](pyproject.toml) before making compatibility decisions. Do not
duplicate version numbers here. Keep syntax and standard-library APIs
compatible with the declared range.

Keep changes focused on the request and, when available, its linked issue or
discussion. The contribution process expects substantial work to be discussed
before implementation. If that prerequisite is missing, report it to the user;
do not create issues, branches, commits, pushes, or pull requests unless the
user explicitly asks for that external action.

## Sources of truth

This file summarizes repository practice. If instructions conflict, use the
following project files as the authoritative source for their respective area:

- `.github/CONTRIBUTING.md` — contribution, testing, and documentation process.
- `pyproject.toml` — supported Python range, dependencies, extras, and Black and
  isort settings.
- `setup.cfg` — Flake8, docstring, and rstcheck rules.
- `tox.ini` — executable test, lint, documentation, and spelling environments.
- `.pre-commit-config.yaml` — hooks that run on commits and pushes.

When repository configuration and prose disagree, follow the executable
configuration and mention the discrepancy in the handoff.

Follow the shared contributor rules in
[Branch naming](.github/CONTRIBUTING.md#branch-naming),
[Code conventions](.github/CONTRIBUTING.md#code-conventions),
[Tests and checks](.github/CONTRIBUTING.md#tests-and-checks), and
[Preparing a release](.github/CONTRIBUTING.md#preparing-a-release).

## Repository layout

- `lightautoml/` — library source code.
  - `automl/` — AutoML classes, blenders, and presets.
  - `dataset/`, `reader/`, `tasks/`, `validation/` — data interfaces, input
    handling, task definitions, and validation.
  - `ml_algo/`, `pipelines/`, `transformers/` — algorithms, pipelines, and
    feature transformations.
  - `text/`, `image/`, `addons/`, `report/` — optional and domain-specific
    functionality.
  - `utils/` — shared utilities such as logging, timers, and profiling.
- `tests/` — unit and integration tests.
- `docs/` — Sphinx documentation and tutorials.
- `examples/` — runnable usage examples and notebooks.
- `scripts/` — repository maintenance and experiment helpers.

## Working in the repository

- Inspect `git status` before editing and again before handing work off.
- Treat pre-existing modified and untracked files as user-owned. Do not revert,
  overwrite, reformat, stage, or commit them.
- Keep the diff limited to the requested behavior; avoid drive-by cleanup.
- Do not use destructive Git operations such as `git reset --hard`,
  `git clean`, or forced checkout to discard changes.
- Do not commit, push, create a pull request, or modify remote state unless the
  user explicitly requests it.
- Do not create or switch branches unless the user requests it. Take the base
  branch from the request or linked issue; if none is specified, ask the user
  before creating the branch.
- Flag a missing issue before pull-request preparation rather than creating
  one without permission.

## Code Review Rules

When asked to review code, review the complete relevant diff and its affected
call sites without modifying files unless the user also asks for fixes. Focus
on actionable defects and regressions rather than summarizing the patch.

Prioritize findings in this order:

1. Correctness, data loss, security or privacy problems, and crashes.
2. Data leakage between training, validation, and test sets; invalid metric or
   split behavior; nondeterminism that makes results unreliable.
3. Backward compatibility of public APIs, serialized models, configuration,
   dataset roles, feature names, shapes, indexes, and dtypes.
4. Missing error handling, optional-dependency guards, or support for the
   declared Python versions and CPU-only environments.
5. Material performance or memory regressions and missing tests or
   documentation for changed behavior.

For every finding:

- Assign a severity (`critical`, `major`, or `minor`) and order findings from
  highest to lowest severity.
- Cite the narrowest relevant file and line, describe the triggering scenario,
  explain the concrete impact, and suggest a direction for remediation.
- Verify that the issue is introduced or exposed by the reviewed change. Do not
  report speculative concerns without a plausible failure path.
- Do not report formatting issues already handled by Black, Flake8, or another
  configured check unless they reveal a functional or maintainability problem.

Check whether tests cover the success path, boundary conditions, failure path,
and regression being fixed. If there are no findings, say so explicitly and
state any residual risks or checks that were not run.

## Pull Request Preparation

Prepare or create a pull request only when the user explicitly requests it.
Before doing so, verify that the diff contains only intentional files, tests
cover the changed behavior, and relevant documentation or examples are present
for user-visible functionality.

The pull request body must use `.github/pull_request_template.md‎` as its
starting point and retain all of the template's headings. The current filename
has a trailing Unicode left-to-right mark (`U+200E`); use the existing file
rather than creating a second file with a visually identical name. Read the
template immediately before preparing the body because its contents may
change.

Complete the template as follows:

- `What was done` — summarize the behavior-level changes and list the relevant
  validation commands and results. Call out any checks that were not run and
  explain why.
- `Why / Goal` — explain the motivation and include the real goal or milestone
  identifier and title when one was supplied.
- `Related issues` — reference the actual issue number, for example `#123`. Use
  `N/A` when no issue exists and flag that fact to the user before PR creation.

Do not invent issue, goal, milestone, test, or benchmark results. Remove
placeholder text and instructional comments after completing the template, but
do not omit or rename its sections. Review the rendered body for empty fields,
broken references, and user-specific or sensitive information before handing
it off or creating the pull request.
