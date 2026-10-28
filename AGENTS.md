# Repository Guidelines

## Project Structure & Module Organization

This repository is currently a project scaffold: `README.md` contains the hackathon brief and submission requirements. As implementation begins, keep code under `src/`, tests under `tests/`, and documentation assets under `docs/` or `assets/`. Group code by responsibility (for example, `src/data/`, `src/models/`, and `src/inference/`). Store experiment settings in `configs/`. Do not commit datasets, checkpoints, model weights, or generated videos; document their download or generation steps and add their paths to `.gitignore`.

## Build, Test, and Development Commands

No build system, dependency manifest, or test runner exists yet. When adding one, document its commands here and in `README.md`. Prefer reproducible commands such as:

```bash
python -m venv .venv       # create a local environment
python -m pytest           # run the full Python test suite
python -m src              # run a package entry point
```

Pin direct dependencies. Never rely on global packages or developer-specific absolute paths.

## Coding Style & Naming Conventions

Follow the standard formatter and linter for the selected language, and commit their configuration with the first source files. For Python, use four-space indentation, `snake_case` for modules and functions, `PascalCase` for classes, and type hints on public interfaces. Use descriptive configuration names such as `configs/vial_baseline.yaml`; avoid ambiguous names like `test2.py` or `final_new.py`. Keep data loading, training, evaluation, and hardware integration separate so each can be tested independently.

## Testing Guidelines

Mirror source paths in `tests/` and name Python tests `test_<module>.py`. Add unit tests for preprocessing and decision logic, plus a small smoke test for each runnable pipeline. Tests must use tiny synthetic fixtures or documented samples, not private or full-size datasets. Record evaluation metrics, dataset version, and random seed for model-related changes.

## Commit & Pull Request Guidelines

The history currently contains only `Initial commit`, so no established convention exists. Use short, imperative subjects such as `Add vial defect baseline`; keep each commit focused. Pull requests should explain the problem, approach, validation commands, and known limitations; link relevant issues. Include before/after metrics for model changes and screenshots or short clips for visual or robotics behavior. Call out new data, compute, credentials, or hardware requirements explicitly.

## Security & Configuration

Keep API keys, tokens, credentials, and machine-specific settings in ignored environment files. Commit an `.env.example` containing names only. Before pushing, check `git status` and confirm that outputs and sensitive files are excluded.
