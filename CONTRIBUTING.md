# Contributing to AlphaMoo

Thanks for your interest in contributing! AlphaMoo is an active research project building toward ARC Prize 2026. All contributions are welcome — bug reports, fixes, new modules, performance improvements, documentation.

## Code of Conduct

Be excellent to each other. We're building an AI agent, not a Twitter pile-on.

## How to contribute

### Reporting bugs
1. Search existing [issues](https://github.com/0xKunalsharma/Alphamoo/issues) to avoid duplicates
2. Open a new issue using the **Bug report** template
3. Include reproduction steps and environment info

### Suggesting features
1. Open an issue using the **Feature request** template
2. Tag it with the relevant module (see the issue template)
3. Discuss the design before opening a PR for large changes

### Submitting code
1. Fork the repo and create a feature branch:
   ```bash
   git checkout -b feature/my-new-module
   ```
2. Write code following the existing style (enforced by `ruff`)
3. Add tests in `tests/` for any new functionality
4. Make sure CI passes:
   ```bash
   ruff check src tests scripts
   ruff format --check src tests scripts
   pytest tests/ -m "not slow and not integration and not gpu"
   ```
5. Open a pull request with a clear description of what changed and why

## Development setup

```bash
# Clone
git clone https://github.com/0xKunalsharma/Alphamoo.git
cd Alphamoo

# Create venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: install LLM stack
pip install -e ".[llm]"

# Optional: install ARC-AGI-3 toolkit
pip install -e ".[arc]"

# Run tests
pytest tests/

# Run a quick smoke test against real data
PYTHONPATH=src python scripts/test_reader.py
```

## Architecture overview

Read [`README.md`](./README.md) first for the high-level picture, then dive into [`docs/architecture.md`](./docs/architecture.md) (TODO — for now see `/home/z/my-project/download/AlphaMoo_v4_Design.md` and `AlphaMoo_v4.1_Delta.md` for the full spec).

The codebase follows the v4.1 module structure:

| Module | File | Status |
|--------|------|--------|
| 1. Perception | `src/alphamoo/perception.py` | ✅ |
| 2. Affordance | (TBD) | ⏳ |
| 3. Type Inferencer | (TBD) | ⏳ |
| 4. Working Memory | (TBD) | ⏳ |
| 5. Hypothesis Generator | (TBD) | ⏳ |
| 6. Goal Inference | (TBD) | ⏳ |
| 7. Near-Miss Tracker | (TBD) | ⏳ |
| 8. Experiment Planner | (TBD) | ⏳ |
| 9. World Model | (TBD) | ⏳ |
| 10. Verifier | (TBD) | ⏳ |
| 11. Planner Interface | (TBD) | ⏳ |
| 12. Context Compressor | (TBD) | ⏳ |
| 13. Reasoning Engine | (TBD) | ⏳ |
| 14. Cascade Interpreter | `src/alphamoo/cascade_interpreter.py` | ✅ |
| 15. Spatial Memory Map | (TBD) | ⏳ |
| 16. Agent State Tracker | `src/alphamoo/agent_tracker.py` | ✅ |
| 17. Policy Prior | (TBD) | ⏳ |
| 18. Goal Predicate Vocabulary | (in schemas) | ✅ |

## Coding standards

- **Style:** enforced by `ruff` (PEP 8 + isort + bugbear)
- **Types:** type hints encouraged; `mypy` is non-blocking in CI for now
- **Tests:** pytest, with markers (`slow`, `integration`, `gpu`)
- **Docstrings:** Google-style for public functions
- **No emojis in code** — emojis are fine in comments and docs, not in identifiers or output

## Security

**Never commit secrets.** This includes:
- API tokens (OpenAI, Anthropic, GitHub PATs, Hugging Face tokens)
- Model weights with restrictive licenses
- Private Kaggle data

The `.gitignore` blocks common patterns (`ghp_*`, `sk-*`, `*.env`), but **you are responsible** for double-checking before each commit. If you accidentally push a secret, treat it as compromised and rotate immediately.

## Roadmap

See [`README.md`](./README.md) → "What's Next" section. The current focus is Phase 2 (Exploration Loop). PRs aligned with the roadmap get priority review.

## License

By contributing, you agree that your contributions will be licensed under the MIT license (see [`LICENSE`](./LICENSE)).
