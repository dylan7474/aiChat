# Agent Guidelines

## Scope
These instructions apply to the entire repository.

## Build & Testing Expectations
- Always run `./configure` before attempting to build so the script can verify toolchain prerequisites.
- The project is intended to compile with GNU Make on Linux (`make`) and via the Windows-specific makefile (`make -f Makefile.win`).
- If `json-c` development headers are unavailable in the environment, call this out as a known limitation rather than attempting to implement a workaround.
- Prefer `pkg-config` to locate external libraries.

## Code Style Notes
- This is a C project; keep functions small and focused, favouring early returns on error conditions.
- Avoid introducing new dependencies without discussing them in documentation.

## Documentation
- Update `README.md` whenever build steps or prerequisites change.
- Mention any new command-line options or keybindings in the Basic Controls section.

## Pull Requests
- Summaries should highlight notable build or configuration changes.
- List any failing commands in the testing section with explanations.
