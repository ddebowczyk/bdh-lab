# BDH Lab instructions

Use Simplified Technical English. State what changed and name the evidence.

- Use `uv` for every Python command. Do not use a global virtual environment.
- Use `just` for project commands. Run `just --list --unsorted` before using an
  unfamiliar recipe.
- Treat `components/`, `assemblies/`, `experiments/`, and `profiles/` as
  versioned metadata. Use `yq` to inspect or preview YAML changes and `ys` to
  validate them.
- Keep provider credentials only in environment variables or provider-managed
  user configuration. Never write credentials to profiles, run records, or
  command output.
- `var/runs/` is generated and ignored. A run must write resolved inputs and
  metrics there. Do not edit a previous run record.
- A result can support promotion or rejection, but it must not silently edit a
  component or assembly. Record design decisions as reviewed metadata changes.
