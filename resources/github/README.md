# Source repositories

These are shallow checkouts collected for local inspection. The links are the
canonical upstream repositories.

| Local path | Upstream | Role |
| --- | --- | --- |
| [`pathway-bdh`](pathway-bdh/) | [pathwaycom/bdh](https://github.com/pathwaycom/bdh) | Official public BDH model and toy trainer. |
| [`attempt-ssrhaso-bdh`](attempt-ssrhaso-bdh/) | [ssrhaso/bdh](https://github.com/ssrhaso/bdh) | Independent synthetic multi-hop reasoning attempt. |
| [`nanoGPT`](nanoGPT/) | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Small, readable Transformer control and training baseline. |
| [`nanochat`](nanochat/) | [karpathy/nanochat](https://github.com/karpathy/nanochat) | Recommended modular training, evaluation, and inference harness. |
| [`modded-nanogpt`](modded-nanogpt/) | [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Speedrun catalogue of architecture and systems optimizations. |
| [`autoresearch`](autoresearch/) | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Fixed-budget agent experiment loop. |

## What each source contributes

`pathway-bdh` is the reference implementation. Keep its model equations
available as an exact baseline before introducing abstractions.

`nanoGPT` is useful as a control model and as a readable example of a compact
training loop. Its README now marks the project as deprecated in favor of
`nanochat`.

`nanochat` is the best starting point for the reusable harness. It separates
the model, data, optimizer, evaluation, checkpoint, and training concerns
without imposing a large framework.

`modded-nanogpt` is a catalogue, not a clean base. Port one optimization at a
time, and separate algorithmic changes from kernel, communication, and
precision changes.

`autoresearch` supplies the operating loop: a fixed evaluator and time budget,
one editable training file, one committed change per attempt, and a
keep/discard decision based on a measured result.

## Related implementations and attempts

These repositories were found through GitHub search and are not all checked
out locally:

- [pathwaycom/arc-task-gen](https://github.com/pathwaycom/arc-task-gen) —
  official ARC task generation and evaluation support.
- [severian42/BDH-MLX](https://github.com/severian42/BDH-MLX) — MLX port.
- [jploski/bdh-transformers](https://github.com/jploski/bdh-transformers) —
  Transformers-oriented implementation.
- [takzen/BDH-research](https://github.com/takzen/BDH-research) — independent
  research implementation.
- [takzen/bdh-research-emergency](https://github.com/takzen/bdh-research-emergency)
  — independent research branch.
- [promefeus/bdh-architecture-analysis](https://github.com/promefeus/bdh-architecture-analysis)
  — architecture analysis.
- [spandan11106/BDH-Explainer](https://github.com/spandan11106/BDH-Explainer)
  — explanatory implementation.
- [PrimeIntellect-ai/experiments-autonomous-speedrunning](https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning)
  — related autonomous speedrunning experiments.
- [facebookresearch/llm-speedrunner](https://github.com/facebookresearch/llm-speedrunner)
  — training speedrun work.
- [openai/parameter-golf](https://github.com/openai/parameter-golf) — compact
  model and training efficiency experiments.

The exact BDH-CQ model implementation was not found. The public repositories
above should be classified as direct BDH implementations, ports, evaluators,
explainers, or adjacent speedrun work rather than as evidence that BDH-CQ was
released.

## Local snapshot refs

These refs identify the shallow checkouts created for this workspace:

| Repository | Commit |
| --- | --- |
| `pathway-bdh` | `2b0d7a4` |
| `attempt-ssrhaso-bdh` | `8c1173a` |
| `nanoGPT` | `3adf61e` |
| `nanochat` | `92d63d4` |
| `modded-nanogpt` | `ecbb586` |
| `autoresearch` | `228791f` |
