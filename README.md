# BDH Lab

Independent BDH-inspired architecture experiments.

This is not an official Pathway project or a claim to reproduce its internal
BDH implementation. It is a lab for testing one narrow question: can a
persistent, writable inference state provide useful retrieval or adaptation per
unit of cost compared with matched Transformer and other sequence-model
baselines?

## Model of the lab

The repository keeps experimental decisions explicit and reproducible:

- `components/` contains versioned building blocks. A component may be
  `proposed`, `active`, `rejected`, `superseded`, or `retired`.
- `assemblies/` combines exact component versions into one runnable candidate.
- `experiments/` describes a task, assembly, profile, seeds, and metrics.
- `datasets/` describes checksum-pinned validation and golden-evaluation
  samples. Training rejects every example in either held-out split.
- `profiles/` describes execution environments. Profiles contain environment
  variable names, never credentials.
- `var/runs/` contains ignored, append-only run records with resolved inputs
  and metrics. A Runpod Pod also writes a separate immutable billing record;
  a delayed provider reply is shown as pending, never as USD 0.

This separation supports automated search without automated self-deception.
An agent can propose a new component or assembly, schedule its experiment, and
record results. A reviewed metadata change is still required to promote or
reject the candidate.

## First candidates

`transformer-recall-v1` is the causal-Transformer control. It has no writable
state beyond activations and the usual context window.

`bdh-fastweight-recall-v1` is a deliberately small, BDH-inspired hypothesis:
a fixed-size Hebbian fast-weight matrix is updated while a sequence is read and
queried later. It tests persistent state, not the authors' unpublished kernel
or reported benchmarks.

The first task is synthetic associative recall. It is a mechanism test, not a
language-model or reasoning result.

Each candidate evaluates fixed, disjoint validation and golden-evaluation
splits. Validation is for development decisions. Golden evaluation is for the
final report only; a fixed checksum and a separate metric make accidental
training overlap or silent generator changes detectable.

The first GPU comparison is a matched three-seed pair: a causal Transformer
control and the bounded fast-weight candidate. Both use the same task, step
budget, GPU profile, maximum runtime, and USD cap.

## Local quick start

```sh
uv sync --extra dev
just check all
just experiment smoke
```

Use the metadata commands before editing or launching a candidate:

```sh
just experiment list
just experiment plan experiments/synthetic-associative-recall-smoke-v1.yaml
just metadata validate
just experiment report
```

## Runpod preparation

The local integration targets one isolated Runpod Pod per experiment through
`runpodctl`. It does not create a Pod until an explicit launch command is used.

```sh
just provider runpod doctor
just provider runpod plan experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
just provider runpod plan experiments/synthetic-associative-recall-fastweight-runpod-v1.yaml
# after promotion of the proposed record:
just provider runpod launch experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
```

Run the low-cost lifecycle smoke test before any GPU comparison. See
[docs/runpod.md](docs/runpod.md) for the credential-safe setup sequence and
the exact success evidence.

## Current limits

The released BDH code is a useful architectural control, but it does not
provide the persistent state described in the paper. Therefore this project
labels the fast-weight implementation as an independent hypothesis. A passing
synthetic-recall run proves only that the configured code executed and
reported the specified metric.
