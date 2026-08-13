# Runpod Pod workflow

The lab uses one isolated Runpod Pod for one versioned experiment. This makes
the provider's Pod ID a direct cost attribution key. Do not use a shared Pod
for a comparison that needs an exact cost result.

## Local preparation

`runpodctl` is installed locally. Configure its API key in its user-level
configuration, outside this repository:

```sh
runpodctl doctor
runpodctl version
```

The key is kept by `runpodctl` in `~/.runpod/config.toml`. Do not put it in a
profile, a Just recipe, `.env`, shell history, generated run record, or this
conversation.

The checked-in Runpod profile uses the public official PyTorch 2.8 template
`runpod-torch-v280`. The launcher uses it without a shell variable. You can
inspect the template through the authenticated local client:

```sh
runpodctl template get runpod-torch-v280 --output yaml
just provider runpod doctor
```

To test another public or private template for one invocation, use the
non-secret optional override:

```sh
RUNPOD_TEMPLATE_ID=another-template-id just provider runpod plan \
  experiments/synthetic-associative-recall-fastweight-runpod-v1.yaml
```

The selected template must have Python, `pip`, and SSH access. The launcher
installs the pinned `uv` command only if it is absent, then uses `uv.lock` to
create the remote environment.

## Review before launch

## First paid lifecycle smoke test

Run the Runpod smoke test before either GPU comparison candidate. It uses the
same tiny task as the local smoke test, but runs one CUDA seed on one Pod. It
tests the complete provider path: provisioning, SSH, source transfer, remote
`uv` environment setup, CUDA execution, result retrieval, default Pod
deletion, and cost-record collection. It does not test model quality.

The record is `proposed`, has a 30-minute hard lifetime, and a USD 0.20
projected-cost cap. Current 4090 community pricing was USD 0.34/hour when this
record was added, which projects to USD 0.17 for 30 minutes. The launcher
will delete a newly created Pod if its returned rate exceeds the declared cap.

```sh
just provider runpod plan \
  experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
# Review and change only this record's status from proposed to active.
just provider runpod launch \
  experiments/synthetic-associative-recall-fastweight-runpod-smoke-v1.yaml
just experiment report-experiment synthetic-associative-recall-fastweight-runpod-smoke-v1
```

A successful test has one completed result with `provider: runpod`, `device:
cuda`, a Pod ID and template ID in its provenance, plus one local cost record.
The billing status can initially be `pending`; that still proves that billing
collection was attempted and recorded.

The two initial GPU candidates are intentionally `proposed`:

- `synthetic-associative-recall-transformer-runpod-v1`
- `synthetic-associative-recall-fastweight-runpod-v1`

They use the same task, three seeds, 100 steps per seed, a 45-minute maximum
Pod lifetime, and a USD 2.00 projected-cost cap. A launch creates the Pod only
if Runpod's returned hourly rate fits that cap. The Pod also receives a hard
`--terminate-after` time limit.

Inspect the resolved command and budget first:

```sh
just provider runpod plan experiments/synthetic-associative-recall-transformer-runpod-v1.yaml
just provider runpod plan experiments/synthetic-associative-recall-fastweight-runpod-v1.yaml
```

Promoting a candidate to `active` is a reviewed metadata decision. The
launcher rejects a `proposed` experiment before it reads the template ID or
creates a Pod.

## Launch and evidence retrieval

After promotion, launch one candidate:

```sh
just provider runpod launch experiments/synthetic-associative-recall-fastweight-runpod-v1.yaml
```

The command performs these ordered actions:

1. creates one capped, SSH-ready Pod;
2. checks the returned hourly rate against the declared budget;
3. transfers a source archive without `.env`, Git data, virtual environments,
   or past results;
4. runs the exact versioned experiment remotely with its source revision and
   archive SHA-256 recorded as provenance;
5. retrieves schema-valid results to `var/runs/`; and
6. terminates the Pod by default, then queries its billing history.

Pass `--keep-pod` only for debugging. It weakens final cost attribution,
because the Pod can continue billing after experiment completion. The automatic
hard termination still applies.

## Cost records

The launcher writes `var/runs/<experiment-id>/costs/cost-*.yaml` and a matching
raw JSON billing response. The YAML record is schema-validated and stores the
Pod ID, collection window, USD amount, billed milliseconds, and normalized
hourly billing entries. An empty reply is `pending`; it is never interpreted as
a free run.

If Runpod posts final billing later, append another snapshot:

```sh
just provider runpod cost \
  experiments/synthetic-associative-recall-fastweight-runpod-v1.yaml \
  pod-id-here \
  2026-08-13T10:00:00Z \
  2026-08-13T10:45:00Z
just experiment report-experiment synthetic-associative-recall-fastweight-runpod-v1
```

The report separates `reported_amount_usd` from pending and unavailable
records. It does not convert unavailable billing into an estimate.
