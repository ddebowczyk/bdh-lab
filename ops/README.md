# Repository operations

`ops/` is the lab control plane. It owns repository maintenance and provider
planning. It is not a model component and model code must not import it. Its
capability layout follows the small validated-manifest pattern used by the
requested Intercom and Shipctl references.

Each operation capability has a small YAML manifest and a thin Justfile. The
manifest records owned paths, important inputs, generated outputs, and exposed
commands. `ops.yaml` explicitly selects the active provider for each operational
interface.

Use these commands:

```sh
just ops
just ops validate
just ops check all
just ops runpod doctor
just ops runpod plan <experiment>
just ops runpod launch <experiment>
```

The `runpod` capability can render a plan and append billing evidence without a
Pod write. It can create a Pod only when the caller uses its explicit `apply`
or `launch` command. `launch` retrieves evidence, captures a Pod-specific cost
record, and terminates the Pod by default. Credentials remain outside the
checkout.
