set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments := true

mod check 'justfiles/check.just'
mod experiment 'justfiles/experiment.just'
mod provider 'justfiles/provider.just'
mod metadata 'justfiles/metadata.just'
mod ops 'ops/justfile'

# Show the local command surface.
[group("Discovery")]
default:
    @just --list --list-submodules
