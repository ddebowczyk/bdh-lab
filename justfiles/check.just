# Run all local checks that do not create a remote resource.
[group("Quality")]
all:
    just --justfile ../justfile --working-directory .. metadata validate
    just --justfile ../justfile --working-directory .. ops research validate
    just --justfile ../justfile --working-directory .. check yaml
    just --justfile ../justfile --working-directory .. check markdown
    just --justfile ../justfile --working-directory .. check lint
    just --justfile ../justfile --working-directory .. check test

# Validate all versioned YAML data and schemas.
[group("Quality")]
yaml:
    cd .. && yamllint components assemblies datasets experiments profiles research schemas ops

# Check all Markdown documentation.
[group("Quality")]
markdown:
    cd .. && rg --files -g '*.md' -0 | xargs -0 markdownlint

# Run Ruff against source and tests.
[group("Quality")]
lint:
    cd .. && uv run ruff check src tests

# Run unit and metadata tests.
[group("Quality")]
test:
    cd .. && uv run pytest
