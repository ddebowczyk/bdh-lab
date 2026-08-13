# Validate versioned schemas and cross-file references.
[group("Metadata")]
validate:
    cd .. && uv run bdh-lab validate

# Print the active component catalogue in a stable machine-readable form.
[group("Metadata")]
components:
    cd .. && yq -o=json -I=0 '[.id, .status, .kind] | @tsv' components/*.yaml | sort
