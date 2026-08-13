# List known experiments.
[group("Experiments")]
list:
    cd .. && yq -o=json -I=0 '[.id, .status, .assembly_id, .profile_id] | @tsv' experiments/*.yaml | sort

# Resolve one experiment without training or creating a provider resource.
[group("Experiments")]
plan experiment:
    cd .. && uv run bdh-lab plan "{{experiment}}"

# Run one experiment locally. Results are written under var/runs/.
[group("Experiments")]
run experiment:
    cd .. && uv run bdh-lab run "{{experiment}}"

# Run a very small local smoke experiment.
[group("Experiments")]
smoke:
    cd .. && uv run bdh-lab run experiments/synthetic-associative-recall-smoke-v1.yaml

# Summarize completed, failed, and provider-cost records.
[group("Experiments")]
report:
    cd .. && uv run bdh-lab report

# Summarize records for one exact experiment ID.
[group("Experiments")]
report-experiment experiment_id:
    cd .. && uv run bdh-lab report --experiment-id "{{experiment_id}}"
