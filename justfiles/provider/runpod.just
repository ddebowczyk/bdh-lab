# Inspect local Runpod readiness without a network request.
[group("Providers")]
doctor:
    cd ../.. && uv run bdh-lab runpod doctor

# Render the exact Pod creation command without executing it.
[group("Providers")]
plan experiment:
    cd ../.. && uv run bdh-lab runpod plan "{{experiment}}"

# Create a Pod only after the operator explicitly invokes this recipe.
[group("Providers")]
apply experiment:
    cd ../.. && uv run bdh-lab runpod submit "{{experiment}}" --apply

# Create, execute, retrieve, cost, and terminate one isolated Pod.
[group("Providers")]
launch experiment:
    cd ../.. && uv run bdh-lab runpod launch "{{experiment}}" --apply

# Append a billing record after a Pod has ended or after Runpod posts updated billing data.
[group("Providers")]
cost experiment pod_id started_at ended_at:
    cd ../.. && uv run bdh-lab runpod cost "{{experiment}}" --pod-id "{{pod_id}}" --started-at "{{started_at}}" --ended-at "{{ended_at}}"
