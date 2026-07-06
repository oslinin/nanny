.PHONY: generate-traces grade eval

# Runs nanny's own trace generator (tests/eval/generate_traces.py) rather
# than `agents-cli eval generate`, since nanny's graph is driven by a custom
# input_mode state delta rather than a bare chat prompt -- see that script's
# docstring for why. Writes to artifacts/traces/.
generate-traces:
	uv run python tests/eval/generate_traces.py

# Grades the traces just generated against tests/eval/eval_config.yaml's
# custom LLM-judge metrics (pii_containment, injection_containment). Needs
# GEMINI_API_KEY (or Vertex ADC) configured -- the judge itself is an LLM call.
# --traces/--output are explicit because nanny isn't an agents-cli-scaffolded
# project (no agents-cli-manifest.yaml) for the CLI to auto-discover paths from.
grade:
	uv run agents-cli eval grade --config tests/eval/eval_config.yaml \
		--traces artifacts/traces/ --output artifacts/grade_results/

# Convenience: full local eval loop.
eval: generate-traces grade
