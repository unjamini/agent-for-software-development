## LLM Agent for Fixing Python Code

This project implements a LangGraph-based agent for fixing buggy Python code & evaluates quality on the HumanEvalFix using pass@1.

To install dependencies:

```bash
pip install -e .
```

### Run Evaluation

```bash
# Set PYTHONPATH to include the src directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Run with verbose logging on subsample
python3 -m agent_eval.eval --limit 10 --verbose

# Full evaluation
python3 -m agent_eval.eval
```

The script prints `pass@1` (pass@1 metric) and per-task PASS/FAIL results.
