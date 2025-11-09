from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .agent_graph import build_bugfix_agent
from .data import load_humanevalfix
from .model_loader import load_qwen_model
from .python_tool import create_python_tool, PythonSandbox


SYSTEM_TASK_TEMPLATE = (
    "You are given a Python programming task and a buggy implementation. "
    "Your goal is to fix the function so that it passes the provided tests.\n\n"
    "Task:\n{prompt}\n\n"
    "Buggy implementation:\n```python\n{buggy}\n```\n\n"
    "Entry point function: {entry_point}.\n\n"
    "Make sure you import the necessary libraries. Come up with 3 different test cases for the function and run them. If the function is not working or returns the wrong output, you fix the code. If the function is working and returns the correct output, you return the code."
    "Provide the fixed Python code. Return only the corrected function code and import statements, do not add any other text, test cases. Follow the signature of the inital buggy function."
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _extract_code_from_response(response_text: str) -> str:
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    lines = response_text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("import ") or line.strip().startswith("from "):
            in_code = True
        if in_code:
            code_lines.append(line)
        if in_code and line.strip() and not line.startswith(" ") and not line.startswith("\t") and not line.strip().startswith("def "):
            if code_lines and code_lines[-1].strip():
                break
    
    if code_lines:
        return "\n".join(code_lines).strip()
    
    return response_text.strip()


def _extract_final_answer(agent_response: Dict) -> str:
    messages = agent_response.get("messages", [])
    if not messages:
        if isinstance(agent_response, str):
            return _extract_code_from_response(agent_response)
        text = str(agent_response.get("final_output", agent_response.get("output", "")))
        if text:
            return _extract_code_from_response(text)
        return ""
    
    from langchain_core.messages import AIMessage
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return _extract_code_from_response(content)
        elif hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return _extract_code_from_response(content)
        elif isinstance(msg, dict):
            msg_type = msg.get("type", "") or msg.get("__class__", "")
            if "ai" in str(msg_type).lower() or "assistant" in str(msg_type).lower():
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    return _extract_code_from_response(content)
    
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            content = str(last_msg.content)
            return _extract_code_from_response(content)
        elif isinstance(last_msg, dict):
            content = str(last_msg.get("content", ""))
            return _extract_code_from_response(content)
    
    return ""


def _run_tests(solution_code: str, test_code: str, sandbox: PythonSandbox) -> Tuple[bool, str]:
    program = f"{solution_code}\n\n{test_code}"
    out = sandbox(program)
    ok = not out.startswith("Error (") and "Traceback" not in out
    return ok, out


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    details: str
    solution: str


def run_eval(
    *,
    model: BaseChatModel,
    limit: int | None = None,
    verbose: bool | None = False,
) -> Tuple[float, List[EvalResult]]:
    tool = create_python_tool(timeout_seconds=10)
    agent = build_bugfix_agent(model=model, tools=[tool])
    data = load_humanevalfix()

    if limit is not None:
        data = data[:limit]

    results: List[EvalResult] = []
    sandbox = PythonSandbox(timeout_seconds=10)

    print(f"Evaluating {len(data)} tasks...")
    for idx, ex in enumerate(data, 1):
        task_id = str(ex.get("task_id", f"task_{idx}"))
        prompt = str(ex.get("prompt", ""))
        buggy = str(ex.get("buggy_solution", ""))
        tests = str(ex.get("test", ""))
        entry_point = str(ex.get("entry_point", ""))
        canonical = str(ex.get("canonical_solution", ""))

        print(f"[{idx}/{len(data)}] Processing {task_id}...", flush=True)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"TASK: {task_id}")
            print(f"{'='*70}")
            print(f"Entry Point: {entry_point}")
            print(f"\nPrompt:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")
            print(f"\nBuggy Solution:\n{buggy}")
            if canonical:
                print(f"\nCanonical Solution (reference):\n{canonical[:500]}{'...' if len(canonical) > 500 else ''}")

        user_text = SYSTEM_TASK_TEMPLATE.format(
            prompt=prompt, buggy=buggy, entry_point=entry_point
        )

        try:
            resp = agent.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config={"recursion_limit": 12}
            )
            
            if verbose:
                print(f"Agent response received. {resp}")
            
            solution = _extract_final_answer(resp)
            if not solution:
                print(f"  Warning: No solution extracted for {task_id}")
                results.append(EvalResult(task_id=task_id, passed=False, details="No solution extracted", solution=""))
                continue
            
            if verbose:
                print(f"\nExtracted Solution:\n{solution}")
                print(f"\nTest Code:\n{tests[:500]}{'...' if len(tests) > 500 else ''}")
            
            passed, details = _run_tests(solution, tests, sandbox)
            status = "PASS" if passed else "FAIL"
            print(f"  {status}", flush=True)
            
            if verbose:
                print(f"\nTest Execution Output:")
                print(f"{details}")
                if not passed:
                    print(f"\nTest FAILED for {task_id}")
                else:
                    print(f"\nTest PASSED for {task_id}")
                print(f"{'='*70}\n")
            
            results.append(EvalResult(task_id=task_id, passed=passed, details=details, solution=solution))
        except Exception as e:
            error_msg = f"Exception: {e}"
            print(f"  ERROR: {e}", flush=True)
            if verbose:
                import traceback
                print(f"\nException occurred for {task_id}:")
                print(f"Exception type: {type(e).__name__}")
                print(f"Exception message: {str(e)}")
                print(f"\nFull traceback:")
                traceback.print_exc()
                print(f"{'='*70}\n")
                error_msg = f"Exception: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            results.append(EvalResult(task_id=task_id, passed=False, details=error_msg, solution=""))

    pass_at_1 = sum(1 for r in results if r.passed) / max(1, len(results))
    return pass_at_1, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HumanEvalFix eval. Pass limit to eval on subset"
    )
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of tasks for quick runs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    args = parser.parse_args()

    model = load_qwen_model(MODEL_ID)
    try:
        score, results = run_eval(model=model, limit=args.limit, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Pass@1: {score:.4f} ({sum(1 for r in results if r.passed)}/{len(results)})")
    print(f"\nTask Results:")
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status} - {r.task_id}")
        if not r.passed:
            if args.verbose:
                print(f"    Error: {r.details[:500]}")
            else:
                error_summary = r.details.split("\n")[0] if r.details else "No details"
                print(f"    Error: {error_summary[:200]}")
    print("="*70)
    
    if args.verbose:
        print(f"\nVerbose Summary:")
        print(f"  Total tasks: {len(results)}")
        print(f"  Passed: {sum(1 for r in results if r.passed)}")
        print(f"  Failed: {sum(1 for r in results if not r.passed)}")
        print(f"  Pass rate: {score:.2%}")


if __name__ == "__main__":
    main()


