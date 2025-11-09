from __future__ import annotations

from typing import Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior Python engineer tasked with fixing buggy functions. "
    "Use the tools provided to run snippets, inspect behaviour, and produce a corrected solution. "
    "Always return working Python code that defines the target function along with any helpers required. "
    "Structure your reasoning using the ReAct pattern with 'Thought', 'Action', 'Action Input', and 'Observation'. "
    "Return only fixed code in the 'Final Answer' block. Make sure you import the necessary libraries. Come up with 3 different test cases for the function and run them. If the function is not working or returns the wrong output, you fix the code. If the function is working and returns the correct output, you return the code."
)


def build_bugfix_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_iterations: int = 12,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    agent = create_react_agent(
        model,
        list(tools),
        prompt=prompt,
    )

    return agent


