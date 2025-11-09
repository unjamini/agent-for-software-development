from __future__ import annotations

from typing import Optional

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.language_models.chat_models import BaseChatModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_qwen_model(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    top_p: float = 0.9,
) -> BaseChatModel:
    """Load a small Qwen instruct model compatible with LangGraph agents.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier
    temperature:
        Sampling temperature for generation.
    max_new_tokens:
        Maximum length for generated completions.
    top_p:
        nucleus sampling parameter.

    Returns
    -------
    BaseChatModel
        LangChain-compatible chat model ready to be used with LangGraph.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_id)

    text_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=top_p,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    hf_llm = HuggingFacePipeline(pipeline=text_pipeline)
    return ChatHuggingFace(llm=hf_llm)


