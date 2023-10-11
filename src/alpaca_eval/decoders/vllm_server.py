import logging
from typing import Sequence
import openai

import numpy as np
from tqdm import tqdm
from .. import utils

__all__ = ["vllm_server"]

llm = None
llmModelName = None

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "functionary" # We just need to set this something other than None, so it works with openai package. No API key is required.


def generate(query, temperature, max_tokens):
    response = openai.ChatCompletion.create(
        model="../functionary-13b",
        messages=[{"role": "user", "content": query}],
        max_tokens=max_tokens,
        temperature = temperature,
        functions=[]
    )
    response_message = response["choices"][0]["message"]
    if "functiona_call" in response_message:
        breakpoint()
    return response_message

def vllm_server_completions(
    prompts: Sequence[str],
    model_name: str,
    max_new_tokens: int,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs=None,
    **kwargs,
) -> dict[str, list]:
    completions = []
    responses_vllm = []
    with utils.Timer() as t:
        for i, query in tqdm(enumerate(prompts)):
            outputs = generate(query, kwargs["temperature"], max_new_tokens)
            completions.append(outputs["content"])
            responses_vllm.append(str(outputs))
    price = [np.nan] * len(completions)
    avg_time = [t.duration / len(prompts)] * len(completions)
    return dict(completions=completions, responses_vllm =responses_vllm, price_per_example=price, time_per_example=avg_time)
