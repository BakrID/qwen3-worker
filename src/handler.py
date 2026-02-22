import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

MODEL_PATH = "/model"

engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    dtype="auto",          # picks up FP8 weights automatically
    quantization="fp8",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    enforce_eager=False,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)


def build_prompt(messages: list[dict]) -> str:
    """Apply Qwen3 chat template manually as a fallback."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


async def handler(job):
    job_input: dict = job["input"]

    # Accept either OpenAI-style messages or a raw prompt string
    messages: list[dict] | None = job_input.get("messages")
    prompt: str | None = job_input.get("prompt")

    if messages:
        # Use the tokenizer's chat template when available
        try:
            tokenizer = await engine.get_tokenizer()
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=job_input.get("enable_thinking", False),
            )
        except Exception:
            prompt = build_prompt(messages)
    elif not prompt:
        yield {"error": "Provide either 'messages' or 'prompt' in input."}
        return

    sampling_params = SamplingParams(
        temperature=float(job_input.get("temperature", 0.7)),
        top_p=float(job_input.get("top_p", 0.9)),
        max_tokens=int(job_input.get("max_tokens", 512)),
        repetition_penalty=float(job_input.get("repetition_penalty", 1.0)),
        stop=job_input.get("stop", None),
    )

    request_id = random_uuid()
    stream = engine.generate(prompt, sampling_params, request_id)

    generated_text = ""
    async for request_output in stream:
        token = request_output.outputs[0].text[len(generated_text):]
        generated_text = request_output.outputs[0].text
        yield {"token": token, "text": generated_text, "finished": request_output.finished}


runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": True,
    }
)
