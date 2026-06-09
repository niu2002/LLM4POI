#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import uuid
from typing import Any, List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 64


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
    max_tokens: int = 64


def create_app(model, processor, device: str) -> FastAPI:
    app = FastAPI()
    tokenizer = processor.tokenizer

    def generate_from_messages(messages: List[dict], max_tokens: int) -> str:
        normalized = [
            {"role": msg["role"], "content": message_content_to_text(msg["content"])}
            for msg in messages
        ]
        prompt = tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        generated = outputs[:, inputs.input_ids.shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def generate_from_prompt(prompt: str, max_tokens: int) -> str:
        inputs = processor(text=[prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        generated = outputs[:, inputs.input_ids.shape[1] :]
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "local-qwen3.5-2b", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        text = generate_from_messages([m.model_dump() for m in req.messages], req.max_tokens)
        now = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        text = generate_from_prompt(req.prompt, req.max_tokens)
        now = int(time.time())
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": now,
            "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    app = create_app(model, processor, device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
