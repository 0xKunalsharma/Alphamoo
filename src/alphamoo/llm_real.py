"""
AlphaMoo v4.1 — Real LLM Integration (Phase 5).

Replaces the StubLLM with actual Qwen2.5-0.5B-Instruct-AWQ via vLLM.
Same interface as StubLLM so the agent loop doesn't change.

Two modes:
  1. vLLM mode (recommended for Kaggle GPU): high throughput, continuous batching
  2. Transformers mode (fallback): slower but more portable

Usage:
    from alphamoo.llm_real import RealLLM
    llm = RealLLM(model_name="Qwen/Qwen2.5-0.5B-Instruct-AWQ", backend="vllm")
    response = llm.generate(prompt=..., available_actions=...)

On Kaggle:
    - Set accelerator to GPU RTX 6000
    - Internet ON for first run (downloads weights)
    - Internet OFF for submission (weights cached as Kaggle dataset)
"""
from __future__ import annotations

import time

from .llm_stub import LLMResponse

# =============================================================================
# Real LLM — vLLM backend
# =============================================================================

class RealLLM:
    """
    Real LLM using vLLM (preferred) or transformers (fallback).

    Same interface as StubLLM — drop-in replacement.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        backend: str = "vllm",  # "vllm" or "transformers"
        max_output_tokens: int = 80,
        temperature: float = 0.3,
        quantization: str = "awq",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ):
        self.model_name = model_name
        self.backend = backend
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.quantization = quantization
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        self._llm = None
        self._tokenizer = None
        self._sampling_params = None
        self._total_calls = 0
        self._total_wall_clock = 0.0
        self._total_prompt_tokens = 0
        self._total_output_tokens = 0

        self._load_model()

    def _load_model(self) -> None:
        """Load the model using the specified backend."""
        print(f"[RealLLM] Loading {self.model_name} via {self.backend}...")
        t0 = time.perf_counter()

        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        load_time = time.perf_counter() - t0
        print(f"[RealLLM] Loaded in {load_time:.1f}s")

    def _load_vllm(self) -> None:
        """Load via vLLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            ) from e

        self._llm = LLM(
            model=self.model_name,
            quantization=self.quantization if self.quantization != "none" else None,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=False,
            enable_prefix_caching=True,
            trust_remote_code=False,
        )

        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stop=["\n\n", "</think>"],
        )

        # Load tokenizer for token counting
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_transformers(self) -> None:
        """Load via transformers + bitsandbytes (fallback)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=bnb_config, device_map="auto",
            )
        else:
            self._llm = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", torch_dtype=torch.float16,
            )

    def generate(
        self,
        prompt: str,
        available_actions: list[int],
        agent_position: tuple[int, int] | None = None,
        perceived_objects: list[dict] | None = None,
    ) -> LLMResponse:
        """
        Generate an LLM response. Same interface as StubLLM.
        """
        self._total_calls += 1
        t0 = time.perf_counter()

        # Count prompt tokens
        prompt_tokens = len(self._tokenizer.encode(prompt)) if self._tokenizer else len(prompt) // 4

        # Generate
        if self.backend == "vllm":
            text, output_tokens = self._generate_vllm(prompt)
        else:
            text, output_tokens = self._generate_transformers(prompt)

        wall_clock = time.perf_counter() - t0
        self._total_wall_clock += wall_clock
        self._total_prompt_tokens += prompt_tokens
        self._total_output_tokens += output_tokens

        # Parse action from response
        action_id, click_coords = self._parse_action(text, available_actions)

        return LLMResponse(
            text=text,
            action_id=action_id,
            click_coords=click_coords,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            wall_clock_sec=wall_clock,
            reasoning=text,
        )

    def _generate_vllm(self, prompt: str) -> tuple[str, int]:
        """Generate via vLLM."""
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self._llm.generate([formatted], self._sampling_params)
        output = outputs[0].outputs[0]
        text = output.text
        output_tokens = len(output.token_ids)
        return text, output_tokens

    def _generate_transformers(self, prompt: str) -> tuple[str, int]:
        """Generate via transformers."""
        import torch

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._llm.device)

        with torch.no_grad():
            outputs = self._llm.generate(
                **inputs,
                max_new_tokens=self.max_output_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Extract only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, len(new_tokens)

    def _parse_action(
        self, text: str, available_actions: list[int]
    ) -> tuple[int, tuple[int, int] | None]:
        """
        Parse the LLM's response to extract an action ID.

        Expects the response to contain an action number (1-7) or a click
        with coordinates. Falls back to a random valid action if parsing fails.
        """
        import re

        # Look for "Action: N" pattern
        match = re.search(r"[Aa]ction[:\s]+(\d)", text)
        if match:
            action_id = int(match.group(1))
            if action_id in available_actions:
                # Check for click coordinates
                if action_id == 6:
                    coords_match = re.search(r"\((\d+),\s*(\d+)\)", text)
                    if coords_match:
                        return action_id, (int(coords_match.group(1)), int(coords_match.group(2)))
                return action_id, None

        # Look for any digit 1-7 in the response
        for char in text:
            digit = int(char) if char.isdigit() else None
            if digit is not None and digit in available_actions:
                return digit, None

        # Fallback: pick first available action
        return available_actions[0] if available_actions else 0, None

    def get_stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_wall_clock_sec": self._total_wall_clock,
            "avg_wall_clock_ms": (self._total_wall_clock / max(1, self._total_calls)) * 1000,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_output_tokens": self._total_output_tokens,
            "avg_prompt_tokens": self._total_prompt_tokens / max(1, self._total_calls),
            "avg_output_tokens": self._total_output_tokens / max(1, self._total_calls),
            "model_name": self.model_name,
            "backend": self.backend,
            "throughput_tokens_per_sec": (
                self._total_output_tokens / max(0.001, self._total_wall_clock)
            ),
        }


# =============================================================================
# Factory function
# =============================================================================

def create_llm(
    use_real: bool = False,
    latency_model: str = "qwen2.5-0.5b-4bit",
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    backend: str = "vllm",
    max_output_tokens: int = 80,
) -> object:
    """
    Factory: create a real LLM or a stub LLM.

    Args:
        use_real: if True, use RealLLM (requires GPU + vLLM/transformers)
        latency_model: stub latency model name (if use_real=False)
        model_name: HuggingFace model name (if use_real=True)
        backend: "vllm" or "transformers" (if use_real=True)
        max_output_tokens: max tokens to generate per call

    Returns:
        LLM instance (RealLLM or StubLLM) with same interface.
    """
    if use_real:
        return RealLLM(
            model_name=model_name,
            backend=backend,
            max_output_tokens=max_output_tokens,
        )
    else:
        from .llm_stub import StubLLM
        return StubLLM(
            latency_model=latency_model,
            max_output_tokens=max_output_tokens,
            seed=42,
        )
