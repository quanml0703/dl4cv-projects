"""Qwen2-VL model wrapper for humanitarian crisis classification.

Supports:
- 4-bit quantization (NF4 double-quant, recommended for consumer GPUs)
- 8-bit quantization (LLM.int8())
- No quantization (full float16)

Usage example::

    classifier = CrisisClassifier(quantization="4bit")
    messages   = build_zero_shot_messages(tweet_text)
    label, t   = classifier.predict(messages, image=pil_image)
"""

from __future__ import annotations

import copy
import time
from typing import Literal, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError(
        "qwen_vl_utils is required.  "
        "Install with:  pip install qwen-vl-utils"
    ) from exc


# ─────────────────────────────────────────────────────────
# Type alias & constants
# ─────────────────────────────────────────────────────────

QuantizationType = Literal["4bit", "8bit", "none"]

#: Default model identifier on HuggingFace Hub.
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2-VL-2B-Instruct"

#: The 5 valid output labels.
VALID_LABELS: list[str] = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

#: Keyword → label fallback table used when the model output is ambiguous.
#: Ordered from most-specific to most-general to reduce false-positive matches.
_KEYWORD_MAP: dict[str, str] = {
    # ── not_humanitarian (most distinct marker — check first) ─────────────
    "not humanitarian":  "not_humanitarian",
    "not_humanitarian":  "not_humanitarian",
    "opinion":           "not_humanitarian",
    "commentary":        "not_humanitarian",
    "unrelated":         "not_humanitarian",
    # ── rescue_volunteering_or_donation_effort ────────────────────────────
    "rescue":            "rescue_volunteering_or_donation_effort",
    "volunteer":         "rescue_volunteering_or_donation_effort",
    "donat":             "rescue_volunteering_or_donation_effort",
    "fundrais":          "rescue_volunteering_or_donation_effort",
    "relief effort":     "rescue_volunteering_or_donation_effort",
    "distribution":      "rescue_volunteering_or_donation_effort",
    # ── other_relevant_information ────────────────────────────────────────
    "other relevant":    "other_relevant_information",
    "other_relevant":    "other_relevant_information",
    "situational":       "other_relevant_information",
    "evacuat":           "other_relevant_information",
    "weather forecast":  "other_relevant_information",
    "situation report":  "other_relevant_information",
    # ── infrastructure_and_utility_damage ─────────────────────────────────
    "infrastructure":    "infrastructure_and_utility_damage",
    "utility":           "infrastructure_and_utility_damage",
    "vehicle":           "infrastructure_and_utility_damage",
    "structural damage": "infrastructure_and_utility_damage",
    "power outage":      "infrastructure_and_utility_damage",
    "damage":            "infrastructure_and_utility_damage",
    # ── affected_individuals ──────────────────────────────────────────────
    "affected":          "affected_individuals",
    "injur":             "affected_individuals",
    "casualt":           "affected_individuals",
    "fatali":            "affected_individuals",
    "victim":            "affected_individuals",
    "dead":              "affected_individuals",
    "missing":           "affected_individuals",
}


# ─────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────

class CrisisClassifier:
    """Wrapper around Qwen2-VL for humanitarian crisis label prediction.

    Args:
        model_name:   HuggingFace model identifier.
        quantization: Memory quantization level — ``'4bit'``, ``'8bit'``,
                      or ``'none'`` for full float16.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        quantization: QuantizationType = "4bit",
    ) -> None:
        self.model_name   = model_name
        self.quantization = quantization
        self.model:     Optional[Qwen2VLForConditionalGeneration] = None
        self.processor: Optional[AutoProcessor]                   = None
        self._load_model()

    # ── private helpers ──────────────────────────────────

    def _get_bnb_config(self) -> Optional[BitsAndBytesConfig]:
        """Return the appropriate BitsAndBytesConfig, or ``None`` for full precision."""
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _load_model(self) -> None:
        """Download and load the model + processor from HuggingFace Hub."""
        print(
            f"[CrisisClassifier] Loading {self.model_name} "
            f"with {self.quantization} quantization …"
        )
        bnb_config = self._get_bnb_config()
        kwargs: dict = {"device_map": "auto", "torch_dtype": torch.float16}
        if bnb_config:
            kwargs["quantization_config"] = bnb_config

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print("[CrisisClassifier] Model ready.")

    def free(self) -> None:
        """Release GPU memory by deleting the model and clearing the cache.

        Call this before loading a model with a different quantization level.
        """
        import gc

        del self.model
        del self.processor
        self.model     = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[CrisisClassifier] GPU memory released.")

    # ── static helpers ───────────────────────────────────

    @staticmethod
    def _inject_image(
        messages: list[dict],
        image: Image.Image,
    ) -> list[dict]:
        """Inject a PIL image into the LAST user-role message (the query).

        Transforms the plain-text ``content`` string of the final user turn
        into a multimodal list::

            [{"type": "image", "image": <PIL>}, {"type": "text", "text": <str>}]

        Targeting the LAST user message ensures correctness for both:
        - Zero-shot (single user turn)
        - Multi-turn few-shot where earlier user turns already contain images

        A deep copy is made so the original message list is not mutated.

        Args:
            messages: Original message list.
            image:    PIL Image to embed.

        Returns:
            Deep-copied message list with the query image injected.
        """
        msgs = copy.deepcopy(messages)
        # Find the index of the LAST user message
        last_user_idx = None
        for i, msg in enumerate(msgs):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx is not None:
            msg = msgs[last_user_idx]
            original_text = msg["content"]
            # Only inject if content is still a plain string (not already multimodal)
            if isinstance(original_text, str):
                msg["content"] = [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": original_text},
                ]
        return msgs

    @staticmethod
    def _extract_label(text: str) -> Optional[str]:
        """Try to match text to a valid label; return ``None`` if no match.

        Checks in order:
        1. Exact substring match against ``VALID_LABELS``.
        2. Keyword fuzzy match via ``_KEYWORD_MAP``.

        Args:
            text: Candidate string (should be lowercased before passing in).

        Returns:
            A valid label string or ``None``.
        """
        clean = text.strip().lower().replace("-", "_")
        for label in VALID_LABELS:
            if label in clean:
                return label
        for keyword, label in _KEYWORD_MAP.items():
            if keyword in clean:
                return label
        return None

    def _parse_label(self, raw_output: str) -> str:
        """Map raw model output to one of the five valid labels.

        Handles both standard (direct-label) and chain-of-thought (CoT)
        outputs.  Parsing strategy:

        1. If a ``LABEL:`` marker is present (CoT format), extract the text
           after the **last** occurrence and attempt to match it first.
        2. Fall back to matching against the full raw output.
        3. Default to ``'not_humanitarian'`` if no match is found.

        Args:
            raw_output: Decoded generation string from the model.

        Returns:
            One of the five valid label strings.
        """
        lower = raw_output.lower()

        # 1. CoT format — extract text after the last "label:" marker
        if "label:" in lower:
            after_marker = lower.split("label:")[-1]
            result = self._extract_label(after_marker)
            if result:
                return result

        # 2. Full-output standard matching
        result = self._extract_label(lower)
        if result:
            return result

        # 3. Default fallback
        return "not_humanitarian"

    # ── public API ───────────────────────────────────────

    def predict(
        self,
        messages: list[dict],
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 20,
        num_beams: int = 1,
    ) -> tuple[str, float]:
        """Run one inference pass and return the predicted label and timing.

        Args:
            messages:       Message list from ``PromptBuilder`` (text only).
            image:          Optional PIL Image; injected as multimodal input.
            max_new_tokens: Token budget for the generated response.
                            Use ≥ 80 for chain-of-thought prompts.
            num_beams:      Beam-search width.  ``1`` = greedy (default).
                            Values > 1 are not supported with 4-bit NF4
                            quantization.

        Returns:
            ``(predicted_label, inference_time_seconds)`` tuple.
        """
        if image is not None:
            messages = self._inject_image(messages, image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "num_beams":      num_beams,
        }
        if num_beams > 1:
            generate_kwargs["early_stopping"] = True

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        inference_time = time.perf_counter() - t0

        # Trim prompt tokens from the generated output
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_label(raw_text), inference_time
