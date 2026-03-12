"""
asr_refiner.py — FrameProcessor that intercepts ASR transcriptions,
                  calls a lightweight LLM to fix transcription errors,
                  then forwards the corrected text downstream.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

load_dotenv(override=True)

# ── Refinement system prompt ──────────────────────────────────
ASR_REFINEMENT_PROMPT = """You are a text correction assistant. Your ONLY job is to fix
transcription errors from speech-to-text output.

Rules:
- Fix misspellings, misheard words, and broken grammar.
- Do NOT add new information, opinions, or content.
- Do NOT change the user's intent or meaning.
- Do NOT add extra punctuation beyond what's natural.
- If the text is already correct, return it unchanged.
- Return ONLY the corrected text. No explanations, no quotes.

Examples:
  Input:  "show me the sell data for las munth"
  Output: "show me the sales data for last month"

  Input:  "how many customs do we hav"
  Output: "how many customers do we have"
"""


class ASRRefinerProcessor(FrameProcessor):
    """
    Sits between STT and the LLM context aggregator.
    Intercepts TranscriptionFrame → calls LLM to clean up text → forwards corrected frame.
    All other frames pass through untouched.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        name: str = "ASRRefiner",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("NVIDIA_API_KEY", ""),
            base_url=base_url or os.getenv("NIM_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        )
        self._model = model or os.getenv("NIM_LLM_MODEL", "mistralai/ministral-14b-instruct-2512")

        logger.info(f"[ASRRefiner] Initialized with model={self._model}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            original = frame.text
            if original and original.strip():
                logger.debug(f"[ASRRefiner] Original : '{original}'")
                try:
                    refined = await self._refine(original)
                    logger.debug(f"[ASRRefiner] Refined  : '{refined}'")
                    frame.text = refined
                except Exception as e:
                    # On failure → pass original through, never block the pipeline
                    logger.warning(f"[ASRRefiner] Failed, using original: {e}")

        await self.push_frame(frame, direction)

    async def _refine(self, text: str) -> str:
        """Single LLM call to fix ASR output."""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": ASR_REFINEMENT_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,   # deterministic corrections
            max_tokens=256,
            stream=False,      # short text, no need to stream
        )
        refined = resp.choices[0].message.content.strip()
        return refined if refined else text