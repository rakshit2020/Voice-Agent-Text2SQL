"""
bot.py — Voice Database Agent

Pipeline:
  Audio In → Riva ASR → ASR Refiner → LLM Context → NIM LLM (+ tools) → Riva TTS → Audio Out

Run:   python bot.py
Open:  http://localhost:7860 in browser → click Connect → talk
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# ── Pipecat core ──────────────────────────────────────────────
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams, BaseTransport
from pipecat.runner.run import RunnerArguments

# ── Audio / VAD ───────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# ── NVIDIA services (new unified path since v0.0.96+) ────────
# from pipecat.services.nvidia import NvidiaLLMService
# from pipecat.services.nvidia.stt import NvidiaSTTService
# from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
# ── Self-hosted alternatives ──────────────────────────────────
# The SAME classes work for self-hosted. You only change the
# constructor params (server URL, use_ssl, api_key).
# See commented examples below inside bot().

# ── Project modules ───────────────────────────────────────────
from asr_refiner import ASRRefinerProcessor
from tools import TOOL_REGISTRY, shutdown_db

# ── Logging ───────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# ╔══════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION                             ║
# ╚══════════════════════════════════════════════════════════════╝

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NIM_LLM_BASE_URL = os.getenv("NIM_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_LLM_MODEL    = os.getenv("NIM_LLM_MODEL", "openai/gpt-oss-20b")
RIVA_ASR_URL     = os.getenv("RIVA_ASR_URL", "grpc.nvcf.nvidia.com:443")
RIVA_TTS_URL     = os.getenv("RIVA_TTS_URL", "grpc.nvcf.nvidia.com:443")

# ── Database schema metadata (FILL THIS IN with your real schema) ──
DATABASE_SCHEMA = """
### DATABASE SCHEMA ###

-- Table: customers
-- Columns:
--   id            SERIAL PRIMARY KEY
--   name          VARCHAR(255) NOT NULL
--   email         VARCHAR(255) UNIQUE NOT NULL
--   phone         VARCHAR(50)
--   created_at    TIMESTAMP DEFAULT NOW()

-- Table: orders
-- Columns:
--   id            SERIAL PRIMARY KEY
--   customer_id   INTEGER REFERENCES customers(id)
--   order_date    DATE NOT NULL
--   total_amount  NUMERIC(12,2)
--   status        VARCHAR(50) DEFAULT 'pending'   -- pending | shipped | delivered | cancelled

-- Table: products
-- Columns:
--   id            SERIAL PRIMARY KEY
--   name          VARCHAR(255) NOT NULL
--   category      VARCHAR(100)
--   price         NUMERIC(10,2)
--   stock_qty     INTEGER DEFAULT 0

-- Table: order_items
-- Columns:
--   id            SERIAL PRIMARY KEY
--   order_id      INTEGER REFERENCES orders(id)
--   product_id    INTEGER REFERENCES products(id)
--   quantity      INTEGER NOT NULL
--   unit_price    NUMERIC(10,2)

### RELATIONSHIPS ###
-- orders.customer_id  → customers.id
-- order_items.order_id → orders.id
-- order_items.product_id → products.id

### IMPORTANT RULES ###
-- Only generate SELECT queries. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
-- Always use table aliases for clarity.
-- Limit results to 20 rows unless the user explicitly asks for more.
"""

SYSTEM_PROMPT = f"""You are a friendly, professional voice-based database assistant.

Your capabilities:
1. Answer general greetings and small talk naturally and directly.
2. When the user asks a data question, use the `query_database` tool to fetch results.
3. After receiving query results, reframe them into a clear, natural spoken response.

RULES:
- Your output will be spoken aloud via TTS — keep answers conversational and concise.
- Do NOT include raw SQL, code blocks, markdown, or special characters in your spoken response.
- If a query returns no results, say so politely.
- If you are unsure what the user wants, ask a clarifying question.
- Never fabricate data. Only report what the database returns.
- For greetings or general questions, respond directly without using any tools.

You have access to the following database schema:
{DATABASE_SCHEMA}
"""


# ╔══════════════════════════════════════════════════════════════╗
# ║           BOT — called per session by the runner             ║
# ╚══════════════════════════════════════════════════════════════╝

async def bot(runner_args: RunnerArguments):
    """Build and run the voice agent pipeline for one WebRTC session."""

    logger.info("=" * 60)
    logger.info("  Voice Database Agent — new session")
    logger.info("=" * 60)

    # ── 1. Transport ──────────────────────────────────────────
    # The runner injects a SmallWebRTCConnection into runner_args.
    # We need to import and build the transport here.
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,                # ← DEPRECATED
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    min_volume=0.4,
                )
            ),
            vad_audio_passthrough=True,      # ← DEPRECATED
        ),
    )

    # ── 2. Speech-to-Text (NVIDIA Riva ASR) ───────────────────
    stt = NvidiaSTTService(
        api_key=NVIDIA_API_KEY,
        server=RIVA_ASR_URL,
    )
    # ── Self-hosted Riva ASR (uncomment when ready) ───────────
    # stt = NvidiaSTTService(
    #     server="localhost:50051",
    #     use_ssl=False,
    #     # api_key not needed for local Riva
    # )

    # ── 3. ASR Refinement (LLM corrects transcription errors) ─
    asr_refiner = ASRRefinerProcessor(
        api_key=NVIDIA_API_KEY,
        base_url=NIM_LLM_BASE_URL,
        model=NIM_LLM_MODEL,
        # TIP: for faster refinement you can use a smaller model:
        # model="nvidia/llama-3.1-8b-instant",
    )

    # ── 4. Main LLM (NVIDIA NIM) ─────────────────────────────
    llm = NvidiaLLMService(
        api_key=NVIDIA_API_KEY,
        base_url=NIM_LLM_BASE_URL,
        model=NIM_LLM_MODEL,
        temperature=1,
        top_p=0.95,
        max_tokens=1048,
        reasoning_budget=16384,
        chat_template_kwargs={"enable_thinking":False},
    )
    # ── Self-hosted NIM LLM (uncomment when ready) ────────────
    # llm = NvidiaLLMService(
    #     api_key="not-used",
    #     base_url="http://localhost:8000/v1",
    #     model="meta/llama-3.1-70b-instruct",
    # )

    # ── 5. Register tools with the LLM ───────────────────────
    for schema, name, handler in TOOL_REGISTRY:
        llm.register_function(name, handler)
        logger.info(f"Registered tool: {name}")

    tool_schemas = [t[0] for t in TOOL_REGISTRY]

    # ── 6. Text-to-Speech (NVIDIA Riva TTS) ───────────────────
    tts = NvidiaTTSService(
        api_key=NVIDIA_API_KEY,
        server=RIVA_TTS_URL,
        # voice="English-US.Female-1",
        # sample_rate=16000,
    )
    # ── Self-hosted Riva TTS (uncomment when ready) ───────────
    # tts = NvidiaTTSService(
    #     server="localhost:50051",
    #     use_ssl=False,
    # )

    # ── 7. Conversation context + tools ───────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = OpenAILLMContext(messages, tools=tool_schemas)
    context_aggregator = llm.create_context_aggregator(context)

    # ── 8. Pipeline ───────────────────────────────────────────
    #
    #   Audio In → STT → ASR Refiner → User Context → LLM → TTS → Audio Out
    #                                                   ↕
    #                                           Tool calls (DB)
    #
    pipeline = Pipeline([
        transport.input(),              # WebRTC audio in
        stt,                            # Riva ASR: audio → text
        asr_refiner,                    # LLM fixes transcription
        context_aggregator.user(),      # accumulate user turn
        llm,                            # NIM LLM: reason + tool calls
        tts,                            # Riva TTS: text → audio
        transport.output(),             # WebRTC audio out
        context_aggregator.assistant(), # accumulate assistant turn
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,       # user can interrupt bot mid-speech
            enable_metrics=True,            # track latency
            enable_usage_metrics=True,      # track token usage
        ),
    )

    # ── 9. Events ─────────────────────────────────────────────
    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        logger.info("Client connected — sending greeting")
        await task.queue_frames([
            LLMMessagesFrame([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Introduce yourself briefly and ask how you can help."},
            ])
        ])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        logger.info("Client disconnected — cleaning up")
        await shutdown_db()
        await task.cancel()

    # ── 10. Run ───────────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)


# ╔══════════════════════════════════════════════════════════════╗
# ║                       ENTRY POINT                            ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()