#!/usr/bin/env python3
"""
Local-folder voice-call usage analyzer.

No S3 is used.

Pipeline
--------
local MP3/WAV/etc.
    -> faster-whisper (local STT)
    -> Amazon Bedrock LLM (speaker-role + RAG/tool scenario classification)
    -> provider-neutral usage calculations
    -> CSV

The Bedrock calls made by THIS offline analyzer are not counted as the
production voice-agent LLM usage. The CSV estimates what the production
voice agent would consume.

Requirements
------------
Python 3.10+
ffmpeg installed and available in PATH
AWS credentials configured locally for Bedrock

Environment
-----------
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=<your Bedrock model or inference profile id>
INPUT_FOLDER=./recordings
MIN_AUDIO_DURATION_SECONDS=60
TRANSCRIPT_FOLDER=
# For speaker-labeled TXT files, map the labels after checking one sample.
TRANSCRIPT_AGENT_LABEL=Speaker-1
TRANSCRIPT_CUSTOMER_LABEL=Speaker-2
TRANSCRIPT_UNKNOWN_LABEL=Speaker-?
# keywords = transparent transcript-only candidates; llm = Bedrock heuristic;
# none = do not model RAG/tool usage.
SCENARIO_CLASSIFIER_MODE=keywords

Optional usage assumptions:
TOKENS_PER_WORD=1.30
SYSTEM_PROMPT_TOKENS_PER_LLM_CALL=700
TOOL_SCHEMA_TOKENS_PER_LLM_CALL=600
RAG_CONTEXT_TOKENS_PER_QUERY=700
TOOL_RESULT_TOKENS_PER_CALL=150
TOOL_CALL_OUTPUT_TOKENS_PER_CALL=40
RAG_QUERY_EMBEDDING_TOKENS=20

Example
-------
python analyze_voice_calls_local.py \
    --output-csv ./voice_call_usage.csv

The input folder is read from INPUT_FOLDER in .env. You can still override it
for one run with --input-folder.

For subfolders:
python analyze_voice_calls_local.py \
    --output-csv ./voice_call_usage.csv \
    --recursive
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydub.utils import mediainfo


load_dotenv(Path(__file__).with_name(".env"))


SUPPORTED_AUDIO = {
    ".mp3", ".wav", ".m4a", ".mp4", ".flac",
    ".ogg", ".webm", ".aac", ".wma"
}

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")
INPUT_FOLDER = os.getenv("INPUT_FOLDER", "./recordings")
MIN_AUDIO_DURATION_SECONDS = float(
    os.getenv("MIN_AUDIO_DURATION_SECONDS", "60")
)
TRANSCRIPT_FOLDER = os.getenv("TRANSCRIPT_FOLDER", "").strip()
TRANSCRIPT_AGENT_LABEL = os.getenv(
    "TRANSCRIPT_AGENT_LABEL", "Speaker-1"
).strip()
TRANSCRIPT_CUSTOMER_LABEL = os.getenv(
    "TRANSCRIPT_CUSTOMER_LABEL", "Speaker-2"
).strip()
TRANSCRIPT_UNKNOWN_LABEL = os.getenv(
    "TRANSCRIPT_UNKNOWN_LABEL", "Speaker-?"
).strip()
SCENARIO_CLASSIFIER_MODE = os.getenv(
    "SCENARIO_CLASSIFIER_MODE", "keywords"
).strip().lower()

TOKENS_PER_WORD = float(os.getenv("TOKENS_PER_WORD", "1.30"))
SYSTEM_PROMPT_TOKENS_PER_LLM_CALL = int(
    os.getenv("SYSTEM_PROMPT_TOKENS_PER_LLM_CALL", "700")
)
TOOL_SCHEMA_TOKENS_PER_LLM_CALL = int(
    os.getenv("TOOL_SCHEMA_TOKENS_PER_LLM_CALL", "600")
)
RAG_CONTEXT_TOKENS_PER_QUERY = int(
    os.getenv("RAG_CONTEXT_TOKENS_PER_QUERY", "700")
)
TOOL_RESULT_TOKENS_PER_CALL = int(
    os.getenv("TOOL_RESULT_TOKENS_PER_CALL", "150")
)
TOOL_CALL_OUTPUT_TOKENS_PER_CALL = int(
    os.getenv("TOOL_CALL_OUTPUT_TOKENS_PER_CALL", "40")
)
RAG_QUERY_EMBEDDING_TOKENS = int(
    os.getenv("RAG_QUERY_EMBEDDING_TOKENS", "20")
)

CSV_FIELDS = [
    "file_name",
    "file_path",
    "status",
    "error",
    "transcript_source",
    "speaker_label_source",
    "unknown_speaker_segments",
    "speech_duration_basis",
    "scenario_classifier_method",

    "call_duration_seconds",
    "call_duration_minutes",
    "audio_active_speech_seconds",
    "silence_seconds_est",

    "whisper_language",
    "whisper_language_probability",
    "whisper_segments",

    "customer_speech_seconds_est",
    "customer_speech_minutes_est",
    "agent_speech_seconds_est",
    "agent_speech_minutes_est",

    "customer_words",
    "customer_text_tokens_est",
    "agent_words",
    "agent_text_tokens_est",
    "total_spoken_words",
    "total_spoken_tokens_est",

    # STT usage: keep both possibilities.
    "stt_full_audio_seconds",
    "stt_full_audio_minutes",
    "stt_customer_audio_seconds_est",
    "stt_customer_audio_minutes_est",

    # Production voice-agent LLM estimate.
    "llm_calls_est",
    "system_prompt_tokens_total",
    "conversation_history_tokens_total_est",
    "tool_schema_tokens_total",
    "rag_queries_est",
    "rag_context_tokens_total",
    "tool_calls_est",
    "tool_result_tokens_total",
    "llm_input_tokens_total_est",
    "llm_response_tokens_total_est",
    "tool_call_output_tokens_total_est",
    "llm_output_tokens_total_est",

    # TTS.
    "tts_text_tokens_est",
    "tts_characters",
    "tts_audio_seconds_est",
    "tts_audio_minutes_est",

    # RAG.
    "rag_embedding_tokens_est",

    # Audit/debug.
    "bedrock_model_id_used_for_offline_analysis",
    "rag_turn_indices_est",
    "tool_turn_indices_est",
    "bedrock_offline_input_tokens",
    "bedrock_offline_output_tokens",
    "analysis_notes",
]


def round3(value: float) -> float:
    return round(float(value), 3)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def estimate_tokens(text: str) -> int:
    # Provider-neutral approximation rather than a vendor tokenizer.
    return int(math.ceil(word_count(text) * TOKENS_PER_WORD))


def get_audio_metrics(path: Path) -> dict[str, float]:
    audio = AudioSegment.from_file(path)
    duration_seconds = len(audio) / 1000.0

    if len(audio) == 0:
        return {
            "duration_seconds": 0.0,
            "active_speech_seconds": 0.0,
        }

    # Dynamic silence threshold works reasonably well for telephone calls.
    if math.isinf(audio.dBFS):
        silence_thresh = -50.0
    else:
        silence_thresh = max(-55.0, audio.dBFS - 18.0)

    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=400,
        silence_thresh=silence_thresh,
        seek_step=10,
    )

    active_seconds = sum(end - start for start, end in nonsilent) / 1000.0
    active_seconds = min(active_seconds, duration_seconds)

    return {
        "duration_seconds": duration_seconds,
        "active_speech_seconds": active_seconds,
    }


def get_audio_duration_seconds(path: Path) -> float:
    """Read container duration without decoding the entire audio stream."""
    duration = mediainfo(str(path)).get("duration")
    if duration in (None, ""):
        raise RuntimeError("Audio container did not provide a duration.")
    return float(duration)


def find_transcript(path: Path) -> Path | None:
    """Find a transcript matching an audio file by filename stem."""
    candidates = [path.with_suffix(".txt")]

    if TRANSCRIPT_FOLDER:
        transcript_folder = Path(TRANSCRIPT_FOLDER)
        candidates.extend(
            [
                transcript_folder / f"{path.stem}.txt",
                transcript_folder / f"{path.name}.txt",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def transcribe_from_text(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a line-by-line transcript into Whisper-compatible segments.

    If a line starts with ``Speaker-1:`` or another ``label:`` prefix, keep
    the label separately so it can be mapped directly to AGENT/CUSTOMER.
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    segments: list[dict[str, Any]] = []
    cursor = 0.0

    for index, line in enumerate(lines, start=1):
        match = re.match(r"^\s*([^:]{1,80})\s*:\s*(.*?)\s*$", line)
        speaker_label = None
        segment_text = line
        if match:
            speaker_label = match.group(1).strip()
            segment_text = match.group(2).strip()

        # Approximate timing is only used for speech-duration estimates. The
        # transcript text remains unchanged so speaker labels are preserved.
        duration = max(0.25, word_count(segment_text) / 2.5)
        segments.append(
            {
                "segment_id": index,
                "start": cursor,
                "end": cursor + duration,
                "duration": duration,
                "text": segment_text,
                "speaker_label": speaker_label,
            }
        )
        cursor += duration

    return segments, {
        "language": "transcript",
        "language_probability": 1.0,
        "source": "txt",
    }


def labels_from_transcript(
    segments: list[dict[str, Any]],
) -> dict[int, str] | None:
    """Map configured TXT speaker labels without using an LLM guess.

    Return None when the transcript is not fully speaker-labeled or contains
    an unrecognized label, allowing the caller to use the legacy fallback.
    Explicit unknown labels such as ``Speaker-?`` are retained as UNKNOWN.
    """
    if (
        not segments
        or not TRANSCRIPT_AGENT_LABEL
        or not TRANSCRIPT_CUSTOMER_LABEL
    ):
        return None

    configured = {
        TRANSCRIPT_AGENT_LABEL.casefold(): "AGENT",
        TRANSCRIPT_CUSTOMER_LABEL.casefold(): "CUSTOMER",
    }
    if len(configured) != 2:
        return None

    labels: dict[int, str] = {}
    for segment in segments:
        raw_label = str(segment.get("speaker_label") or "").strip()
        role = configured.get(raw_label.casefold())
        if (
            role is None
            and TRANSCRIPT_UNKNOWN_LABEL
            and raw_label.casefold() == TRANSCRIPT_UNKNOWN_LABEL.casefold()
        ):
            role = "UNKNOWN"
        if role is None:
            return None
        labels[int(segment["segment_id"])] = role

    return labels


def transcribe_local(
    path: Path,
    whisper_model: WhisperModel,
    language: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    segments_iter, info = whisper_model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 400,
        },
        condition_on_previous_text=True,
    )

    segments: list[dict[str, Any]] = []
    for i, seg in enumerate(segments_iter, start=1):
        text = (seg.text or "").strip()
        if not text:
            continue

        segments.append(
            {
                "segment_id": i,
                "start": float(seg.start),
                "end": float(seg.end),
                "duration": max(0.0, float(seg.end) - float(seg.start)),
                "text": text,
            }
        )

    metadata = {
        "language": getattr(info, "language", ""),
        "language_probability": float(
            getattr(info, "language_probability", 0.0) or 0.0
        ),
    }
    return segments, metadata


def extract_json(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        return json.loads(value[start:end + 1])

    raise ValueError("Bedrock response did not contain a valid JSON object.")


def call_bedrock_json(
    bedrock_client: Any,
    prompt: str,
    max_tokens: int = 4000,
) -> tuple[dict[str, Any], dict[str, int]]:
    response = bedrock_client.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
        inferenceConfig={
            "maxTokens": max_tokens,
            "temperature": 0,
        },
    )

    content = (
        response.get("output", {})
        .get("message", {})
        .get("content", [])
    )
    text = "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and "text" in block
    )

    usage = response.get("usage", {}) or {}
    return extract_json(text), {
        "input_tokens": int(usage.get("inputTokens", 0) or 0),
        "output_tokens": int(usage.get("outputTokens", 0) or 0),
    }


def classify_segments_with_bedrock(
    segments: list[dict[str, Any]],
    bedrock_client: Any,
) -> tuple[dict[int, str], dict[str, int]]:
    """
    Whisper does not perform speaker diarization. Bedrock uses the semantics
    and order of the transcript to estimate whether each Whisper segment was
    spoken by AGENT or CUSTOMER.

    This is an estimate. For exact speaker diarization, use dual-channel audio
    or a dedicated diarization system.
    """
    if not segments:
        return {}, {"input_tokens": 0, "output_tokens": 0}

    lines = [
        (
            f'Segment {s["segment_id"]} '
            f'[{s["start"]:.2f}-{s["end"]:.2f}s]: {s["text"]}'
        )
        for s in segments
    ]

    prompt = f"""
You are labeling a two-party customer-service telephone conversation.

The transcript was generated from a mono recording by speech-to-text.
There are two roles:
- AGENT: employee/customer-service representative
- CUSTOMER: person calling for help/service

Infer the speaker of EACH transcript segment from dialogue semantics,
question/answer flow, greetings, identity/account questions, support actions,
and conversational continuity.

Important:
- Preserve every segment_id exactly.
- Return one role for every segment.
- Do not omit segments.
- Do not invent extra segment IDs.
- If one STT segment appears to contain words from both speakers, choose the
  role responsible for most of the segment.

Return JSON ONLY:
{{
  "segments": [
    {{"segment_id": 1, "role": "AGENT"}},
    {{"segment_id": 2, "role": "CUSTOMER"}}
  ]
}}

TRANSCRIPT
----------
{chr(10).join(lines)}
""".strip()

    result, usage = call_bedrock_json(
        bedrock_client,
        prompt,
        max_tokens=max(1200, len(segments) * 20 + 500),
    )

    valid_ids = {int(s["segment_id"]) for s in segments}
    labels: dict[int, str] = {}

    for item in result.get("segments", []):
        try:
            seg_id = int(item.get("segment_id"))
        except (TypeError, ValueError):
            continue

        role = str(item.get("role", "")).upper().strip()
        if seg_id in valid_ids and role in {"AGENT", "CUSTOMER"}:
            labels[seg_id] = role

    missing = sorted(valid_ids - set(labels))
    if missing:
        raise RuntimeError(
            "Bedrock did not label every STT segment. Missing segment IDs: "
            + ", ".join(map(str, missing[:20]))
        )

    return labels, usage


def build_turns(
    segments: list[dict[str, Any]],
    labels: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Merge consecutive Whisper segments with the same inferred role into
    conversation turns.
    """
    turns: list[dict[str, Any]] = []

    for seg in segments:
        role = labels[int(seg["segment_id"])]
        item = {
            "role": role,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "duration": float(seg["duration"]),
            "text": seg["text"],
            "segment_ids": [int(seg["segment_id"])],
        }

        if turns and turns[-1]["role"] == role:
            turns[-1]["end"] = item["end"]
            turns[-1]["duration"] += item["duration"]
            turns[-1]["text"] = (
                turns[-1]["text"] + " " + item["text"]
            ).strip()
            turns[-1]["segment_ids"].extend(item["segment_ids"])
        else:
            turns.append(item)

    for idx, turn in enumerate(turns, start=1):
        turn["turn_index"] = idx
        turn["words"] = word_count(turn["text"])
        turn["tokens_est"] = estimate_tokens(turn["text"])

    return turns


def classify_rag_tools_with_bedrock(
    turns: list[dict[str, Any]],
    bedrock_client: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    if not turns:
        return {
            "rag_turn_indices": [],
            "tool_turn_indices": [],
            "analysis_notes": "No speech turns.",
        }, {"input_tokens": 0, "output_tokens": 0}

    if SCENARIO_CLASSIFIER_MODE == "none":
        return {
            "rag_turn_indices": [],
            "tool_turn_indices": [],
            "analysis_notes": (
                "No RAG/tool candidates modeled; production logs were not available."
            ),
        }, {"input_tokens": 0, "output_tokens": 0}

    if SCENARIO_CLASSIFIER_MODE == "keywords":
        return classify_rag_tools_with_keywords(turns)

    lines = [
        f'Turn {t["turn_index"]} | {t["role"]}: {t["text"]}'
        for t in turns
    ]

    prompt = f"""
You are estimating the architecture usage of a production customer-service
voice agent from a historical human/agent phone-call transcript.

Consider ONLY AGENT turns.

RAG:
Mark an AGENT turn as requiring RAG when producing that answer would
reasonably require retrieval from a knowledge base, policy manual, FAQ,
product/service documentation, eligibility/rules documentation,
troubleshooting documentation, or other unstructured/semi-structured
knowledge source.

Do NOT count:
- greetings
- thank-yous
- simple confirmations
- conversational filler
- facts already fully established in the immediate dialogue

TOOLS:
Mark an AGENT turn as requiring an external tool/API when producing or
carrying out the response would reasonably require operations such as:
- customer/account lookup
- order/reservation/case lookup
- status lookup
- CRM read/write
- ticket creation/update
- scheduling
- cancellation
- refund/payment operation
- backend data update
- another transactional API/function

RAG and TOOL are independent. One turn can require both.

This is scenario estimation only. Do not claim the historical human agent
actually used these systems.

Return JSON ONLY:
{{
  "rag_turn_indices": [2, 6],
  "tool_turn_indices": [4, 8],
  "analysis_notes": "one short sentence"
}}

Use only existing AGENT turn indices.

TRANSCRIPT
----------
{chr(10).join(lines)}
""".strip()

    result, usage = call_bedrock_json(
        bedrock_client,
        prompt,
        max_tokens=1200,
    )

    valid_agent_turns = {
        int(t["turn_index"])
        for t in turns
        if t["role"] == "AGENT"
    }

    def clean_indices(value: Any) -> list[int]:
        cleaned = set()
        for x in value or []:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if i in valid_agent_turns:
                cleaned.add(i)
        return sorted(cleaned)

    return {
        "rag_turn_indices": clean_indices(
            result.get("rag_turn_indices", [])
        ),
        "tool_turn_indices": clean_indices(
            result.get("tool_turn_indices", [])
        ),
        "analysis_notes": str(
            result.get("analysis_notes", "")
        )[:500],
    }, usage


def classify_rag_tools_with_keywords(
    turns: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Find transparent candidate signals in AGENT text.

    These are not claims that the historical call used RAG or tools. They are
    conservative scenario markers derived only from the TXT transcript.
    """
    rag_pattern = re.compile(
        r"\b(policy|eligible|eligibility|return|exchange|how do|what is|"
        r"product|documentation|faq|troubleshoot|troubleshooting|pricing|"
        r"coverage|rule|rules|requirement|requirements)\b",
        flags=re.I,
    )
    tool_pattern = re.compile(
        r"\b(account|order|case|ticket|crm|look(?:up|\s+up)|invoice|"
        r"payment|card|refund|schedule|cancel|update|ship|address|"
        r"purchase|transaction|status|booking|reservation)\b",
        flags=re.I,
    )

    rag_turns: list[int] = []
    tool_turns: list[int] = []
    for turn in turns:
        if turn["role"] != "AGENT":
            continue
        text = str(turn["text"])
        if rag_pattern.search(text):
            rag_turns.append(int(turn["turn_index"]))
        if tool_pattern.search(text):
            tool_turns.append(int(turn["turn_index"]))

    return {
        "rag_turn_indices": sorted(set(rag_turns)),
        "tool_turn_indices": sorted(set(tool_turns)),
        "analysis_notes": (
            "RAG/tool counts are transcript keyword candidates, not historical usage."
        ),
    }, {"input_tokens": 0, "output_tokens": 0}


def aggregate_role(
    turns: list[dict[str, Any]],
    role: str,
) -> tuple[str, float]:
    selected = [t for t in turns if t["role"] == role]
    text = " ".join(t["text"] for t in selected).strip()
    seconds = sum(float(t["duration"]) for t in selected)
    return text, seconds


def conversation_history_tokens(
    turns: list[dict[str, Any]],
) -> int:
    """
    For each AGENT turn, assume one production LLM call whose input includes
    all prior customer + agent conversation text.

    This intentionally estimates repeated history transmission.
    """
    running = 0
    total = 0

    for turn in turns:
        if turn["role"] == "AGENT":
            total += running

        running += int(turn["tokens_est"])

    return total


def analyze_file(
    path: Path,
    whisper_model: WhisperModel | None,
    bedrock_client: Any,
    language: str | None,
    audio_metrics: dict[str, float] | None = None,
    transcript_path: Path | None = None,
) -> dict[str, Any]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "status": "processing",
            "bedrock_model_id_used_for_offline_analysis": BEDROCK_MODEL_ID,
        }
    )

    try:
        audio = audio_metrics or get_audio_metrics(path)
        duration = float(audio["duration_seconds"])
        active = float(audio["active_speech_seconds"])

        if transcript_path is not None:
            segments, whisper_info = transcribe_from_text(transcript_path)
            row["transcript_source"] = "matching_txt"
        else:
            if whisper_model is None:
                raise RuntimeError(
                    "Whisper model is unavailable and no transcript was found."
                )
            segments, whisper_info = transcribe_local(
                path,
                whisper_model,
                language=language,
            )
            row["transcript_source"] = "local_faster_whisper"

        # Preserve measured/basic values even if a later analysis step fails.
        # This prevents a Bedrock or parsing error from turning known audio
        # and transcript facts into blank CSV cells.
        row.update(
            {
                "call_duration_seconds": round3(duration),
                "call_duration_minutes": round3(duration / 60),
                "audio_active_speech_seconds": round3(active),
                "silence_seconds_est": round3(max(0.0, duration - active)),
                "whisper_language": whisper_info["language"],
                "whisper_language_probability": round3(
                    whisper_info["language_probability"]
                ),
                "whisper_segments": len(segments),
            }
        )

        transcript_labels = labels_from_transcript(segments)
        if transcript_labels is not None:
            labels = transcript_labels
            role_usage = {"input_tokens": 0, "output_tokens": 0}
            row["speaker_label_source"] = "matching_txt_configured_labels"
            row["unknown_speaker_segments"] = sum(
                role == "UNKNOWN" for role in labels.values()
            )
        else:
            labels, role_usage = classify_segments_with_bedrock(
                segments,
                bedrock_client,
            )
            row["speaker_label_source"] = "bedrock_inferred_from_text"
            row["unknown_speaker_segments"] = 0

        turns = build_turns(segments, labels)

        scenario, scenario_usage = classify_rag_tools_with_bedrock(
            turns,
            bedrock_client,
        )
        row["scenario_classifier_method"] = SCENARIO_CLASSIFIER_MODE

        customer_text, customer_seconds = aggregate_role(
            turns, "CUSTOMER"
        )
        agent_text, agent_seconds = aggregate_role(
            turns, "AGENT"
        )

        customer_words = word_count(customer_text)
        agent_words = word_count(agent_text)
        customer_tokens = estimate_tokens(customer_text)
        agent_tokens = estimate_tokens(agent_text)

        if transcript_path is not None:
            # TXT files have speaker labels but no timestamps. Allocate the
            # measured active audio time by known speaker word share so role
            # durations cannot exceed the MP3's measured active duration.
            known_words = customer_words + agent_words
            if known_words:
                customer_seconds = active * customer_words / known_words
                agent_seconds = active * agent_words / known_words
            else:
                customer_seconds = 0.0
                agent_seconds = 0.0
            row["speech_duration_basis"] = (
                "audio_active_time_allocated_by_txt_word_share"
            )
        else:
            row["speech_duration_basis"] = "whisper_segment_timestamps"

        agent_turns = [
            t for t in turns
            if t["role"] == "AGENT"
        ]
        llm_calls = len(agent_turns)

        rag_turns = scenario["rag_turn_indices"]
        tool_turns = scenario["tool_turn_indices"]

        rag_queries = len(rag_turns)
        tool_calls = len(tool_turns)

        system_tokens = (
            llm_calls * SYSTEM_PROMPT_TOKENS_PER_LLM_CALL
        )
        history_tokens = conversation_history_tokens(turns)
        tool_schema_tokens = (
            llm_calls * TOOL_SCHEMA_TOKENS_PER_LLM_CALL
        )
        rag_context_tokens = (
            rag_queries * RAG_CONTEXT_TOKENS_PER_QUERY
        )
        tool_result_tokens = (
            tool_calls * TOOL_RESULT_TOKENS_PER_CALL
        )

        llm_input_total = (
            system_tokens
            + history_tokens
            + tool_schema_tokens
            + rag_context_tokens
            + tool_result_tokens
        )

        # Spoken agent response text is the main production LLM output.
        llm_response_tokens = agent_tokens
        tool_call_output_tokens = (
            tool_calls * TOOL_CALL_OUTPUT_TOKENS_PER_CALL
        )
        llm_output_total = (
            llm_response_tokens + tool_call_output_tokens
        )

        offline_input = (
            role_usage["input_tokens"]
            + scenario_usage["input_tokens"]
        )
        offline_output = (
            role_usage["output_tokens"]
            + scenario_usage["output_tokens"]
        )

        row.update(
            {
                "status": "ok",
                "call_duration_seconds": round3(duration),
                "call_duration_minutes": round3(duration / 60),
                "audio_active_speech_seconds": round3(active),
                "silence_seconds_est": round3(
                    max(0.0, duration - active)
                ),

                "whisper_language": whisper_info["language"],
                "whisper_language_probability": round3(
                    whisper_info["language_probability"]
                ),
                "whisper_segments": len(segments),

                "customer_speech_seconds_est": round3(customer_seconds),
                "customer_speech_minutes_est": round3(
                    customer_seconds / 60
                ),
                "agent_speech_seconds_est": round3(agent_seconds),
                "agent_speech_minutes_est": round3(
                    agent_seconds / 60
                ),

                "customer_words": customer_words,
                "customer_text_tokens_est": customer_tokens,
                "agent_words": agent_words,
                "agent_text_tokens_est": agent_tokens,
                "total_spoken_words": customer_words + agent_words,
                "total_spoken_tokens_est": (
                    customer_tokens + agent_tokens
                ),

                "stt_full_audio_seconds": round3(duration),
                "stt_full_audio_minutes": round3(duration / 60),
                "stt_customer_audio_seconds_est": round3(
                    customer_seconds
                ),
                "stt_customer_audio_minutes_est": round3(
                    customer_seconds / 60
                ),

                "llm_calls_est": llm_calls,
                "system_prompt_tokens_total": system_tokens,
                "conversation_history_tokens_total_est": history_tokens,
                "tool_schema_tokens_total": tool_schema_tokens,
                "rag_queries_est": rag_queries,
                "rag_context_tokens_total": rag_context_tokens,
                "tool_calls_est": tool_calls,
                "tool_result_tokens_total": tool_result_tokens,
                "llm_input_tokens_total_est": llm_input_total,
                "llm_response_tokens_total_est": llm_response_tokens,
                "tool_call_output_tokens_total_est": (
                    tool_call_output_tokens
                ),
                "llm_output_tokens_total_est": llm_output_total,

                "tts_text_tokens_est": agent_tokens,
                "tts_characters": len(agent_text),
                "tts_audio_seconds_est": round3(agent_seconds),
                "tts_audio_minutes_est": round3(agent_seconds / 60),

                "rag_embedding_tokens_est": (
                    rag_queries * RAG_QUERY_EMBEDDING_TOKENS
                ),

                "rag_turn_indices_est": json.dumps(rag_turns),
                "tool_turn_indices_est": json.dumps(tool_turns),
                "bedrock_offline_input_tokens": offline_input,
                "bedrock_offline_output_tokens": offline_output,
                "analysis_notes": scenario["analysis_notes"],
            }
        )

    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def write_csv(
    rows: list[dict[str, Any]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local call recordings into provider-neutral "
            "STT/LLM/TTS/RAG/tool usage CSV."
        )
    )
    parser.add_argument(
        "--input-folder",
        default=Path(INPUT_FOLDER),
        type=Path,
        help=(
            "Folder containing recordings. Defaults to INPUT_FOLDER from .env."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=Path("voice_call_usage.csv"),
        type=Path,
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
    )
    parser.add_argument(
        "--whisper-model",
        default="small.en",
        help=(
            "faster-whisper model name. Examples: tiny.en, base.en, "
            "small.en, medium.en, large-v3."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help=(
            "Typical CPU: int8. Typical NVIDIA GPU: float16."
        ),
    )
    parser.add_argument(
        "--language",
        default="en",
        help=(
            "Whisper language code. Use 'auto' for auto-detection."
        ),
    )
    return parser.parse_args()


def validate_config() -> None:
    if not BEDROCK_MODEL_ID:
        raise RuntimeError(
            "BEDROCK_MODEL_ID environment variable is required."
        )


def main() -> int:
    args = parse_args()
    validate_config()

    if (
        not args.input_folder.exists()
        or not args.input_folder.is_dir()
    ):
        print(
            f"Input folder does not exist: {args.input_folder}",
            file=sys.stderr,
        )
        return 2

    iterator = (
        args.input_folder.rglob("*")
        if args.recursive
        else args.input_folder.glob("*")
    )

    files = sorted(
        p for p in iterator
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_AUDIO
    )

    if not files:
        print(
            "No supported audio files found.",
            file=sys.stderr,
        )
        return 2

    # Check duration before loading the STT model or making any Bedrock calls.
    # Read only container metadata here; full audio decoding for silence
    # analysis is deferred until an eligible file is actually processed.
    # Files exactly 60 seconds long are excluded because the requirement is
    # strictly longer than one minute.
    eligible_files: list[Path] = []
    audio_metrics_by_path: dict[Path, dict[str, float]] = {}
    skipped_short = 0

    for path in files:
        try:
            duration_seconds = get_audio_duration_seconds(path)
        except Exception:
            # Keep unreadable files so analyze_file can report the actual error
            # in the CSV rather than silently dropping them.
            eligible_files.append(path)
            continue

        if duration_seconds > MIN_AUDIO_DURATION_SECONDS:
            eligible_files.append(path)
        else:
            skipped_short += 1

    files = eligible_files

    if not files:
        print(
            "No supported audio files longer than "
            f"{MIN_AUDIO_DURATION_SECONDS:g} seconds found.",
            file=sys.stderr,
        )
        print(f"Skipped {skipped_short} file(s) at or below the duration limit.")
        return 2

    language = (
        None
        if args.language.lower() == "auto"
        else args.language
    )

    transcript_by_path = {
        path: find_transcript(path)
        for path in files
    }
    transcript_count = sum(
        transcript_path is not None
        for transcript_path in transcript_by_path.values()
    )

    whisper_model: WhisperModel | None = None
    if transcript_count < len(files):
        print("Loading local STT model for files without transcripts...")
        whisper_model = WhisperModel(
            args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
        )
    else:
        print("Matching transcripts found for all recordings; Whisper skipped.")

    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )

    rows: list[dict[str, Any]] = []

    print(
        f"Found {len(files)} recording(s) longer than "
        f"{MIN_AUDIO_DURATION_SECONDS:g} seconds"
    )
    print(f"Matching text transcripts: {transcript_count}/{len(files)}")
    if skipped_short:
        print(
            f"Skipped {skipped_short} recording(s) at or below "
            f"{MIN_AUDIO_DURATION_SECONDS:g} seconds"
        )
    print(f"Local STT model: {args.whisper_model}")
    print(f"Bedrock model: {BEDROCK_MODEL_ID}")
    print(f"AWS region: {AWS_REGION}")
    print("S3: NOT USED")
    print()

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path.name}")

        row = analyze_file(
            path=path,
            whisper_model=whisper_model,
            bedrock_client=bedrock_client,
            language=language,
            audio_metrics=audio_metrics_by_path.get(path),
            transcript_path=transcript_by_path.get(path),
        )
        rows.append(row)

        # Save after every file so a long run is recoverable.
        write_csv(rows, args.output_csv)

        if row["status"] == "ok":
            print(
                "  OK | "
                f'call={row["call_duration_minutes"]} min | '
                f'STT(customer)='
                f'{row["stt_customer_audio_minutes_est"]} min | '
                f'LLM in={row["llm_input_tokens_total_est"]} | '
                f'LLM out={row["llm_output_tokens_total_est"]} | '
                f'TTS chars={row["tts_characters"]} | '
                f'RAG={row["rag_queries_est"]} | '
                f'Tools={row["tool_calls_est"]}'
            )
        else:
            print(f'  ERROR | {row["error"]}')

    success = sum(
        1 for row in rows
        if row["status"] == "ok"
    )
    failed = len(rows) - success

    print()
    print(
        f"Finished: {success} succeeded, {failed} failed"
    )
    print(f"CSV: {args.output_csv.resolve()}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClientError, BotoCoreError) as exc:
        print(f"AWS error: {exc}", file=sys.stderr)
        raise SystemExit(1)
