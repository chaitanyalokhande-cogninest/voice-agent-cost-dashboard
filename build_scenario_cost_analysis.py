#!/usr/bin/env python3
"""Build a transparent production-cost scenario from audio/TXT-derived CSV.

This report is intentionally not called an invoice. Audio duration and
transcript text are measured from the supplied artifacts; production call
counts, prompt sizes, RAG, tools, and vendor rates are scenario inputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


OUTPUT_FIELDS = [
    "file_name",
    "file_path",
    "transcript_source",
    "speaker_label_source",
    "call_duration_minutes_measured",
    "agent_turns_observed",
    "customer_words_measured",
    "agent_words_measured",
    "agent_characters_proxy",
    "rag_candidates_transcript_only",
    "tool_candidates_transcript_only",
    "unknown_speaker_segments",
    "stt_cost_usd",
    "tts_cost_usd",
    "offline_analysis_cost_excluded_usd",
    "base_llm_input_tokens",
    "base_llm_output_tokens",
    "base_llm_cost_usd",
    "base_total_cost_usd",
    "basis_notes",
]


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def number(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = (row.get(name) or "").strip()
    return float(value) if value else default


def integer(row: dict[str, str], name: str, default: int = 0) -> int:
    value = (row.get(name) or "").strip()
    return int(float(value)) if value else default


def money(value: float) -> float:
    return round(value, 6)


def base_scenario_tokens(
    row: dict[str, str],
    system_tokens: int,
    schema_tokens: int,
    rag_context_tokens: int,
    tool_result_tokens: int,
    tool_output_tokens: int,
) -> tuple[int, int]:
    """Return tokens for the single Base production-cost estimate."""
    agent_turns = integer(row, "llm_calls_est")
    tool_candidates = integer(row, "tool_calls_est")
    rag_candidates = integer(row, "rag_queries_est")
    history = integer(row, "conversation_history_tokens_total_est")
    agent_output = integer(row, "agent_text_tokens_est")

    # Base assumption: one production LLM call per observed agent turn and
    # one possible tool follow-up for each transcript tool candidate.
    extra_tool_calls = tool_candidates
    calls = agent_turns + extra_tool_calls
    average_history = history / max(agent_turns, 1)
    modeled_history = history + round(average_history * extra_tool_calls)
    modeled_input = (
        modeled_history
        + round(calls * (system_tokens + schema_tokens))
        + round(rag_candidates * rag_context_tokens)
        + round(tool_candidates * tool_result_tokens)
    )
    modeled_output = (
        agent_output
        + round(tool_candidates * tool_output_tokens)
    )
    return modeled_input, modeled_output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create one transparent Base cost estimate from usage CSV."
    )
    parser.add_argument(
        "--usage-csv",
        type=Path,
        default=ROOT / "voice_call_usage.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "voice_call_cost_scenarios.csv",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=ROOT / "voice_call_cost_scenarios_summary.md",
    )
    args = parser.parse_args()

    stt_rate = env_float("STT_SCENARIO_USD_PER_MINUTE", 0.006)
    tts_rate = env_float("TTS_SCENARIO_USD_PER_1M_CHARS", 15.0)
    llm_input_rate = env_float("LLM_SCENARIO_INPUT_USD_PER_1M", 0.30)
    llm_output_rate = env_float("LLM_SCENARIO_OUTPUT_USD_PER_1M", 2.50)
    embedding_rate = env_float("EMBEDDING_SCENARIO_USD_PER_1M", 0.0)

    system_tokens = env_int("SYSTEM_PROMPT_TOKENS_PER_LLM_CALL", 700)
    schema_tokens = env_int("TOOL_SCHEMA_TOKENS_PER_LLM_CALL", 600)
    rag_context_tokens = env_int("RAG_CONTEXT_TOKENS_PER_QUERY", 700)
    tool_result_tokens = env_int("TOOL_RESULT_TOKENS_PER_CALL", 150)
    tool_output_tokens = env_int("TOOL_CALL_OUTPUT_TOKENS_PER_CALL", 40)
    model_id = os.getenv("BEDROCK_MODEL_ID", "not specified")

    with args.usage_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))

    successful = [row for row in source_rows if row.get("status") == "ok"]
    output_rows: list[dict[str, object]] = []

    for row in successful:
        minutes = number(row, "call_duration_minutes")
        agent_chars = integer(row, "tts_characters")
        stt_cost = minutes * stt_rate
        tts_cost = agent_chars * tts_rate / 1_000_000
        offline_cost = (
            number(row, "bedrock_offline_input_tokens") * llm_input_rate / 1_000_000
            + number(row, "bedrock_offline_output_tokens") * llm_output_rate / 1_000_000
        )

        input_tokens, output_tokens = base_scenario_tokens(
            row,
            system_tokens,
            schema_tokens,
            rag_context_tokens,
            tool_result_tokens,
            tool_output_tokens,
        )
        llm_cost = (
            input_tokens * llm_input_rate / 1_000_000
            + output_tokens * llm_output_rate / 1_000_000
        )
        embedding_tokens = integer(row, "rag_queries_est") * env_int(
            "RAG_QUERY_EMBEDDING_TOKENS", 20
        )
        embedding_cost = embedding_tokens * embedding_rate / 1_000_000
        total_cost = stt_cost + tts_cost + llm_cost + embedding_cost

        out: dict[str, object] = {
            "file_name": row.get("file_name", ""),
            "file_path": row.get("file_path", ""),
            "transcript_source": row.get("transcript_source", "legacy_csv"),
            "speaker_label_source": row.get("speaker_label_source", "legacy_csv"),
            "call_duration_minutes_measured": money(minutes),
            "agent_turns_observed": integer(row, "llm_calls_est"),
            "customer_words_measured": integer(row, "customer_words"),
            "agent_words_measured": integer(row, "agent_words"),
            "agent_characters_proxy": agent_chars,
            "rag_candidates_transcript_only": integer(row, "rag_queries_est"),
            "tool_candidates_transcript_only": integer(row, "tool_calls_est"),
            "unknown_speaker_segments": integer(row, "unknown_speaker_segments"),
            "stt_cost_usd": money(stt_cost),
            "tts_cost_usd": money(tts_cost),
            "offline_analysis_cost_excluded_usd": money(offline_cost),
            "basis_notes": (
                "Audio/TXT workload only; LLM calls are agent-turn proxies; "
                "RAG/tools are transcript candidates; offline analysis excluded."
            ),
        }
        out["base_llm_input_tokens"] = input_tokens
        out["base_llm_output_tokens"] = output_tokens
        out["base_llm_cost_usd"] = money(llm_cost)
        out["base_total_cost_usd"] = money(total_cost)
        output_rows.append(out)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    def total(field: str) -> float:
        return sum(float(row.get(field) or 0) for row in output_rows)

    total_minutes = total("call_duration_minutes_measured")
    summary = f"""# Voice-call Base production cost estimate

Generated: {date.today().isoformat()}

## Result

| Metric | Value |
|---|---:|
| Successful calls in source CSV | {len(successful)} |
| Excluded/error rows | {len(source_rows) - len(successful)} |
| Measured call minutes | {total_minutes:.3f} |
| Agent turns used as call proxies | {int(total('agent_turns_observed'))} |
| RAG transcript candidates | {int(total('rag_candidates_transcript_only'))} |
| Tool transcript candidates | {int(total('tool_candidates_transcript_only'))} |
| Unattributed TXT speaker segments | {int(total('unknown_speaker_segments'))} |
| Base STT cost | ${total('stt_cost_usd'):.4f} |
| Base TTS cost | ${total('tts_cost_usd'):.4f} |
| Base LLM cost | ${total('base_llm_cost_usd'):.4f} |
| Base estimated total | ${total('base_total_cost_usd'):.4f} |
| Base estimated cost per successful call | ${(total('base_total_cost_usd') / len(successful)) if successful else 0:.4f} |
| Base estimated cost per measured minute | ${(total('base_total_cost_usd') / total_minutes) if total_minutes else 0:.6f} |

## Rate card used

- Bedrock model used for estimate: `{model_id}`; input `${llm_input_rate:.4f}`/1M tokens; output `${llm_output_rate:.4f}`/1M tokens.
- STT rate: `${stt_rate:.4f}`/audio minute.
- TTS rate: `${tts_rate:.2f}`/1M characters.
- Embedding rate: `${embedding_rate:.4f}`/1M tokens (zero unless configured).
- Prompt tokens per call: `{system_tokens}`; tool schema tokens per call: `{schema_tokens}`; RAG context: `{rag_context_tokens}`; tool result: `{tool_result_tokens}`; tool output: `{tool_output_tokens}`.

## Interpretation

The audio duration and transcript-derived words/characters are the evidence.
The Base estimate uses one production LLM call per observed agent turn, the
configured prompt/schema assumptions, and one possible tool follow-up per
transcript candidate. This is an explicit modeling choice, not observed
production behavior.

The local analyzer's Bedrock calls are post-call analysis and are excluded from
the production total. They are shown separately in the per-call CSV so they do
not get confused with the hypothetical deployed voice-agent bill.

STT and TTS are vendor scenarios: the current analyzer uses local
`faster-whisper` and does not invoke TTS, so those local operations do not create
an API invoice. Replace the rates with the selected production vendors before
presenting a final budget.

Reference rate pages: <https://aws.amazon.com/bedrock/pricing/>,
<https://developers.openai.com/api/docs/models/gpt-transcribe>, and
<https://developers.openai.com/api/docs/models/gpt-4o-mini-tts>.
"""
    args.summary_md.write_text(summary, encoding="utf-8")

    print(f"Wrote {len(output_rows)} successful-call scenarios")
    print(f"CSV: {args.output_csv.resolve()}")
    print(f"Summary: {args.summary_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
