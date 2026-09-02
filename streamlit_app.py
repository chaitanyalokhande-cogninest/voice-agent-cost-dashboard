"""Interactive dashboard for recalculating voice-call cost scenarios."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import set_key

from build_scenario_cost_analysis import base_scenario_tokens


ROOT = Path(__file__).resolve().parent
CATALOG_DATE = "2026-09-02"

STT = {
    # Current OpenAI transcription models priced directly by audio minute.
    # Whisper-1 is intentionally not included in the dashboard's recommended
    # catalog; the OpenAI docs still list it, but these newer models are the
    # better forward-looking choices for a new estimate.
    "OpenAI GPT Transcribe": (0.0045, "gpt-transcribe", "https://developers.openai.com/api/docs/models/gpt-transcribe"),
    "OpenAI GPT Live Transcribe": (0.017, "gpt-live-transcribe", "https://developers.openai.com/api/docs/models/gpt-live-transcribe"),
    "OpenAI GPT Realtime Whisper": (0.017, "gpt-realtime-whisper", "https://developers.openai.com/api/docs/models/gpt-realtime-whisper"),
    "Google Cloud Speech-to-Text V2 Standard": (0.016, "Speech-to-Text V2 Standard", "https://cloud.google.com/speech-to-text/pricing"),
    "Google Cloud Speech-to-Text V2 Dynamic Batch": (0.003, "Speech-to-Text V2 Dynamic Batch", "https://cloud.google.com/speech-to-text/pricing"),
    "Google Cloud STT V2 Phone Call": (0.016, "phone_call", "https://cloud.google.com/speech-to-text/pricing"),
    "Google Cloud STT V2 Chirp": (0.016, "chirp", "https://cloud.google.com/speech-to-text/pricing"),
    "Amazon Transcribe Standard": (0.024, "Standard batch/streaming", "https://aws.amazon.com/transcribe/pricing/"),
    "Amazon Transcribe Call Analytics": (0.030, "Call Analytics", "https://aws.amazon.com/transcribe/pricing/"),
    "Deepgram Nova-3 Pre-recorded": (float(os.getenv("DEEPGRAM_STT_SCENARIO_USD_PER_MINUTE", "0.0048")), "nova-3", "https://deepgram.com/pricing"),
    "Deepgram Nova-3 Streaming": (0.0077, "nova-3", "https://deepgram.com/pricing"),
    "Deepgram Nova-3 Multilingual Pre-recorded": (0.0058, "nova-3", "https://deepgram.com/pricing"),
    "Deepgram Nova-3 Multilingual Streaming": (0.0092, "nova-3", "https://deepgram.com/pricing"),
    "Deepgram Flux English Streaming": (0.0065, "flux", "https://deepgram.com/pricing"),
    "Deepgram Flux Multilingual Streaming": (0.0078, "flux", "https://deepgram.com/pricing"),
}
TTS = {
    "OpenAI TTS-1": (15.0, "tts-1", "https://developers.openai.com/api/docs/models/tts-1"),
    "OpenAI TTS-1 HD": (30.0, "tts-1-hd", "https://developers.openai.com/api/docs/models/tts-1"),
    "Google Cloud TTS Standard": (4.0, "Standard voices", "https://cloud.google.com/text-to-speech/pricing"),
    "Google Cloud TTS Neural2": (16.0, "Neural2 voices", "https://cloud.google.com/text-to-speech/pricing"),
    "Google Cloud TTS WaveNet": (16.0, "WaveNet voices", "https://cloud.google.com/text-to-speech/pricing"),
    "Google Cloud TTS Chirp 3 HD": (30.0, "Chirp 3 HD voices", "https://cloud.google.com/text-to-speech/pricing"),
    "Google Cloud TTS Studio": (160.0, "Studio voices", "https://cloud.google.com/text-to-speech/pricing"),
    "Amazon Polly Standard": (4.0, "Standard voices", "https://aws.amazon.com/polly/pricing/"),
    "Amazon Polly Neural": (16.0, "Neural voices", "https://aws.amazon.com/polly/pricing/"),
    "Amazon Polly Generative": (30.0, "Generative voices", "https://aws.amazon.com/polly/pricing/"),
    "Amazon Polly Long-Form": (100.0, "Long-Form voices", "https://aws.amazon.com/polly/pricing/"),
    "Deepgram Aura-2": (float(os.getenv("DEEPGRAM_TTS_SCENARIO_USD_PER_1M_CHARS", "30")), "aura-2-thalia-en", "https://deepgram.com/pricing"),
}
LLM = {
    # OpenAI API models. These are included as provider-comparison scenarios;
    # they are not Amazon Bedrock model IDs.
    "OpenAI GPT-5.4 nano": (0.20, 1.25, "gpt-5.4-nano", "https://developers.openai.com/api/docs/models/gpt-5.4-nano"),
    "OpenAI GPT-5.4 mini": (0.75, 4.50, "gpt-5.4-mini", "https://developers.openai.com/api/docs/models/gpt-5.4-mini"),
    "OpenAI GPT-4.1 mini": (0.40, 1.60, "gpt-4.1-mini", "https://developers.openai.com/api/docs/models/gpt-4.1-mini"),
    "OpenAI GPT-4o mini": (0.15, 0.60, "gpt-4o-mini", "https://developers.openai.com/api/docs/models/gpt-4o-mini"),

    # Current Claude models available through Amazon Bedrock. Prices use the
    # standard global/inference-profile list rates in USD per 1M tokens.
    "Anthropic Claude Haiku 4.5 (Bedrock)": (1.00, 5.00, "us.anthropic.claude-haiku-4-5-20251001-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-haiku-4-5.html"),
    "Anthropic Claude Sonnet 4.5 (Bedrock)": (3.00, 15.00, "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-5.html"),
    "Anthropic Claude Sonnet 4.6 (Bedrock)": (3.00, 15.00, "us.anthropic.claude-sonnet-4-6", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html"),

    # Amazon's active Nova understanding models. Prices are us-east-1
    # on-demand list-rate scenarios in USD per 1M tokens.
    "Amazon Nova Micro (Bedrock)": (0.035, 0.14, "amazon.nova-micro-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-micro.html"),
    "Amazon Nova Lite (Bedrock)": (0.06, 0.24, "amazon.nova-lite-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html"),
    "Amazon Nova Pro (Bedrock)": (0.80, 3.20, "amazon.nova-pro-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-pro.html"),

    "Amazon Nova 2 Lite (Bedrock)": (0.30, 2.50, "us.amazon.nova-2-lite-v1:0", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-2-lite.html"),
    "DeepSeek V3.2 (Bedrock)": (0.62, 1.85, "deepseek.v3.2", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html"),
    "Google Gemma 3 27B (Bedrock)": (0.23, 0.38, "google.gemma-3-27b-it", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-27b-pt.html"),
    "Google Gemma 4 E2B (Bedrock)": (0.04, 0.08, "google.gemma-4-e2b-it", "https://aws.amazon.com/bedrock/pricing/"),
    "Google Gemma 4 26B A4B (Bedrock)": (0.13, 0.40, "google.gemma-4-26b-a4b-it", "https://aws.amazon.com/bedrock/pricing/"),
    "Google Gemma 4 31B (Bedrock)": (0.14, 0.40, "google.gemma-4-31b-it", "https://aws.amazon.com/bedrock/pricing/"),
    "Google Gemma 3 4B (Bedrock)": (0.04, 0.08, "google.gemma-3-4b-it", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-4b-it.html"),
    "Google Gemma 3 12B (Bedrock)": (0.09, 0.29, "google.gemma-3-12b-it", "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-12b-it.html"),
    "MiniMax M2 (Bedrock)": (0.30, 1.20, "minimax.minimax-m2", "https://aws.amazon.com/bedrock/pricing/"),
    "MiniMax M2.1 (Bedrock)": (0.30, 1.20, "minimax.minimax-m2.1", "https://aws.amazon.com/bedrock/pricing/"),
    "MiniMax M2.5 (Bedrock)": (0.30, 1.20, "minimax.minimax-m2.5", "https://aws.amazon.com/bedrock/pricing/"),
    "Mistral Large 3 (Bedrock)": (0.50, 1.50, "mistral.mistral-large-3-675b-instruct", "https://aws.amazon.com/bedrock/pricing/"),
    "Mistral Voxtral Mini 1.0 (Bedrock)": (0.04, 0.04, "mistral.voxtral-mini-1-0", "https://aws.amazon.com/bedrock/pricing/"),
}


def num(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: object, default: int = 0) -> int:
    return int(num(value, default))


def load_csv(path_text: str, uploaded: object | None) -> pd.DataFrame:
    if uploaded is not None:
        return pd.read_csv(uploaded)
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Usage CSV not found: {path}")
    return pd.read_csv(path)


def scaled_row(source: pd.Series, token_factor: float, source_factor: float) -> dict[str, str]:
    row = {str(k): "" if pd.isna(v) else str(v) for k, v in source.items()}
    scale = token_factor / max(source_factor, 0.0001)
    row["conversation_history_tokens_total_est"] = str(round(num(row.get("conversation_history_tokens_total_est")) * scale))
    row["agent_text_tokens_est"] = str(round(num(row.get("agent_words")) * token_factor))
    return row


def calculate(data: pd.DataFrame, a: dict[str, float], stt_rate: float, tts_rate: float, llm_in: float, llm_out: float) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for _, source in data[data["status"].fillna("") == "ok"].iterrows():
        row = scaled_row(source, a["tokens_per_word"], a["source_tokens_per_word"])
        minutes = num(row.get("call_duration_minutes"))
        stt_cost = minutes * stt_rate
        tts_cost = integer(row.get("tts_characters")) * tts_rate / 1_000_000
        offline_cost = (
            num(row.get("bedrock_offline_input_tokens")) * llm_in / 1_000_000
            + num(row.get("bedrock_offline_output_tokens")) * llm_out / 1_000_000
        )
        result: dict[str, object] = {
            "file_name": row.get("file_name", ""),
            "call_minutes": round(minutes, 3),
            "agent_turns_proxy": integer(row.get("llm_calls_est")),
            "rag_candidates": integer(row.get("rag_queries_est")),
            "tool_candidates": integer(row.get("tool_calls_est")),
            "stt_cost_usd": stt_cost,
            "tts_cost_usd": tts_cost,
            "offline_analysis_excluded_usd": offline_cost,
        }
        input_tokens, output_tokens = base_scenario_tokens(
            row, int(a["system_prompt_tokens"]), int(a["tool_schema_tokens"]),
            int(a["rag_context_tokens"]), int(a["tool_result_tokens"]), int(a["tool_output_tokens"]),
        )
        embedding_tokens = integer(row.get("rag_queries_est")) * a["rag_embedding_tokens"]
        llm_cost = input_tokens * llm_in / 1_000_000 + output_tokens * llm_out / 1_000_000
        embedding_cost = embedding_tokens * a["embedding_rate"] / 1_000_000
        result["base_llm_input_tokens"] = input_tokens
        result["base_llm_output_tokens"] = output_tokens
        result["base_llm_cost_usd"] = llm_cost
        result["base_total_cost_usd"] = stt_cost + tts_cost + llm_cost + embedding_cost
        output.append(result)
    return pd.DataFrame(output)


def save_txt_settings(agent: str, customer: str, unknown: str, classifier: str) -> None:
    path = ROOT / ".env"
    for key, value in {
        "TRANSCRIPT_AGENT_LABEL": agent,
        "TRANSCRIPT_CUSTOMER_LABEL": customer,
        "TRANSCRIPT_UNKNOWN_LABEL": unknown,
        "SCENARIO_CLASSIFIER_MODE": classifier,
    }.items():
        set_key(str(path), key, value, quote_mode="auto")


st.set_page_config(page_title="Voice Cost Scenarios", layout="wide")
st.title("Voice-call production cost estimate")
st.caption("Measured MP3/TXT workload plus explicit provider rates and architecture assumptions. This is a Base estimate, not an invoice.")

with st.sidebar:
    st.header("Dataset")
    csv_path = st.text_input("Usage CSV path", "voice_call_usage_from_txt.csv")
    uploaded = st.file_uploader("Or upload a usage CSV", type="csv")

    st.header("Provider/model pricing")
    stt_name = st.selectbox("STT", list(STT))
    tts_name = st.selectbox("TTS", list(TTS))
    llm_name = st.selectbox("LLM", list(LLM))
    stt_default, stt_model, stt_source = STT[stt_name]
    tts_default, tts_model, tts_source = TTS[tts_name]
    llm_in_default, llm_out_default, llm_model, llm_source = LLM[llm_name]
    # Model-specific keys prevent a previously edited rate from silently
    # carrying over when the user selects a different provider/model.
    stt_rate = st.number_input("STT $ / audio minute", min_value=0.0, value=stt_default, format="%.6f", key=f"stt_rate_{stt_name}")
    tts_rate = st.number_input("TTS $ / 1M characters", min_value=0.0, value=tts_default, format="%.4f", key=f"tts_rate_{tts_name}")
    llm_in = st.number_input("LLM $ / 1M input tokens", min_value=0.0, value=llm_in_default, format="%.4f", key=f"llm_in_{llm_name}")
    llm_out = st.number_input("LLM $ / 1M output tokens", min_value=0.0, value=llm_out_default, format="%.4f", key=f"llm_out_{llm_name}")
    embedding_rate = st.number_input("Embedding $ / 1M tokens", min_value=0.0, value=float(os.getenv("EMBEDDING_SCENARIO_USD_PER_1M", "0")), format="%.4f")

    st.header("Workload assumptions")
    tokens_per_word = st.number_input("TOKENS_PER_WORD", min_value=0.1, value=float(os.getenv("TOKENS_PER_WORD", "1.30")), step=0.05, format="%.2f")
    source_tokens_per_word = st.number_input("Source CSV token factor", min_value=0.1, value=float(os.getenv("TOKENS_PER_WORD", "1.30")), step=0.05, format="%.2f")
    system_tokens = st.number_input("SYSTEM_PROMPT_TOKENS_PER_LLM_CALL", min_value=0, value=int(os.getenv("SYSTEM_PROMPT_TOKENS_PER_LLM_CALL", "700")))
    schema_tokens = st.number_input("TOOL_SCHEMA_TOKENS_PER_LLM_CALL", min_value=0, value=int(os.getenv("TOOL_SCHEMA_TOKENS_PER_LLM_CALL", "600")))
    rag_context = st.number_input("RAG_CONTEXT_TOKENS_PER_QUERY", min_value=0, value=int(os.getenv("RAG_CONTEXT_TOKENS_PER_QUERY", "700")))
    tool_result = st.number_input("TOOL_RESULT_TOKENS_PER_CALL", min_value=0, value=int(os.getenv("TOOL_RESULT_TOKENS_PER_CALL", "150")))
    tool_output = st.number_input("TOOL_CALL_OUTPUT_TOKENS_PER_CALL", min_value=0, value=int(os.getenv("TOOL_CALL_OUTPUT_TOKENS_PER_CALL", "40")))
    rag_embedding = st.number_input("RAG_QUERY_EMBEDDING_TOKENS", min_value=0, value=int(os.getenv("RAG_QUERY_EMBEDDING_TOKENS", "20")))

    st.header("TXT settings for next analyzer run")
    agent = st.text_input("TRANSCRIPT_AGENT_LABEL", os.getenv("TRANSCRIPT_AGENT_LABEL", "Speaker-1"))
    customer = st.text_input("TRANSCRIPT_CUSTOMER_LABEL", os.getenv("TRANSCRIPT_CUSTOMER_LABEL", "Speaker-2"))
    unknown = st.text_input("TRANSCRIPT_UNKNOWN_LABEL", os.getenv("TRANSCRIPT_UNKNOWN_LABEL", "Speaker-?"))
    current_mode = os.getenv("SCENARIO_CLASSIFIER_MODE", "keywords")
    mode_options = ["keywords", "none", "llm"]
    mode = st.selectbox("SCENARIO_CLASSIFIER_MODE", mode_options, index=mode_options.index(current_mode) if current_mode in mode_options else 0)
    if st.button("Save TXT settings to .env"):
        save_txt_settings(agent, customer, unknown, mode)
        st.success("Saved. Rerun the analyzer for these settings to affect the CSV.")

a = {
    "tokens_per_word": tokens_per_word, "source_tokens_per_word": source_tokens_per_word,
    "system_prompt_tokens": system_tokens, "tool_schema_tokens": schema_tokens,
    "rag_context_tokens": rag_context, "tool_result_tokens": tool_result,
    "tool_output_tokens": tool_output, "rag_embedding_tokens": rag_embedding,
    "embedding_rate": embedding_rate,
}

try:
    source_data = load_csv(csv_path, uploaded)
    missing = sorted({"status", "call_duration_minutes"} - set(source_data.columns))
    if missing:
        raise ValueError("CSV missing required columns: " + ", ".join(missing))
    results = calculate(source_data, a, stt_rate, tts_rate, llm_in, llm_out)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if results.empty:
    st.warning("No successful rows found in this CSV.")
    st.stop()

total_minutes = results["call_minutes"].sum()
base_total = results["base_total_cost_usd"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Successful calls", f"{len(results):,}")
c2.metric("Measured audio minutes", f"{total_minutes:,.3f}")
c3.metric("Base estimate", f"${base_total:,.4f}")
c4.metric("Base cost / call", f"${base_total / len(results):.4f}")

st.subheader("Base estimate")
base_summary = pd.DataFrame({
    "Metric": ["Total cost (USD)", "Cost / call", "Cost / minute"],
    "Value": [base_total, base_total / len(results), base_total / total_minutes],
})
st.dataframe(base_summary.style.format({"Value": "${:.6f}"}), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("Base component cost")
    components = pd.DataFrame({
        "Component": ["STT", "TTS", "LLM", "Offline analysis (excluded)"],
        "Cost (USD)": [results.stt_cost_usd.sum(), results.tts_cost_usd.sum(), results.base_llm_cost_usd.sum(), results.offline_analysis_excluded_usd.sum()],
    })
    st.dataframe(components.style.format({"Cost (USD)": "${:.4f}"}), use_container_width=True, hide_index=True)
with right:
    st.subheader("Selected rates")
    st.markdown(f"**STT:** {stt_name} — `${stt_rate:.6f}`/minute ([official source]({stt_source}))")
    st.markdown(f"**TTS:** {tts_name} — `${tts_rate:.4f}`/1M characters ([official source]({tts_source}))")
    st.markdown(f"**LLM:** {llm_name} (`{llm_model}`) — `${llm_in:.4f}` input / `${llm_out:.4f}` output per 1M tokens ([official source]({llm_source}))")
    st.caption(f"Catalog checked {CATALOG_DATE}; verify region, service tier, free tier, and billing SKU before invoicing.")

with st.expander("Data quality and interpretation"):
    st.write({"source_rows": len(source_data), "status_counts": source_data["status"].fillna("blank").value_counts().to_dict()})
    st.info("Audio duration is measured. TXT words/characters and labels are artifact-derived. Agent turns are call proxies; RAG/tool counts are transcript candidates. Changing TXT settings affects the next analyzer run, not existing CSV rows.")

st.subheader("Per-call results")
display = ["file_name", "call_minutes", "agent_turns_proxy", "rag_candidates", "tool_candidates", "stt_cost_usd", "tts_cost_usd", "base_llm_cost_usd", "base_total_cost_usd"]
st.dataframe(results[display].style.format({column: "${:.6f}" for column in display if "cost" in column}), use_container_width=True, height=500)
st.download_button("Download recalculated scenario CSV", results.to_csv(index=False).encode("utf-8"), "voice_call_dashboard_scenarios.csv", "text/csv")
