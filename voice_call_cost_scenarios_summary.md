# Voice-call Base production cost estimate

Generated: 2026-09-02

## Result

| Metric | Value |
|---|---:|
| Successful calls in source CSV | 124 |
| Excluded/error rows | 11 |
| Measured call minutes | 521.153 |
| Agent turns used as call proxies | 2132 |
| RAG transcript candidates | 373 |
| Tool transcript candidates | 452 |
| Unattributed TXT speaker segments | 0 |
| Base STT cost | $3.1269 |
| Base TTS cost | $2.8747 |
| Base LLM cost | $1.5969 |
| Base estimated total | $7.5985 |
| Base estimated cost per successful call | $0.0613 |
| Base estimated cost per measured minute | $0.014580 |

## Rate card used

- Bedrock model used for estimate: `us.amazon.nova-2-lite-v1:0`; input `$0.3000`/1M tokens; output `$2.5000`/1M tokens.
- STT rate: `$0.0060`/audio minute.
- TTS rate: `$15.00`/1M characters.
- Embedding rate: `$0.0000`/1M tokens (zero unless configured).
- Prompt tokens per call: `700`; tool schema tokens per call: `600`; RAG context: `700`; tool result: `150`; tool output: `40`.

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
