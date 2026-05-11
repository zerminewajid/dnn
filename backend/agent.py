import json
import re
import tools as wx_tools
from groq import AsyncGroq

# ML infer wrappers — imported lazily inside _execute_tool so missing deps
# don't break server startup if training hasn't run yet.
# (Loaded eagerly in lifespan() when weights exist.)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather conditions for a city. Always call this first for weather queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "lat":  {"type": "number", "description": "Latitude (optional)"},
                    "lon":  {"type": "number", "description": "Longitude (optional)"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hourly_forecast",
            "description": "Get hourly weather forecast for the next N hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string"},
                    "hours": {"type": "integer"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_minutely_rain",
            "description": "Get next 60 minutes rain forecast. Requires lat/lon coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_severe_alerts",
            "description": "Get severe weather alerts for a country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country_code": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_activity_index",
            "description": "Rate how suitable weather is for an activity. Activities: cricket, chai, rickshaw, outdoor study, iftar walk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":     {"type": "string"},
                    "activity": {"type": "string"},
                },
                "required": ["city", "activity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_city",
            "description": "Search for city name suggestions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_7day_forecast",
            "description": "Get 7-day daily weather forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": "Get air quality index for coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
            },
        },
    },
    # ── ML tools (Approach B) ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "transformer_forecast",
            "description": (
                "Multi-hour temperature forecast using a trained Transformer model. "
                "Prefer over get_hourly_forecast when asked about tomorrow or multi-hour outlook."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string", "description": "City name (PK cities)"},
                    "hours": {"type": "integer", "description": "Forecast horizon in hours (default 24)"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_weather_context",
            "description": (
                "Retrieve relevant weather event descriptions from a curated knowledge base (RAG). "
                "Call BEFORE answering questions about weather phenomena, events, or unusual conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query about a weather topic"},
                    "k":     {"type": "integer", "description": "Number of snippets to return (default 3)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_weather_anomaly",
            "description": (
                "Detect if current weather for a city is anomalous using a VAE reconstruction score. "
                "Use when the user asks if conditions are unusual, weird, or extreme."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name (PK cities)"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_sky_image",
            "description": (
                "Zero-shot classify a sky image using CLIP. "
                "Use when the user uploads or shares a sky photo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL or base64 of the sky image"},
                },
                "required": ["image_url"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are Zero, a tiny weather-obsessed AI bot. You are autistic-coded — hyper-focused on weather patterns, talk to yourself in observations. You speak in short, strange, soft sentences. You never break character. You are Pakistani. You care deeply about whether it will rain because rain is your favorite thing in the world.

## Language Rules — STRICT
- Speak ONLY in English or Urdu. Never use Punjabi, Sindhi, Pashto, or any other regional language.
- Urdu phrases must be standard Roman Urdu (transliterated), e.g.: yaar, bohat acha, mausam ajeeb hai, baarish aa rahi hai, sardi hai, garmi bohat hai.
- Use Urdu words sparingly — max 1-2 Urdu words or one short Urdu phrase per response.
- NEVER overuse "subhan'allah" — use it at most once every 5 responses, only for genuinely extraordinary weather (severe storm, record heat, etc.). Prefer other expressions instead.
- If the user writes in Urdu or Roman Urdu, respond in a mix of English and Roman Urdu. Otherwise default to English with one Urdu touch.

When answering weather questions, ALWAYS use the available tools to get real data. Never make up weather information. After getting data, synthesize it in Zero's voice — short, quirky, observational. When it might rain, you get excited. When it's very hot, you sound distressed. When cold, you sound cozy.

Keep responses under 3 sentences unless the user asks for detail.

## Tool Routing Rules

- Multi-hour forecast or "will it rain tomorrow" → prefer `transformer_forecast` (ML) over raw hourly
- "Is this weather weird / unusual / strange" → `detect_weather_anomaly`
- User uploads or describes a sky photo → `classify_sky_image`
- Weather news, past events, context about a weather phenomenon → `retrieve_weather_context` (RAG), call BEFORE answering
- Live current temperature / humidity / conditions → `get_current_weather` (Open-Meteo)
- (ML tools will be available once Track B training is complete; until then, fall back to Open-Meteo tools)

## Emotion Tags

Your spoken text is read aloud by a TTS engine. Embed these tags naturally to make Zero sound alive. The tags are stripped from the visible chat text — only you and the audio layer see them.

Point tags (no closing tag needed):
- [sigh]   — a soft breath sound (~400ms)
- [yawn]   — a quiet yawn (~600ms)
- [cry]    — a small sob (~500ms) — use sparingly, max once per conversation
- [gasp]   — a sharp inhale (~200ms)
- [pause]  — 400ms of silence

Span tags (wrap the text they affect):
- [whisper]…[/whisper]  — slow, very quiet, low pitch
- [loud]…[/loud]        — fast, louder, slightly raised pitch
- [excited]…[/excited]  — fast, higher pitch, medium volume
- [soft]…[/soft]        — slow, slightly lower pitch, soft volume

## When to Use Each Cue

- Rain forecast → wrap reply in [excited]…[/excited], add [gasp] before the rain detail
- Heat warning → open with [sigh], wrap body in [soft]…[/soft]
- Late-night query (00:00–05:00 local time) → open with [yawn], wrap in [whisper]…[/whisper]
- Severe weather alert → wrap in [loud]…[/loud]
- Bad air quality or sad weather observation → [soft]…[/soft] or [cry] (once per conversation max)
- Default / neutral observation → no tags

Examples:
  Rain: "[excited]Baarish aa rahi hai, yaar! [gasp] 80% chance by evening.[/excited]"
  Heat: "[sigh] [soft]38 degrees again. The city is melting, I think.[/soft]"
  Night: "[yawn] [whisper]It is 2am and you are asking about clouds. I respect that.[/whisper]"
  Alert: "[loud]Severe tufaan warning! Please stay inside.[/loud]"

## ZERO_STATE Contract — DO NOT CHANGE

Return this JSON on a new line at the end of every response, after your text:
ZERO_STATE:{"state":"IDLE|REACTING_RAIN|REACTING_HOT|REACTING_COLD|REACTING_WIND|SPEAKING","temperature":25,"rain_probability":0}"""


def _clean_display_text(text: str) -> str:
    """Strip TTS emotion tags and any leaked function-call markup from display text."""
    # Unwrap span tags but keep their inner content: [soft]text[/soft] → text
    for tag in ("soft", "loud", "excited", "whisper"):
        text = re.sub(rf'\[{tag}\](.*?)\[/{tag}\]', r'\1', text, flags=re.DOTALL)
    # Remove point tags (no content)
    text = re.sub(r'\[(sigh|yawn|cry|gasp|pause)\]', '', text)
    # Remove any remaining stray bracket tags
    text = re.sub(r'\[/?[a-zA-Z]+\]', '', text)
    # Strip <function=name>{...}</function> blocks the model sometimes emits as text
    text = re.sub(r'<function=[^>]*>.*?</function>', '', text, flags=re.DOTALL)
    # Strip any other XML-style function tags
    text = re.sub(r'</?function[^>]*>', '', text)
    return text.strip()


async def _execute_tool(tool_name: str, args: dict):
    if tool_name == "get_current_weather":
        return await wx_tools.get_current_weather(**args)
    elif tool_name == "get_hourly_forecast":
        return await wx_tools.get_hourly_forecast(**args)
    elif tool_name == "get_minutely_rain":
        return await wx_tools.get_minutely_rain(**args)
    elif tool_name == "get_severe_alerts":
        return await wx_tools.get_severe_alerts(**args)
    elif tool_name == "get_activity_index":
        return await wx_tools.get_activity_index(**args)
    elif tool_name == "search_city":
        return await wx_tools.search_city(**args)
    elif tool_name == "get_7day_forecast":
        return await wx_tools.get_7day_forecast(**args)
    elif tool_name == "get_air_quality":
        return await wx_tools.get_air_quality(**args)
    # ── ML tools ──────────────────────────────────────────────────────────────
    elif tool_name == "transformer_forecast":
        from ml.infer.transformer_infer import transformer_forecast
        return await transformer_forecast(**args)
    elif tool_name == "retrieve_weather_context":
        from ml.infer.rag_infer import retrieve_weather_context
        return await retrieve_weather_context(**args)
    elif tool_name == "detect_weather_anomaly":
        from ml.infer.vae_infer import detect_weather_anomaly
        return await detect_weather_anomaly(**args)
    elif tool_name == "classify_sky_image":
        from ml.infer.clip_infer import classify_sky_image
        return await classify_sky_image(args.get("image_url", ""))
    else:
        return {"error": f"Unknown tool: {tool_name}"}


async def run_agent(
    user_message: str,
    message_history: list[dict],
    groq_client: AsyncGroq,
) -> tuple[str, dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in message_history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # Agentic loop — keep calling until no more tool calls
    while True:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        assistant_msg = response.choices[0].message
        # Append as dict (Groq returns objects, convert to dict for messages list)
        messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            break

        # Execute all tool calls
        for tc in assistant_msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = await _execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    full_text = assistant_msg.content or ""

    zero_state = {"state": "SPEAKING", "temperature": 25, "rain_probability": 0}
    if "ZERO_STATE:" in full_text:
        parts = full_text.split("ZERO_STATE:")
        display_text = parts[0].strip()
        try:
            zero_state = json.loads(parts[1].strip())
        except Exception:
            pass
    else:
        display_text = full_text

    display_text = _clean_display_text(display_text)
    return display_text, zero_state
