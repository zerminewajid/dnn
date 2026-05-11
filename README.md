---
title: Weathering With You
emoji: 🌧️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Weathering With You

> A Pakistani Gen-Z agentic AI weather app. Meet Zero — a tiny floating bot who reacts to weather, explodes when it's too hot, and speaks in Urdu-flavored sentences powered by Claude.

**By Zermine Wajid · zermeenewajid@outlook.com**

## Features
- Real-time weather via Open-Meteo (no API key needed)
- Claude `claude-sonnet-4-20250514` with genuine tool-use (agentic, not chatbot)
- Zero bot: EVE-inspired character with temperature-reactive animations
- Pakistani cities: Lahore, Karachi, Islamabad, Topi, Peshawar, Quetta
- Glassmorphism UI with temperature-reactive gradients
- Hourly timeline, minutely rain, 7-day forecast, AQI
- Activity indexes: Cricket, Chai time, Rickshaw ride, Outdoor study, Iftar walk
- Web Speech API voice — Zero speaks forecasts

## Stack
`React + Tailwind + Framer Motion → FastAPI + Claude tool-use → Open-Meteo → Redis cache → HF Spaces Docker`

## Local Dev

```bash
# Backend
cd backend && pip install -r requirements.txt
ANTHROPIC_API_KEY=sk-... uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

## Docker

```bash
docker-compose up --build
```

Set `ANTHROPIC_API_KEY` in `.env` or HF Spaces Secrets.
