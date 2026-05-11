"""
generate_rag_corpus.py — generate ~200+ Pakistani weather snippets for RAG.

Strategy: templates × cities × seasons × event_types → combinatorial expansion.
Deduplication by (city, season, event_type) key — no repeated combinations.

Appends to any existing hand-curated snippets already in weather_snippets.json.
Existing entries with hand-written ids are preserved unchanged.

Usage:
    python -m ml.generate_rag_corpus
    python -m ml.generate_rag_corpus --force   # overwrite generated entries
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "datasets" / "weather_snippets.json"

# ── Vocabulary ────────────────────────────────────────────────────────────────

CITIES = {
    "Karachi": {
        "region": "Sindh coast",
        "elevation": "sea level",
        "climate": "hot semi-arid",
        "lat": 24.86,
    },
    "Lahore": {
        "region": "central Punjab",
        "elevation": "217 m",
        "climate": "semi-arid",
        "lat": 31.55,
    },
    "Islamabad": {
        "region": "Potohar plateau",
        "elevation": "507 m",
        "climate": "sub-humid",
        "lat": 33.68,
    },
    "Peshawar": {
        "region": "Khyber valley",
        "elevation": "331 m",
        "climate": "hot semi-arid",
        "lat": 34.02,
    },
    "Quetta": {
        "region": "northern Balochistan",
        "elevation": "1680 m",
        "climate": "cold semi-arid",
        "lat": 30.18,
    },
    "Topi": {
        "region": "Swabi district, KPK",
        "elevation": "305 m",
        "climate": "semi-arid",
        "lat": 34.07,
    },
}

SEASONS = {
    "winter":     {"months": "December–February", "urdu": "sardi"},
    "spring":     {"months": "March–April",        "urdu": "bahar"},
    "pre-monsoon":{"months": "May–June",           "urdu": "garmi"},
    "monsoon":    {"months": "July–September",     "urdu": "baarish"},
    "post-monsoon":{"months": "October–November",  "urdu": "khushk"},
}

EVENT_TYPES = [
    "temperature_extreme",
    "rainfall_event",
    "wind_event",
    "fog_smog",
    "humidity_comfort",
]

# ── Templates (one per event type × 2 variants = 10 total) ───────────────────
# Placeholders: {city}, {region}, {climate}, {elevation}, {season}, {months},
#               {urdu}, {temp_hi}, {temp_lo}, {rain_mm}, {wind_kmh}

TEMPLATES = {
    "temperature_extreme": [
        (
            "{city} ({region}) experiences its {season} temperature extremes during {months}. "
            "With a {climate} climate at {elevation}, daily highs can reach {temp_hi}°C while "
            "nights cool to {temp_lo}°C. Residents describe the {urdu} season as {feel_adj}."
        ),
        (
            "During {months}, {city}'s {climate} climate produces temperatures between "
            "{temp_lo}°C at night and {temp_hi}°C in the afternoon. {region} topography "
            "moderates extremes compared to the surrounding plains."
        ),
    ],
    "rainfall_event": [
        (
            "{city} receives significant rainfall during {months} ({season} season). "
            "Hourly intensities can exceed {rain_mm} mm/h during convective storms, "
            "causing urban flooding in low-lying {region} neighbourhoods. "
            "Annual contribution from this season is roughly {season_pct}% of total."
        ),
        (
            "Rainfall in {city} during {season} ({months}) averages {rain_mm} mm for the period. "
            "The {climate} climate means dry spells of 10–20 days are common between events. "
            "Farmers in {region} rely on this window for {crop} cultivation."
        ),
    ],
    "wind_event": [
        (
            "{city} is exposed to {wind_name} winds during {months}. "
            "Speeds average {wind_kmh} km/h with gusts to {gust_kmh} km/h, reducing "
            "apparent temperature and raising dust. The {region} geography channels "
            "and amplifies these winds through the {season} season."
        ),
        (
            "Sustained {wind_name} winds in {city} during {season} ({months}) can reach "
            "{wind_kmh} km/h, lifting fine dust from the surrounding {region} terrain. "
            "Visibility drops to under 2 km on peak dust days."
        ),
    ],
    "fog_smog": [
        (
            "{city} experiences {fog_type} conditions during {months} ({season}). "
            "Visibility can fall to {vis_m} metres before 10 AM, delaying road traffic "
            "and flights. The {climate} climate's {season} humidity profile makes "
            "{city} one of the most affected cities in {region}."
        ),
        (
            "Morning {fog_type} in {city} ({region}) during {season} forms when overnight "
            "temperatures drop sharply while surface humidity stays above 85%. "
            "Affected hours are typically 04:00–09:00 local time during {months}."
        ),
    ],
    "humidity_comfort": [
        (
            "Relative humidity in {city} during {season} ({months}) averages {rh_pct}%, "
            "making outdoor activities {comfort_adj}. The {climate} climate at {elevation} "
            "means {region} residents experience distinct seasonal humidity swings of "
            "up to 40 percentage points between {season} and the preceding season."
        ),
        (
            "Heat index in {city} during {months} combines {temp_hi}°C air temperature "
            "with {rh_pct}% relative humidity, producing a feels-like temperature of "
            "{feels_like}°C. Public health advisories are issued when the heat index "
            "exceeds 41°C in {region}."
        ),
    ],
}

# ── Per-city per-season numeric ranges ───────────────────────────────────────

def _city_season_params(city: str, season: str) -> dict:
    """Return plausible numeric values for a given city+season combo."""
    base = {
        "Karachi": {
            "winter":      dict(temp_hi=26, temp_lo=13, rain_mm=8,  wind_kmh=18, rh_pct=62),
            "spring":      dict(temp_hi=33, temp_lo=20, rain_mm=5,  wind_kmh=22, rh_pct=55),
            "pre-monsoon": dict(temp_hi=38, temp_lo=28, rain_mm=3,  wind_kmh=25, rh_pct=70),
            "monsoon":     dict(temp_hi=34, temp_lo=27, rain_mm=55, wind_kmh=28, rh_pct=82),
            "post-monsoon":dict(temp_hi=34, temp_lo=22, rain_mm=4,  wind_kmh=16, rh_pct=60),
        },
        "Lahore": {
            "winter":      dict(temp_hi=18, temp_lo=5,  rain_mm=30, wind_kmh=10, rh_pct=75),
            "spring":      dict(temp_hi=32, temp_lo=17, rain_mm=25, wind_kmh=18, rh_pct=48),
            "pre-monsoon": dict(temp_hi=42, temp_lo=28, rain_mm=12, wind_kmh=20, rh_pct=38),
            "monsoon":     dict(temp_hi=36, temp_lo=26, rain_mm=95, wind_kmh=22, rh_pct=75),
            "post-monsoon":dict(temp_hi=30, temp_lo=15, rain_mm=18, wind_kmh=10, rh_pct=55),
        },
        "Islamabad": {
            "winter":      dict(temp_hi=14, temp_lo=3,  rain_mm=60, wind_kmh=12, rh_pct=72),
            "spring":      dict(temp_hi=26, temp_lo=12, rain_mm=55, wind_kmh=20, rh_pct=55),
            "pre-monsoon": dict(temp_hi=38, temp_lo=24, rain_mm=30, wind_kmh=18, rh_pct=42),
            "monsoon":     dict(temp_hi=33, temp_lo=23, rain_mm=220,wind_kmh=25, rh_pct=78),
            "post-monsoon":dict(temp_hi=27, temp_lo=12, rain_mm=35, wind_kmh=12, rh_pct=52),
        },
        "Peshawar": {
            "winter":      dict(temp_hi=14, temp_lo=3,  rain_mm=38, wind_kmh=14, rh_pct=68),
            "spring":      dict(temp_hi=30, temp_lo=15, rain_mm=42, wind_kmh=22, rh_pct=45),
            "pre-monsoon": dict(temp_hi=42, temp_lo=26, rain_mm=18, wind_kmh=20, rh_pct=35),
            "monsoon":     dict(temp_hi=37, temp_lo=25, rain_mm=80, wind_kmh=24, rh_pct=68),
            "post-monsoon":dict(temp_hi=28, temp_lo=12, rain_mm=22, wind_kmh=14, rh_pct=48),
        },
        "Quetta": {
            "winter":      dict(temp_hi=7,  temp_lo=-5, rain_mm=55, wind_kmh=16, rh_pct=65),
            "spring":      dict(temp_hi=22, temp_lo=7,  rain_mm=35, wind_kmh=24, rh_pct=48),
            "pre-monsoon": dict(temp_hi=34, temp_lo=18, rain_mm=10, wind_kmh=22, rh_pct=28),
            "monsoon":     dict(temp_hi=30, temp_lo=16, rain_mm=30, wind_kmh=18, rh_pct=50),
            "post-monsoon":dict(temp_hi=22, temp_lo=5,  rain_mm=15, wind_kmh=15, rh_pct=42),
        },
        "Topi": {
            "winter":      dict(temp_hi=16, temp_lo=4,  rain_mm=45, wind_kmh=10, rh_pct=70),
            "spring":      dict(temp_hi=28, temp_lo=13, rain_mm=40, wind_kmh=18, rh_pct=52),
            "pre-monsoon": dict(temp_hi=40, temp_lo=25, rain_mm=20, wind_kmh=16, rh_pct=38),
            "monsoon":     dict(temp_hi=35, temp_lo=24, rain_mm=130,wind_kmh=20, rh_pct=74),
            "post-monsoon":dict(temp_hi=28, temp_lo=12, rain_mm=25, wind_kmh=11, rh_pct=50),
        },
    }
    p = base[city][season].copy()
    p["temp_lo"] = p["temp_lo"]
    p["gust_kmh"] = int(p["wind_kmh"] * 1.4)
    p["feels_like"] = min(55, int(p["temp_hi"] + (p["rh_pct"] - 40) * 0.1))
    return p


_WIND_NAMES = {
    "winter":      "westerly",
    "spring":      "nor'wester squall",
    "pre-monsoon": "Loo",
    "monsoon":     "southwesterly monsoon",
    "post-monsoon":"northerly",
}

_FOG_TYPES = {
    "winter":      "dense radiation fog",
    "spring":      "morning mist",
    "pre-monsoon": "dust haze",
    "monsoon":     "low stratus cloud",
    "post-monsoon":"shallow fog",
}

_CROPS = {
    "monsoon":     "kharif (rice, cotton, maize)",
    "winter":      "rabi (wheat, chickpea)",
    "spring":      "rabi harvest",
    "pre-monsoon": "early kharif sowing",
    "post-monsoon":"late kharif harvest",
}

_COMFORT = {
    "winter":      "pleasant", "spring": "warm but comfortable",
    "pre-monsoon": "oppressive", "monsoon": "sticky and humid",
    "post-monsoon":"mild",
}

_FEEL_ADJ = {
    "winter": "bitterly cold", "spring": "refreshing",
    "pre-monsoon": "scorching", "monsoon": "rainy and lush",
    "post-monsoon": "temperate",
}

_SEASON_PCT = {
    "winter": 20, "spring": 12, "pre-monsoon": 8,
    "monsoon": 52, "post-monsoon": 8,
}

_VIS_M = {
    "winter": 50, "spring": 300, "pre-monsoon": 800,
    "monsoon": 1500, "post-monsoon": 200,
}


def _fill(template: str, city: str, season: str) -> str:
    p = _city_season_params(city, season)
    c = CITIES[city]
    s = SEASONS[season]
    return template.format(
        city=city,
        region=c["region"],
        climate=c["climate"],
        elevation=c["elevation"],
        season=season,
        months=s["months"],
        urdu=s["urdu"],
        temp_hi=p["temp_hi"],
        temp_lo=p["temp_lo"],
        rain_mm=p["rain_mm"],
        wind_kmh=p["wind_kmh"],
        gust_kmh=p["gust_kmh"],
        rh_pct=p["rh_pct"],
        feels_like=p["feels_like"],
        wind_name=_WIND_NAMES[season],
        fog_type=_FOG_TYPES[season],
        crop=_CROPS[season],
        comfort_adj=_COMFORT[season],
        feel_adj=_FEEL_ADJ[season],
        season_pct=_SEASON_PCT[season],
        vis_m=_VIS_M[season],
    )


def generate_snippets() -> list[dict]:
    snippets = []
    for city in CITIES:
        for season in SEASONS:
            for event_idx, event_type in enumerate(EVENT_TYPES):
                templates = TEMPLATES[event_type]
                for t_idx, template in enumerate(templates):
                    snippet_id = f"gen_{city.lower()}_{season.replace('-','_')}_{event_type}_{t_idx}"
                    text = _fill(template, city, season)
                    snippets.append({
                        "id":   snippet_id,
                        "text": text,
                        "tags": [city.lower(), season, event_type,
                                 CITIES[city]["region"].split(",")[0].lower()],
                        "generated": True,
                    })
    return snippets


def run(force: bool = False) -> None:
    # Load existing hand-curated snippets
    existing: list[dict] = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)

    # Keep hand-curated (no "generated" key); drop generated if force
    hand_curated = [s for s in existing if not s.get("generated")]
    if not force:
        already_generated = [s for s in existing if s.get("generated")]
    else:
        already_generated = []

    if already_generated and not force:
        print(f"Generated snippets already present ({len(already_generated)}) — use --force to regenerate.")
        print(f"Total: {len(existing)} snippets")
        return

    generated = generate_snippets()
    combined  = hand_curated + generated
    OUTPUT_PATH.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

    print(f"Hand-curated kept : {len(hand_curated)}")
    print(f"Generated new     : {len(generated)}")
    print(f"Total snippets    : {len(combined)}")
    print(f"Saved -> {OUTPUT_PATH}")

    # Quick sanity: unique ids
    ids = [s["id"] for s in combined]
    assert len(ids) == len(set(ids)), "Duplicate IDs detected!"
    print("ID uniqueness: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if generated entries already exist.")
    args = parser.parse_args()
    run(force=args.force)
