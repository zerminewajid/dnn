import httpx
import cache
from typing import Any

BASE_WX = "https://api.open-meteo.com/v1"
BASE_GEO = "https://geocoding-api.open-meteo.com/v1"
BASE_AQI = "https://air-quality-api.open-meteo.com/v1"

PK_CITIES = {
    "karachi":    (24.8607, 67.0011),
    "lahore":     (31.5204, 74.3587),
    "islamabad":  (33.6844, 73.0479),
    "topi":       (34.0740, 72.6145),
    "peshawar":   (34.0150, 71.5249),
    "quetta":     (30.1798, 66.9750),
}

WMO_ART = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "🌧️",
    71: "❄️", 73: "❄️", 75: "❄️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


async def _geo(city: str) -> tuple[float, float, str, str]:
    """Returns (lat, lon, display_name, timezone_id)."""
    city_lower = city.lower()
    if city_lower in PK_CITIES:
        lat, lon = PK_CITIES[city_lower]
        return lat, lon, city.title(), "Asia/Karachi"
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_GEO}/search", params={"name": city, "count": 1, "language": "en", "format": "json"})
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            raise ValueError(f"City not found: {city}")
        res = results[0]
        return res["latitude"], res["longitude"], res["name"], res.get("timezone", "UTC")


async def get_current_weather(city: str, lat: float = None, lon: float = None, timezone: str = None) -> dict[str, Any]:
    cache_key = f"current:{city.lower()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    if lat is None or lon is None:
        lat, lon, city, tz = await _geo(city)
    else:
        tz = timezone or "Asia/Karachi"

    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,uv_index",
        "timezone": tz,
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_WX}/forecast", params=params)
        r.raise_for_status()
        data = r.json()

    cur = data["current"]
    result = {
        "city": city, "lat": lat, "lon": lon,
        "timezone": tz,
        "temperature": cur["temperature_2m"],
        "feels_like": cur["apparent_temperature"],
        "humidity": cur["relative_humidity_2m"],
        "precipitation": cur["precipitation"],
        "weather_code": cur["weather_code"],
        "weather_emoji": WMO_ART.get(cur["weather_code"], "🌡️"),
        "wind_speed": cur["wind_speed_10m"],
        "uv_index": cur["uv_index"],
        "time": cur["time"],
    }
    cache.set(cache_key, result)
    return result


async def get_hourly_forecast(city: str, hours: int = 24) -> dict[str, Any]:
    cache_key = f"hourly:{city.lower()}:{hours}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    lat, lon, city, tz = await _geo(city)
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
        "forecast_days": 2, "timezone": tz,
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_WX}/forecast", params=params)
        r.raise_for_status()
        data = r.json()

    h = data["hourly"]
    result = {
        "city": city,
        "hours": [
            {
                "time": h["time"][i],
                "temperature": h["temperature_2m"][i],
                "precipitation_probability": h["precipitation_probability"][i],
                "weather_code": h["weather_code"][i],
                "weather_emoji": WMO_ART.get(h["weather_code"][i], "🌡️"),
                "wind_speed": h["wind_speed_10m"][i],
            }
            for i in range(min(hours, len(h["time"])))
        ]
    }
    cache.set(cache_key, result)
    return result


async def get_minutely_rain(lat: float, lon: float) -> dict[str, Any]:
    cache_key = f"minutely:{lat:.2f}:{lon:.2f}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    params = {
        "latitude": lat, "longitude": lon,
        "minutely_15": "precipitation,precipitation_probability",
        "forecast_minutely_15": 4, "timezone": "Asia/Karachi",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_WX}/forecast", params=params)
        r.raise_for_status()
        data = r.json()

    m = data.get("minutely_15", {})
    result = {
        "minutes": [
            {
                "time": m["time"][i],
                "precipitation": m["precipitation"][i],
                "probability": m["precipitation_probability"][i],
            }
            for i in range(len(m.get("time", [])))
        ]
    }
    cache.set(cache_key, result, ttl=60)
    return result


async def get_severe_alerts(country_code: str = "PK") -> dict[str, Any]:
    return {
        "country": country_code,
        "alerts": [],
        "note": "No active severe weather alerts API available. Monitor PMD (pmd.gov.pk) for official alerts."
    }


async def get_activity_index(city: str, activity: str) -> dict[str, Any]:
    wx = await get_current_weather(city)
    temp = wx["temperature"]
    humidity = wx["humidity"]
    wind = wx["wind_speed"]
    precip = wx["precipitation"]
    uv = wx["uv_index"]

    activity_lower = activity.lower()
    score = 100

    if "cricket" in activity_lower:
        label = "Cricket"
        if precip > 0:   score -= 50
        if temp > 40:    score -= 30
        elif temp > 38:  score -= 20
        if wind > 40:    score -= 20
        elif wind > 30:  score -= 10
        if uv > 9:       score -= 10
    elif "chai" in activity_lower:
        label = "Chai Time"
        # Chai is always good; hot/rainy weather makes it even better
        if temp < 10:    score -= 10   # too cold to enjoy properly
        # Rain = bonus chai weather, hot = fine with chai
        if precip > 0:   score = min(100, score + 10)
    elif "rickshaw" in activity_lower:
        label = "Rickshaw Ride"
        if precip > 3:   score -= 40
        elif precip > 0: score -= 20
        if temp > 42:    score -= 30
        elif temp > 38:  score -= 15
        if wind > 50:    score -= 20
        elif wind > 40:  score -= 10
    elif "study" in activity_lower or "outdoor" in activity_lower:
        label = "Outdoor Study"
        if precip > 0:   score -= 40
        if temp > 38:    score -= 30
        elif temp > 35:  score -= 15
        if uv > 8:       score -= 20
        elif uv > 6:     score -= 10
        if wind > 35:    score -= 10
    elif "iftar" in activity_lower:
        label = "Iftar Walk"
        if precip > 0:   score -= 30
        if temp > 42:    score -= 30
        elif temp > 38:  score -= 15
        if wind > 40:    score -= 10
    else:
        label = activity.title()

    score = max(0, min(100, score))
    rating = "Perfect" if score >= 80 else "Good" if score >= 60 else "Okay" if score >= 40 else "Risky"
    return {"activity": label, "score": score, "rating": rating, "city": wx["city"], "temperature": temp}


async def search_city(query: str) -> list[dict]:
    cache_key = f"search:{query.lower()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_GEO}/search", params={"name": query, "count": 5, "language": "en", "format": "json"})
        r.raise_for_status()
        results = r.json().get("results", [])

    cities = [
        {"name": c["name"], "country": c.get("country", ""), "lat": c["latitude"], "lon": c["longitude"], "timezone": c.get("timezone", "UTC")}
        for c in results
    ]
    cache.set(cache_key, cities, ttl=3600)
    return cities


async def get_7day_forecast(city: str) -> dict[str, Any]:
    cache_key = f"7day:{city.lower()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    lat, lon, city, tz = await _geo(city)
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,uv_index_max",
        "forecast_days": 7, "timezone": tz,
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_WX}/forecast", params=params)
        r.raise_for_status()
        data = r.json()

    d = data["daily"]
    result = {
        "city": city,
        "days": [
            {
                "date": d["time"][i],
                "temp_max": d["temperature_2m_max"][i],
                "temp_min": d["temperature_2m_min"][i],
                "precipitation": d["precipitation_sum"][i],
                "weather_code": d["weather_code"][i],
                "weather_emoji": WMO_ART.get(d["weather_code"][i], "🌡️"),
                "uv_max": d["uv_index_max"][i],
            }
            for i in range(len(d["time"]))
        ]
    }
    cache.set(cache_key, result)
    return result


async def get_air_quality(lat: float, lon: float) -> dict[str, Any]:
    cache_key = f"aqi:{lat:.2f}:{lon:.2f}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    params = {
        "latitude": lat, "longitude": lon,
        "current": "pm10,pm2_5,us_aqi",
        "timezone": "Asia/Karachi",
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_AQI}/air-quality", params=params)
        r.raise_for_status()
        data = r.json()

    cur = data.get("current", {})
    aqi = cur.get("us_aqi", 0)
    label = (
        "Good 🟢" if aqi <= 50 else
        "Moderate 🟡" if aqi <= 100 else
        "Unhealthy for Sensitive 🟠" if aqi <= 150 else
        "Unhealthy 🔴" if aqi <= 200 else
        "Very Unhealthy 🟣"
    )
    result = {"pm10": cur.get("pm10"), "pm2_5": cur.get("pm2_5"), "us_aqi": aqi, "label": label}
    cache.set(cache_key, result)
    return result
