import { useState, useEffect, useCallback, useRef } from 'react'

const PK_CITIES = [
  { name: 'Lahore',     lat: 31.5204, lon: 74.3587, timezone: 'Asia/Karachi' },
  { name: 'Karachi',   lat: 24.8607, lon: 67.0011, timezone: 'Asia/Karachi' },
  { name: 'Islamabad', lat: 33.6844, lon: 73.0479, timezone: 'Asia/Karachi' },
  { name: 'Topi',      lat: 34.074,  lon: 72.6145, timezone: 'Asia/Karachi' },
  { name: 'Peshawar',  lat: 34.015,  lon: 71.5249, timezone: 'Asia/Karachi' },
  { name: 'Quetta',    lat: 30.1798, lon: 66.975,  timezone: 'Asia/Karachi' },
]

export function useWeather() {
  const [city, setCity]       = useState('Lahore')
  const [current, setCurrent] = useState(null)
  const [hourly, setHourly]   = useState(null)
  const [forecast7, setForecast7] = useState(null)
  const [aqi, setAqi]         = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const coordsRef = useRef({ lat: null, lon: null, timezone: null })

  const fetchJson = async (url) => {
    const res = await fetch(url)
    const text = await res.text()
    if (!text) throw new Error(`Empty response from server (${res.status})`)
    let data
    try { data = JSON.parse(text) } catch { throw new Error('Invalid JSON from server') }
    if (!res.ok) throw new Error(data?.detail || `Request failed (${res.status})`)
    return data
  }

  const fetchWeather = useCallback(async (cityName, lat = null, lon = null, timezone = null) => {
    setLoading(true)
    setError(null)
    const coordSuffix = (lat != null && lon != null) ? `&lat=${lat}&lon=${lon}` : ''
    const tzSuffix    = timezone ? `&timezone=${encodeURIComponent(timezone)}` : ''
    try {
      const [cur, hr, day7] = await Promise.all([
        fetchJson(`/api/weather/current?city=${encodeURIComponent(cityName)}${coordSuffix}${tzSuffix}`),
        fetchJson(`/api/weather/hourly?city=${encodeURIComponent(cityName)}&hours=24`),
        fetchJson(`/api/weather/7day?city=${encodeURIComponent(cityName)}`),
      ])
      setCurrent(cur)
      setHourly(hr)
      setForecast7(day7)

      if (cur.lat && cur.lon) {
        const aqiData = await fetchJson(`/api/weather/aqi?lat=${cur.lat}&lon=${cur.lon}`)
        setAqi(aqiData)
      }
    } catch (e) {
      setError(e?.message || 'Failed to load weather data')
      setCurrent(null); setHourly(null); setForecast7(null); setAqi(null)
    } finally {
      setLoading(false)
    }
  }, [])

  // selectCity: use when picking from search results (has lat/lon/timezone)
  const selectCity = useCallback((name, lat = null, lon = null, timezone = null) => {
    coordsRef.current = { lat, lon, timezone }
    setCity(name)
    fetchWeather(name, lat, lon, timezone)
  }, [fetchWeather])

  useEffect(() => {
    fetchWeather('Lahore', 31.5204, 74.3587, 'Asia/Karachi')
  }, []) // only on mount

  return {
    city, setCity, selectCity,
    current, hourly, forecast7, aqi,
    loading, error,
    pkCities: PK_CITIES,
    refetch: () => {
      const { lat, lon, timezone } = coordsRef.current
      fetchWeather(city, lat, lon, timezone)
    },
  }
}
