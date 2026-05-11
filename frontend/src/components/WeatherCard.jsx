import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

function useClock(timezone) {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const fmtTime = (tz) => now.toLocaleTimeString('en-GB', {
    timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  })
  const fmtDate = (tz) => now.toLocaleDateString('en-GB', {
    timeZone: tz, weekday: 'short', day: 'numeric', month: 'short',
  })
  const offsetLabel = (tz) => {
    try {
      const parts = new Intl.DateTimeFormat('en', { timeZone: tz, timeZoneName: 'short' })
        .formatToParts(now)
      return parts.find(p => p.type === 'timeZoneName')?.value || ''
    } catch { return '' }
  }

  return { cityTime: fmtTime(timezone), cityDate: fmtDate(timezone), cityOffset: offsetLabel(timezone) }
}

function StatCard({ label, value, unit, sub, accent, delay = 0, bar, barColor }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="glass-card p-4 flex flex-col gap-1"
    >
      <p className="font-body text-xs font-medium uppercase tracking-widest" style={{ color: 'var(--text-muted)', letterSpacing: '0.1em' }}>
        {label}
      </p>
      <div className="flex items-end gap-1 mt-1">
        <span className="font-display font-semibold leading-none" style={{ fontSize: '2rem', color: accent || 'var(--text-primary)' }}>
          {value}
        </span>
        {unit && <span className="font-body text-xs mb-1.5" style={{ color: 'var(--text-secondary)' }}>{unit}</span>}
      </div>
      {sub && <p className="font-body text-xs" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
      {bar !== undefined && (
        <div className="mt-2 h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.08)' }}>
          <motion.div
            className="h-full rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(bar, 100)}%` }}
            transition={{ delay: delay + 0.2, duration: 0.7, ease: 'easeOut' }}
            style={{ background: barColor || 'var(--rain-blue)' }}
          />
        </div>
      )}
    </motion.div>
  )
}

export default function WeatherCard({ current, aqi, eyeColor }) {
  if (!current) return null

  const { temperature, feels_like, humidity, wind_speed, uv_index, weather_emoji, city, precipitation, timezone } = current
  const { cityTime, cityDate, cityOffset } = useClock(timezone || 'UTC')

  const uvLabel = uv_index <= 2 ? 'Low' : uv_index <= 5 ? 'Moderate' : uv_index <= 7 ? 'High' : 'Very High'
  const uvColor = uv_index <= 2 ? '#34d399' : uv_index <= 5 ? '#fbbf24' : uv_index <= 7 ? '#f97316' : '#ef4444'

  const aqiColor = !aqi ? '#34d399'
    : aqi.us_aqi <= 50 ? '#34d399'
    : aqi.us_aqi <= 100 ? '#fbbf24'
    : aqi.us_aqi <= 150 ? '#f97316'
    : '#ef4444'

  return (
    <div className="space-y-3">
      {/* ── Main hero card ──────────────────────── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="glass-card p-6 relative overflow-hidden"
      >
        {/* Background emoji watermark */}
        <div
          className="absolute right-6 top-1/2 -translate-y-1/2 pointer-events-none select-none"
          style={{ fontSize: '7rem', opacity: 0.08, filter: 'blur(2px)' }}
        >
          {weather_emoji}
        </div>

        <div className="relative z-10 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <p className="font-body text-sm font-medium" style={{ color: 'var(--gold)' }}>{city}</p>
              <span className="text-xl">{weather_emoji}</span>
            </div>
            {/* Dual clock — city local time + offset */}
            <div className="flex items-center gap-3 mb-2">
              <div>
                <p className="font-mono text-lg font-semibold leading-none" style={{ color: 'var(--text-primary)' }}>
                  {cityTime}
                </p>
                <p className="font-body text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  {cityDate} · <span style={{ color: eyeColor, opacity: 0.8 }}>{cityOffset}</span>
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span
                className="font-display font-light leading-none"
                style={{ fontSize: 'clamp(4rem, 10vw, 6rem)', color: 'var(--text-primary)' }}
              >
                {Math.round(temperature)}
              </span>
              <span className="font-display font-light mt-3" style={{ fontSize: '2rem', color: 'var(--text-secondary)' }}>°C</span>
            </div>
            <p className="font-body text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
              Feels like {Math.round(feels_like)}°C
            </p>
          </div>

          {/* Rain indicator */}
          {precipitation > 0 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center gap-1 px-4 py-3 rounded-2xl"
              style={{ background: 'rgba(74,143,255,0.12)', border: '1px solid rgba(74,143,255,0.25)' }}
            >
              <span className="text-2xl">🌧️</span>
              <p className="font-mono text-xs" style={{ color: 'var(--rain-blue)' }}>{precipitation}mm</p>
            </motion.div>
          )}
        </div>

        {/* Inner glow strip */}
        <div
          className="absolute bottom-0 left-0 right-0 h-px"
          style={{ background: `linear-gradient(90deg, transparent, ${eyeColor || 'rgba(255,255,255,0.15)'}40, transparent)` }}
        />
      </motion.div>

      {/* ── Stat grid ────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard
          label="Humidity"
          value={humidity}
          unit="%"
          delay={0.1}
          bar={humidity}
          barColor="var(--rain-blue)"
        />
        <StatCard
          label="Wind"
          value={Math.round(wind_speed)}
          unit="km/h"
          sub="💨 surface wind"
          delay={0.15}
        />
        <StatCard
          label="UV Index"
          value={uv_index}
          sub={`${uvLabel} ☀️`}
          accent={uvColor}
          delay={0.2}
          bar={(uv_index / 11) * 100}
          barColor={uvColor}
        />
        {aqi && (
          <StatCard
            label="Air Quality"
            value={aqi.us_aqi}
            sub={aqi.label}
            accent={aqiColor}
            delay={0.25}
            bar={Math.min((aqi.us_aqi / 200) * 100, 100)}
            barColor={aqiColor}
          />
        )}
      </div>
    </div>
  )
}
