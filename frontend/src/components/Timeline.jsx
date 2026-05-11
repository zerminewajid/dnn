import { useRef, useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'

export default function Timeline({ hourly, onDrag, eyeColor, timezone }) {
  const containerRef = useRef(null)

  if (!hourly?.hours) return null
  const hours = hourly.hours.slice(0, 24)

  // Find the index closest to the current hour IN THE CITY'S TIMEZONE
  const currentIdx = useMemo(() => {
    const tz = timezone || Intl.DateTimeFormat().resolvedOptions().timeZone
    const cityHour = parseInt(
      new Intl.DateTimeFormat('en-US', { timeZone: tz, hour: 'numeric', hour12: false }).format(new Date()),
      10
    ) % 24
    let best = 0
    for (let i = 0; i < hours.length; i++) {
      // Open-Meteo returns naive local datetime strings — getHours() gives the city's hour
      const h = new Date(hours[i].time).getHours()
      if (h <= cityHour) best = i
      else break
    }
    return best
  }, [hours.map(h => h.time).join(','), timezone])

  const [activeIndex, setActiveIndex] = useState(currentIdx)

  // When city changes (new hours array), jump to current hour and scroll
  useEffect(() => {
    setActiveIndex(currentIdx)
    requestAnimationFrame(() => {
      const el = containerRef.current?.querySelector(`[data-idx="${currentIdx}"]`)
      el?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
    })
  }, [currentIdx])

  const handleClick = (idx) => {
    setActiveIndex(idx)
    onDrag?.(hours[idx])
  }

  const active = hours[activeIndex]

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-5">
        <p
          className="font-body text-xs font-medium uppercase tracking-widest"
          style={{ color: 'var(--gold)', letterSpacing: '0.12em' }}
        >
          Hourly Forecast
        </p>
        {active && (
          <motion.div
            key={activeIndex}
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-2"
          >
            <span className="text-lg">{active.weather_emoji}</span>
            <span className="font-display font-semibold text-white text-lg">{Math.round(active.temperature)}°C</span>
            {active.precipitation_probability > 20 && (
              <span className="font-mono text-xs" style={{ color: 'var(--rain-blue)' }}>
                {active.precipitation_probability}% 💧
              </span>
            )}
          </motion.div>
        )}
      </div>

      {/* Scrollable track */}
      <div ref={containerRef} className="overflow-x-auto pb-2">
        <div className="flex gap-2 min-w-max">
          {hours.map((hour, i) => {
            const time = new Date(hour.time)
            const label = time.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', hour12: true })
            const isActive = i === activeIndex
            const isCurrent = i === currentIdx
            const rainChance = hour.precipitation_probability || 0

            return (
              <motion.button
                key={i}
                data-idx={i}
                onClick={() => handleClick(i)}
                whileHover={{ scale: 1.04, y: -2 }}
                whileTap={{ scale: 0.96 }}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.02 }}
                className="flex flex-col items-center gap-2 px-3.5 py-3 rounded-2xl transition-all duration-200 min-w-[64px] relative overflow-hidden"
                style={{
                  background: isActive
                    ? `rgba(${hexToRgb(eyeColor) || '232,184,75'},0.15)`
                    : 'rgba(255,255,255,0.04)',
                  border: isActive
                    ? `1px solid ${eyeColor || 'var(--gold)'}60`
                    : isCurrent
                    ? '1px solid rgba(255,255,255,0.2)'
                    : '1px solid rgba(255,255,255,0.06)',
                  boxShadow: isActive ? `0 0 20px ${eyeColor || 'var(--gold)'}20` : 'none',
                }}
              >
                {/* Rain probability fill */}
                {rainChance > 20 && (
                  <div
                    className="absolute bottom-0 left-0 right-0 rounded-b-2xl"
                    style={{
                      height: `${rainChance * 0.35}%`,
                      background: 'rgba(74,143,255,0.18)',
                    }}
                  />
                )}

                <p className="font-body text-xs relative z-10" style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {isCurrent && !isActive ? '▶ now' : label}
                </p>
                <span className="text-xl relative z-10">{hour.weather_emoji}</span>
                <p
                  className="font-display font-semibold text-sm relative z-10"
                  style={{ color: isActive ? (eyeColor || 'var(--gold)') : 'var(--text-primary)' }}
                >
                  {Math.round(hour.temperature)}°
                </p>
                {rainChance > 20 && (
                  <p className="font-mono text-[10px] relative z-10" style={{ color: 'var(--rain-blue)' }}>
                    {rainChance}%
                  </p>
                )}
              </motion.button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function hexToRgb(hex) {
  if (!hex) return null
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result ? `${parseInt(result[1],16)},${parseInt(result[2],16)},${parseInt(result[3],16)}` : null
}
