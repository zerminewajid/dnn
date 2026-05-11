import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

const ACTIVITIES = [
  { key: 'cricket',        label: 'Cricket',        emoji: '🏏', urdu: 'کرکٹ' },
  { key: 'chai',           label: 'Chai Time',      emoji: '☕', urdu: 'چائے' },
  { key: 'rickshaw',       label: 'Rickshaw',       emoji: '🛺', urdu: 'رکشہ' },
  { key: 'outdoor study',  label: 'Outdoor Study',  emoji: '📚', urdu: 'پڑھائی' },
  { key: 'iftar walk',     label: 'Iftar Walk',     emoji: '🌙', urdu: 'افطار' },
]

function scoreColor(score) {
  if (score >= 75) return '#34d399'
  if (score >= 50) return '#fbbf24'
  if (score >= 30) return '#f97316'
  return '#f87171'
}

function scoreLabel(score) {
  if (score >= 80) return 'Perfect'
  if (score >= 60) return 'Good'
  if (score >= 40) return 'Okay'
  return 'Risky'
}

export default function ActivityIndex({ city, eyeColor }) {
  const [activities, setActivities] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!city) return
    setLoading(true)
    Promise.all(
      ACTIVITIES.map(a =>
        fetch(`/api/activity?city=${encodeURIComponent(city)}&activity=${encodeURIComponent(a.key)}`)
          .then(r => r.json())
          .then(data => ({ ...a, ...data }))
          .catch(() => ({ ...a, score: 0, rating: 'Unknown' }))
      )
    ).then(results => {
      setActivities(results)
      setLoading(false)
    })
  }, [city])

  return (
    <div className="glass-card p-5 h-full">
      <div className="flex items-center justify-between mb-5">
        <p
          className="font-body text-xs font-medium uppercase tracking-widest"
          style={{ color: 'var(--gold)', letterSpacing: '0.12em' }}
        >
          Activity Index
        </p>
        <p className="font-urdu text-xs" style={{ color: 'var(--text-muted)' }}>
          {city}
        </p>
      </div>

      {loading ? (
        <div className="space-y-4">
          {ACTIVITIES.map((_, i) => (
            <div key={i} className="flex items-center gap-3 animate-pulse">
              <div className="w-8 h-8 rounded-xl bg-white/5" />
              <div className="flex-1 h-2 rounded-full bg-white/5" />
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {activities.map((a, i) => {
            const score = a.score || 0
            const color = scoreColor(score)
            const label = scoreLabel(score)
            return (
              <motion.div
                key={a.key}
                initial={{ opacity: 0, x: -12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.08, duration: 0.4 }}
                className="group"
              >
                <div className="flex items-center gap-3 mb-1.5">
                  <div
                    className="w-8 h-8 rounded-xl flex items-center justify-center shrink-0 transition-transform duration-200 group-hover:scale-110"
                    style={{ background: `${color}18`, border: `1px solid ${color}30` }}
                  >
                    <span className="text-base">{a.emoji}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-body text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                        {a.label}
                      </span>
                      <div className="flex items-center gap-2 shrink-0">
                        <span className="font-body text-xs" style={{ color }}>
                          {label}
                        </span>
                        <span className="font-mono text-xs font-medium" style={{ color, minWidth: '2.5rem', textAlign: 'right' }}>
                          {score}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Progress bar */}
                <div
                  className="h-1.5 rounded-full ml-11"
                  style={{ background: 'rgba(255,255,255,0.07)' }}
                >
                  <motion.div
                    className="h-full rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${score}%` }}
                    transition={{ delay: i * 0.08 + 0.2, duration: 0.8, ease: 'easeOut' }}
                    style={{
                      background: `linear-gradient(90deg, ${color}bb, ${color})`,
                      boxShadow: `0 0 6px ${color}50`,
                    }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      )}
    </div>
  )
}
