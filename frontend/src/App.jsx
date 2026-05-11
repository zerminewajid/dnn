import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Zero from './components/Zero'
import WeatherCard from './components/WeatherCard'
import Timeline from './components/Timeline'
import ChatPanel from './components/ChatPanel'
import ActivityIndex from './components/ActivityIndex'
import { useWeather } from './hooks/useWeather'
import { useVoice } from './hooks/useVoice'

/* ── Background config: temp + weather code + time ── */
function getBgConfig(temp, weatherCode) {
  const hour = new Date().getHours()
  const isNight = hour < 6 || hour >= 20
  const isDusk  = (hour >= 17 && hour < 20) || (hour >= 6 && hour < 8)
  const isRainy = weatherCode >= 51 || [45, 48, 80, 81, 82, 95, 96, 99].includes(weatherCode)
  const isCloudy = weatherCode >= 2 && weatherCode <= 3

  if (isRainy) return {
    base: '#04101e',
    blobs: ['rgba(20,60,140,0.7)', 'rgba(10,40,100,0.6)', 'rgba(30,80,160,0.4)'],
    rainOpacity: 1,
  }

  if (temp === undefined || temp === null) return {
    base: '#060d1f',
    blobs: ['rgba(30,50,120,0.55)', 'rgba(10,30,80,0.45)', 'rgba(60,30,100,0.35)'],
    rainOpacity: 0.4,
  }

  if (isNight) return {
    base: '#04071a',
    blobs: ['rgba(20,20,80,0.6)', 'rgba(40,10,80,0.5)', 'rgba(10,30,60,0.4)'],
    rainOpacity: 0.3,
  }

  if (isDusk) return {
    base: '#120808',
    blobs: ['rgba(140,50,20,0.55)', 'rgba(100,30,60,0.5)', 'rgba(160,80,20,0.35)'],
    rainOpacity: 0.2,
  }

  if (isCloudy) return {
    base: '#080e1a',
    blobs: ['rgba(40,60,100,0.55)', 'rgba(20,50,80,0.5)', 'rgba(50,70,110,0.35)'],
    rainOpacity: 0.04,
    cloudy: true,
  }

  if (temp < 15) return {
    base: '#06091a',
    blobs: ['rgba(40,20,120,0.6)', 'rgba(10,30,100,0.5)', 'rgba(80,20,120,0.4)'],
    rainOpacity: 0.6,
  }
  if (temp < 25) return {
    base: '#060f1e',
    blobs: ['rgba(10,50,120,0.55)', 'rgba(20,80,120,0.45)', 'rgba(10,60,100,0.35)'],
    rainOpacity: 0.45,
  }
  if (temp < 35) return {
    base: '#0f0b03',
    blobs: ['rgba(120,70,10,0.5)', 'rgba(80,50,10,0.45)', 'rgba(140,80,20,0.35)'],
    rainOpacity: 0.15,
  }
  return {
    base: '#130500',
    blobs: ['rgba(160,30,10,0.55)', 'rgba(120,20,5,0.5)', 'rgba(180,60,10,0.35)'],
    rainOpacity: 0.0,
  }
}

const RAIN_PARTICLES = Array.from({ length: 22 }, (_, i) => ({
  id: i,
  left: `${Math.random() * 100}%`,
  height: `${Math.random() * 60 + 60}px`,
  duration: `${Math.random() * 1.5 + 1.2}s`,
  delay: `${Math.random() * 3}s`,
  opacity: Math.random() * 0.4 + 0.15,
}))

const PK_CITIES = ['Lahore', 'Karachi', 'Islamabad', 'Topi', 'Peshawar', 'Quetta']

export default function App() {
  const { city, setCity, selectCity, current, hourly, forecast7, aqi, loading, error, pkCities } = useWeather()
  const { speak } = useVoice()

  const [eyeColor,      setEyeColor]      = useState('#38BDF8')
  const [messages,      setMessages]      = useState([])
  const [isThinking,    setIsThinking]    = useState(false)
  const [directiveMode, setDirectiveMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [wsReady,       setWsReady]       = useState(false)
  const wsRef = useRef(null)

  const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/chat`

  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL)
    ws.onopen = () => {
      wsRef.current = ws
      setWsReady(true)
    }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.type === 'thinking') {
        setIsThinking(true)
      } else if (data.type === 'response') {
        setIsThinking(false)
        setMessages(prev => [...prev, { role: 'zero', text: data.text }])
        speak(data.text)
      } else if (data.type === 'error') {
        setIsThinking(false)
        setMessages(prev => [...prev, { role: 'zero', text: data.text }])
      }
    }
    ws.onclose = () => {
      setWsReady(false)
      setTimeout(connectWS, 2000)
    }
    wsRef.current = ws
  }, [WS_URL])

  useEffect(() => {
    connectWS()
    return () => wsRef.current?.close()
  }, [connectWS])


  const handleSearch = async (q) => {
    setSearchQuery(q)
    if (q.length < 2) { setSearchResults([]); return }
    try {
      const r = await fetch(`/api/search?q=${encodeURIComponent(q)}`).then(r => r.json())
      setSearchResults(r)
    } catch {}
  }

  const temperature = current?.temperature
  const bgConfig = getBgConfig(temperature, current?.weather_code)
  const isRaining = (current?.precipitation > 0) || (current?.weather_code >= 51)
  const isCloudy = bgConfig.cloudy

  return (
    <div
      className="min-h-screen relative overflow-x-hidden"
      style={{ background: bgConfig.base, transition: 'background 2s ease' }}
    >
      {/* ── Gradient mesh blobs ──────────────────── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        <div
          className="absolute w-[700px] h-[700px] rounded-full blur-[120px] animate-mesh-drift"
          style={{ background: bgConfig.blobs[0], top: '-200px', right: '-100px', opacity: 0.8 }}
        />
        <div
          className="absolute w-[600px] h-[600px] rounded-full blur-[100px] animate-mesh-drift-2"
          style={{ background: bgConfig.blobs[1], bottom: '-150px', left: '-100px', opacity: 0.7 }}
        />
        <div
          className="absolute w-[400px] h-[400px] rounded-full blur-[90px] animate-mesh-drift-3"
          style={{ background: bgConfig.blobs[2], top: '40%', left: '40%', opacity: 0.5 }}
        />
      </div>

      {/* ── Rain particles ────────────────────────── */}
      <div
        className="fixed inset-0 pointer-events-none z-[1] transition-opacity duration-2000"
        style={{ opacity: isRaining ? 1 : bgConfig.rainOpacity }}
      >
        {RAIN_PARTICLES.map(p => (
          <div
            key={p.id}
            className="rain-particle"
            style={{
              left: p.left,
              height: p.height,
              animationDuration: p.duration,
              animationDelay: p.delay,
              opacity: p.opacity,
            }}
          />
        ))}
      </div>

      {/* ── Cloud puffs (cloudy weather only) ────── */}
      {isCloudy && (
        <div className="fixed inset-0 pointer-events-none z-[1]">
          <div className="absolute rounded-full blur-[60px]" style={{ width: 400, height: 180, top: '8%', left: '5%', background: 'rgba(160,180,220,0.13)' }} />
          <div className="absolute rounded-full blur-[80px]" style={{ width: 500, height: 200, top: '12%', right: '10%', background: 'rgba(140,165,210,0.10)' }} />
          <div className="absolute rounded-full blur-[50px]" style={{ width: 300, height: 140, top: '22%', left: '35%', background: 'rgba(180,195,230,0.08)' }} />
        </div>
      )}

      {/* ── Bioluminescent Zero overlay ───────────── */}
      <div
        className="fixed inset-0 pointer-events-none z-[2] transition-all duration-1000"
        style={{
          background: `radial-gradient(ellipse 60% 50% at 50% 0%, ${eyeColor}18 0%, transparent 70%)`,
        }}
      />

      {/* ── Main content ─────────────────────────── */}
      <div className="relative z-10 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-10">

        {/* ── Header ───────────────────────────────── */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="flex items-center justify-between mb-8"
        >
          <div>
            <h1
              className="font-display text-white leading-none"
              style={{ fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', fontWeight: 600, letterSpacing: '-0.01em' }}
            >
              Weathering With You
            </h1>
            <p className="font-urdu text-xs mt-1" style={{ color: 'var(--gold)', opacity: 0.75 }}>
              موسم کی دنیا
            </p>
          </div>
          <div className="flex items-center gap-3">
            <motion.button
              onClick={() => setDirectiveMode(v => !v)}
              whileTap={{ scale: 0.94 }}
              className="text-xs px-4 py-2 rounded-full font-body font-medium transition-all duration-300"
              style={{
                background: directiveMode ? 'rgba(239,68,68,0.18)' : 'var(--glass)',
                border: directiveMode ? '1px solid rgba(239,68,68,0.45)' : '1px solid var(--glass-border)',
                color: directiveMode ? '#fca5a5' : 'var(--text-secondary)',
                backdropFilter: 'blur(12px)',
              }}
            >
              {directiveMode ? '⬤ Alerts Only' : '◯ All Weather'}
            </motion.button>
          </div>
        </motion.header>

        {/* ── City pills + Search ───────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-6 space-y-3"
        >
          <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
            {PK_CITIES.map((c, i) => (
              <motion.button
                key={c}
                onClick={() => { const cityObj = pkCities.find(x => x.name === c); selectCity(c, cityObj?.lat, cityObj?.lon, cityObj?.timezone) }}
                whileTap={{ scale: 0.93 }}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15 + i * 0.05 }}
                className="text-xs px-4 py-2 rounded-full whitespace-nowrap font-body font-medium transition-all duration-300"
                style={{
                  background: city === c ? 'rgba(232,184,75,0.18)' : 'var(--glass)',
                  border: city === c ? '1px solid rgba(232,184,75,0.5)' : '1px solid var(--glass-border)',
                  color: city === c ? 'var(--gold)' : 'var(--text-secondary)',
                  backdropFilter: 'blur(12px)',
                  boxShadow: city === c ? '0 0 16px rgba(232,184,75,0.15)' : 'none',
                }}
              >
                {c}
              </motion.button>
            ))}
          </div>

          {/* Search */}
          <div className="relative">
            <input
              value={searchQuery}
              onChange={e => handleSearch(e.target.value)}
              placeholder="Search any city..."
              className="w-full text-sm font-body px-4 py-3 rounded-2xl transition-all duration-300"
              style={{
                background: 'var(--glass)',
                border: '1px solid var(--glass-border)',
                color: 'var(--text-primary)',
                backdropFilter: 'blur(20px)',
              }}
              onFocus={e => e.target.style.borderColor = 'rgba(232,184,75,0.35)'}
              onBlur={e => e.target.style.borderColor = 'var(--glass-border)'}
            />
            <AnimatePresence>
              {searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 6 }}
                  className="absolute top-full mt-2 left-0 right-0 z-30 overflow-hidden rounded-2xl"
                  style={{
                    background: 'rgba(6,13,31,0.97)',
                    border: '1px solid var(--glass-border-bright)',
                    backdropFilter: 'blur(32px)',
                    boxShadow: '0 16px 40px rgba(0,0,0,0.4)',
                  }}
                >
                  {searchResults.map((r, i) => (
                    <button
                      key={i}
                      onClick={() => { selectCity(r.name, r.lat, r.lon, r.timezone); setSearchQuery(''); setSearchResults([]) }}
                      className="w-full text-left px-4 py-3 text-sm font-body transition-all duration-200"
                      style={{ color: 'var(--text-secondary)', borderBottom: i < searchResults.length - 1 ? '1px solid var(--glass-border)' : 'none' }}
                      onMouseEnter={e => { e.currentTarget.style.background = 'var(--glass)'; e.currentTarget.style.color = 'white' }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)' }}
                    >
                      {r.name}, <span style={{ color: 'var(--text-muted)' }}>{r.country}</span>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

        {/* ── Main grid: left + right (desktop) ────── */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5 mb-5">

          {/* LEFT COLUMN */}
          <div className="flex flex-col gap-5">
            {/* Zero hero */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex justify-center py-8 glass-card"
            >
              <Zero
                weatherData={current}
                botSpeaking={isThinking}
                onMoodChange={setEyeColor}
              />
            </motion.div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="rounded-2xl px-4 py-3 text-sm font-body"
                  style={{
                    background: 'rgba(248,113,113,0.1)',
                    border: '1px solid rgba(248,113,113,0.3)',
                    color: '#fecaca',
                  }}
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Weather cards */}
            {loading ? (
              <div className="flex justify-center py-12">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
                  className="w-8 h-8 rounded-full border-2"
                  style={{ borderColor: 'rgba(232,184,75,0.2)', borderTopColor: 'var(--gold)' }}
                />
              </div>
            ) : (
              <WeatherCard current={current} aqi={aqi} forecast7={forecast7} eyeColor={eyeColor} />
            )}
          </div>

          {/* RIGHT COLUMN — Chat */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="lg:sticky lg:top-6 lg:self-start"
            style={{ height: '580px' }}
          >
            <ChatPanel
              ws={wsReady ? wsRef.current : null}
              messages={messages}
              onMessage={(msg) => setMessages(prev => [...prev, msg])}
              isThinking={isThinking}
              eyeColor={eyeColor}
            />
          </motion.div>
        </div>

        {/* ── Timeline (full width) ─────────────────── */}
        {hourly && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="mb-5"
          >
            <Timeline hourly={hourly} onDrag={() => {}} eyeColor={eyeColor} timezone={current?.timezone} />
          </motion.div>
        )}

        {/* ── Bottom grid: 7-day + Activity ────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-5 mb-8">
          {forecast7 && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              <SevenDayCard forecast7={forecast7} />
            </motion.div>
          )}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.55 }}>
            <ActivityIndex city={city} eyeColor={eyeColor} />
          </motion.div>
        </div>

        {/* ── Footer ───────────────────────────────── */}
        <footer className="text-center pb-6">
          <p className="font-body text-xs" style={{ color: 'var(--text-muted)' }}>
            Zermine Wajid · zermeenewajid@outlook.com
          </p>
        </footer>
      </div>
    </div>
  )
}

/* ── 7-Day card (extracted for layout clarity) ──── */
function SevenDayCard({ forecast7 }) {
  if (!Array.isArray(forecast7?.days)) return null
  return (
    <div className="glass-card p-5 h-full">
      <p
        className="font-body text-xs font-medium uppercase tracking-widest mb-5"
        style={{ color: 'var(--gold)', letterSpacing: '0.12em' }}
      >
        7-Day Forecast
      </p>
      <div className="grid grid-cols-7 gap-1">
        {forecast7.days.map((day, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 + i * 0.05 }}
            className="flex flex-col items-center gap-1.5 py-2 px-1 rounded-xl transition-all duration-200 cursor-default"
            style={{ '--hover-bg': 'var(--glass-hover)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.07)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <p className="font-body text-xs" style={{ color: 'var(--text-muted)' }}>
              {new Date(day.date).toLocaleDateString('en', { weekday: 'short' })}
            </p>
            <span className="text-2xl">{day.weather_emoji}</span>
            <p className="font-display text-white font-semibold" style={{ fontSize: '1rem' }}>
              {Math.round(day.temp_max)}°
            </p>
            <p className="font-body text-xs" style={{ color: 'var(--text-muted)' }}>
              {Math.round(day.temp_min)}°
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
