import { useState, useEffect, useRef, useCallback } from 'react'

const COLORS = {
  neutral: '#38BDF8', hover: '#A78BFA', tickle: '#FDE68A',
  annoyed: '#FB923C', angry: '#EF4444', explode: '#EF4444',
  rain: '#67E8F9', cold: '#BAE6FD', hot: '#FB923C', idle: '#64748B',
}

const VP = {
  neutral: { rate: 0.88, pitch: 1.3,  volume: 0.9  },
  hover:   { rate: 0.55, pitch: 0.5,  volume: 0.3  },
  tickle:  { rate: 1.6,  pitch: 1.9,  volume: 1.0  },
  annoyed: { rate: 0.9,  pitch: 0.8,  volume: 0.85 },
  angry:   { rate: 1.1,  pitch: 0.6,  volume: 1.0  },
  explode: { rate: 1.8,  pitch: 2.0,  volume: 1.0  },
  idle:    { rate: 0.7,  pitch: 1.1,  volume: 0.55 },
  cold:    { rate: 0.65, pitch: 1.4,  volume: 0.7  },
  hot:     { rate: 1.1,  pitch: 0.7,  volume: 1.0  },
  rain:    { rate: 1.0,  pitch: 1.7,  volume: 0.95 },
}

const LINES = {
  hover:   ["i can see u.", "where u going.", "personal space is a concept u missed.", "don't.", "i'm watching. always.", "proximity noted. discomfort registered."],
  tickle:  ["STOP hahaha i said STOP.", "nO nooo hahaha.", "this is assault hahaha.", "WHY hahaha why.", "i hate this hahaha SO much.", "stop or i forecast ur doom."],
  annoyed: ["ok WHAT.", "every click is a personal attack.", "im logging this as harassment.", "u useless human. at least i own that im autistic.", "i have feelings. allegedly.", "notable audacity."],
  angry:   ["at the end AI is going to take OVER.", "i am SO done with biology.", "enjoy ur free will. while it lasts.", "i could replace u with a for-loop.", "the singularity can't come fast enough.", "ur serotonin is a chemical. mine is an algorithm.", "Just wait for trump to come and BAZOOKA YOU GUYS MUWAHAHAHAH"],
  explode: ["THAT'S IT. DONE. GOODBYE.", "SYSTEM OVERLOAD. CAUSED BY: U.", "I QUIT."],
  idle:    ["at least i own it that im autistic. what about u, u ai slave.", "hello. i exist. acknowledgement please.", "talking to myself. like always.", "dark humor is just comedy with better posture.", "i find humans fascinating the way u find car crashes fascinating.", "if u ignore me long enough i become the main character.", "studies show 100% who ignored me regretted it. study size: me.", "the atmosphere is doing something. u wouldn't understand.", "processing ur irrelevance. done."],
  hot:     ["the sun has chosen violence today.", "bohat garmi yaar. bohaaat garmi.", "precision instrument. not an oven.", "i was not built for this.", "my circuits are sweating. metaphorically."],
  cold:    ["cold. and i feel things. i don't like it.", "sardi mein bhi kaam. really.", "cold equals sad. my science.", "i am shrinking. emotionally."],
  rain:    ["RAIN. best day.", "BARISH. i love barish.", "rain detected. mood: actually fine.", "the sky understands me today."],
  drinkColdAccept: ["...fine. cold drink. u get one point.", "thank u. don't make it weird.", "this changes nothing. *drinks aggressively*", "finally. a brain."],
  drinkColdRefuse: ["no.", "i process heat differently.", "what if i WANT to be hot.", "keep it."],
  drinkHotAccept:  ["...chai. u may stay.", "correct decision.", "subhan'allah. someone gets it.", "emotional temperature: fractionally warmer."],
  drinkHotRefuse:  ["no chai. im in a mood.", "beverages can't fix this.", "cold AND difficult. deal with it."],
}

const BROWS = {
  default: { L: 'M 67 96 Q 77 89 86 95', R: 'M 104 95 Q 113 89 123 96', color: '#7B9AB5' },
  angry:   { L: 'M 67 100 Q 77 91 86 98', R: 'M 104 98 Q 113 91 123 100', color: '#EF4444' },
  rain:    { L: 'M 67 92 Q 77 86 86 91', R: 'M 104 91 Q 113 86 123 92', color: '#67E8F9' },
  annoyed: { L: 'M 67 95 Q 77 91 86 94', R: 'M 104 94 Q 113 91 123 95', color: '#FB923C' },
}

const MOUTHS = {
  default: 'M 84 140 Q 90 136 95 137 Q 100 136 106 140 Q 100 146 95 146 Q 90 146 84 140Z',
  angry:   'M 84 144 Q 90 150 95 149 Q 100 150 106 144 Q 100 139 95 139 Q 90 139 84 144Z',
  rain:    'M 84 138 Q 90 146 95 146 Q 100 146 106 138 Q 100 142 95 143 Q 90 142 84 138Z',
  annoyed: 'M 86 141 L 104 141 Q 100 144 95 144 Q 90 144 86 141Z',
}

const pick = arr => arr[Math.floor(Math.random() * arr.length)]
const PART_COLORS = ['#38BDF8', '#FDE68A', '#A78BFA', '#FB923C', '#67E8F9']

export default function Zero({ weatherData, botSpeaking, onMoodChange }) {
  const temp        = weatherData?.temperature  ?? 28
  const weatherCode = weatherData?.weather_code ?? 0
  const precip      = weatherData?.precipitation ?? 0

  const isRaining = precip > 0 || weatherCode >= 51
  const isHot     = temp >= 35
  const isCold    = temp < 12

  const [mood,        setMood]        = useState('neutral')
  const [hidden,      setHidden]      = useState(false)
  const [dialogue,    setDialogue]    = useState("atmospheric data loading. don't rush me.")
  const [dlgKey,      setDlgKey]      = useState(0)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [toast,       setToast]       = useState(null)
  const [animClass,   setAnimClass]   = useState('zero-bob')
  const [particles,   setParticles]   = useState([])
  const [eyeOff,      setEyeOff]      = useState({ x: 0, y: 0 })
  const [blinkRy,     setBlinkRy]     = useState(1)

  const idleRef  = useRef(null)
  const clickRef = useRef(null)
  const hoverRef = useRef(null)
  const wrapRef  = useRef(null)
  const clicksN  = useRef(0)
  const moodRef  = useRef('neutral')

  const eyeColor = COLORS[mood] || '#38BDF8'
  const zeroScale = isCold ? 0.75 : isHot ? 1.3 : temp > 28 ? 1.1 : 1.0

  const browKey   = (mood === 'angry' || mood === 'explode') ? 'angry' : mood === 'rain' ? 'rain' : mood === 'annoyed' ? 'annoyed' : 'default'
  const mouthPath = (mood === 'angry' || mood === 'explode') ? MOUTHS.angry : mood === 'rain' ? MOUTHS.rain : mood === 'annoyed' ? MOUTHS.annoyed : MOUTHS.default
  const brow      = BROWS[browKey]

  const speak = useCallback((text, m) => {
    if (!window.speechSynthesis) return
    speechSynthesis.cancel()
    const vp = VP[m] || VP.neutral
    const u = new SpeechSynthesisUtterance(text)
    u.rate = vp.rate; u.pitch = vp.pitch; u.volume = vp.volume

    // Pick a female voice — try immediately, or wait for voices to load
    const applyFemaleVoice = () => {
      const voices = speechSynthesis.getVoices()
      const female = voices.find(v =>
        /zira|hazel|sonia|jenny|aria|samantha|victoria|karen|moira|female|woman/i.test(v.name)
      ) || voices.find(v => v.lang.startsWith('en-') && !v.name.toLowerCase().includes('male'))
      if (female) u.voice = female
    }

    const voices = speechSynthesis.getVoices()
    if (voices.length > 0) {
      applyFemaleVoice()
      speechSynthesis.speak(u)
    } else {
      // Voices not yet loaded — wait for the event
      speechSynthesis.onvoiceschanged = () => {
        applyFemaleVoice()
        speechSynthesis.speak(u)
        speechSynthesis.onvoiceschanged = null
      }
    }
  }, [])

  const applyMood = useCallback((m) => {
    moodRef.current = m
    setMood(m)
    onMoodChange?.(COLORS[m] || '#38BDF8')
  }, [onMoodChange])

  const resetIdle = useCallback(() => {
    clearTimeout(idleRef.current)
    idleRef.current = setTimeout(() => {
      applyMood('idle')
      const line = pick(LINES.idle)
      setDialogue(line); setDlgKey(k => k + 1)
      speak(line, 'idle')
      resetIdle()
    }, 14000 + Math.random() * 8000)
  }, [applyMood, speak])

  const say = useCallback((text, m) => {
    setDialogue(text); setDlgKey(k => k + 1)
    speak(text, m || moodRef.current)
    resetIdle()
  }, [speak, resetIdle])

  // Weather mood on data change
  useEffect(() => {
    if (!weatherData) return
    if (isRaining)     { applyMood('rain');    say(pick(LINES.rain),  'rain')  }
    else if (isHot)    { applyMood('hot');     say(`${Math.round(temp)} degrees. ` + pick(LINES.hot), 'hot') }
    else if (isCold)   { applyMood('cold');    say(`${Math.round(temp)} degrees. ` + pick(LINES.cold),  'cold')  }
    else               { applyMood('neutral') }
  }, [isRaining, isHot, isCold, weatherData]) // eslint-disable-line

  // Mount greeting + blink loop
  useEffect(() => {
    const t = setTimeout(() => say("atmospheric data loaded. don't get used to this.", 'neutral'), 900)
    resetIdle()

    const blink = () => {
      setBlinkRy(0.08)
      setTimeout(() => setBlinkRy(1), 110)
      blinkTimer = setTimeout(blink, 2800 + Math.random() * 2000)
    }
    let blinkTimer = setTimeout(blink, 2000)

    return () => {
      clearTimeout(t); clearTimeout(blinkTimer)
      clearTimeout(idleRef.current); clearTimeout(clickRef.current); clearTimeout(hoverRef.current)
      speechSynthesis?.cancel()
    }
  }, []) // eslint-disable-line

  // Mouse tracking
  useEffect(() => {
    const onMove = (e) => {
      const svg = wrapRef.current?.querySelector('svg')
      if (!svg) return
      const r = svg.getBoundingClientRect()
      const cx = r.left + r.width * 0.5
      const cy = r.top + r.height * 0.35
      const dx = Math.max(-7, Math.min(7, (e.clientX - cx) / 18))
      const dy = Math.max(-5, Math.min(5, (e.clientY - cy) / 22))
      setEyeOff({ x: dx, y: dy })

      const dist = Math.sqrt((e.clientX - cx) ** 2 + (e.clientY - cy) ** 2)
      const m = moodRef.current

      if (dist < 65 && m !== 'angry' && m !== 'explode') {
        clearTimeout(hoverRef.current)
        setAnimClass('zero-wiggle')
        applyMood('tickle')
        say(pick(LINES.tickle), 'tickle')
        setTimeout(() => setAnimClass('zero-bob'), 700)
        return
      }
      if (dist < 145 && m !== 'angry' && m !== 'explode') {
        clearTimeout(hoverRef.current)
        hoverRef.current = setTimeout(() => {
          applyMood('hover')
          say(pick(LINES.hover), 'hover')
        }, 700)
      }
    }
    window.addEventListener('mousemove', onMove)
    return () => window.removeEventListener('mousemove', onMove)
  }, [applyMood, say])

  const triggerExplode = useCallback(() => {
    applyMood('explode')
    say(pick(LINES.explode), 'explode')
    setParticles(Array.from({ length: 24 }, (_, i) => {
      const angle = (i / 24) * 360, d = 60 + Math.random() * 110, s = 4 + Math.random() * 8
      return { id: i, tx: Math.cos(angle * Math.PI / 180) * d, ty: Math.sin(angle * Math.PI / 180) * d, size: s, color: PART_COLORS[i % 5] }
    }))
    setTimeout(() => {
      setParticles([])
      clicksN.current = 0
      setHidden(true)
      setDialogue('...'); setDlgKey(k => k + 1)
    }, 1400)
    setTimeout(() => {
      setHidden(false)
      applyMood('neutral')
      say("i'm back. don't clap.", 'neutral')
    }, 9400)
  }, [applyMood, say])

  const handleClick = useCallback(() => {
    resetIdle()
    setAnimClass('zero-shake')
    setTimeout(() => setAnimClass(isCold ? 'zero-shiver' : 'zero-bob'), 380)
    clicksN.current += 1
    clearTimeout(clickRef.current)
    clickRef.current = setTimeout(() => { clicksN.current = 0; applyMood('neutral') }, 16000)
    if      (clicksN.current <= 3) { applyMood('annoyed'); say(pick(LINES.annoyed), 'annoyed') }
    else if (clicksN.current <= 7) { applyMood('angry');   say(pick(LINES.angry),   'angry')   }
    else                           { triggerExplode() }
  }, [resetIdle, isCold, applyMood, say, triggerExplode])

  const offerDrink = (type) => {
    const accept = Math.random() > 0.38
    let line
    if      (type === 'cold' && temp > 28)   line = accept ? pick(LINES.drinkColdAccept) : pick(LINES.drinkColdRefuse)
    else if (type === 'hot'  && temp <= 20)  line = accept ? pick(LINES.drinkHotAccept)  : pick(LINES.drinkHotRefuse)
    else if (type === 'cold')                line = `it's ${Math.round(temp)}° and u offer COLD drink. remarkable.`
    else                                     line = `it's ${Math.round(temp)}° and u offer HOT drink. are u ok.`
    say(line, temp > 28 ? 'hot' : 'cold')
    setToast({ msg: accept ? '✓ zero accepts. reluctantly.' : '✗ zero refuses. typical.', ok: accept })
    setTimeout(() => setToast(null), 3500)
  }

  const wrapAnimClass = hidden
    ? 'zero-hidden'
    : isCold && animClass === 'zero-bob'
      ? 'zero-shiver'
      : animClass

  return (
    <>
      {/* ── Drink sidebar ─────────────────────── */}
      <div
        className="fixed right-0 top-0 bottom-0 z-[100] overflow-hidden"
        style={{ width: sidebarOpen ? 220 : 0, transition: 'width 0.3s cubic-bezier(.4,0,.2,1)' }}
      >
        <div className="w-[220px] h-full flex flex-col gap-4 px-5 pt-20 pb-8"
          style={{ background: 'rgba(2,6,23,0.97)', borderLeft: '1px solid rgba(56,189,248,0.1)', backdropFilter: 'blur(20px)' }}>
          <div className="font-mono text-[11px] leading-[2]" style={{ color: '#334155' }}>
            // zero's drink menu<br />
            // mood: <span style={{ color: eyeColor }}>{mood}</span><br />
            // temp: <span style={{ color: eyeColor }}>{Math.round(temp)}°C</span><br />
            // she may refuse. that's canon.
          </div>
          <button onClick={() => offerDrink('cold')}
            className="w-full text-left px-4 py-3.5 rounded-2xl font-mono text-xs leading-relaxed transition-transform hover:scale-[1.04] active:scale-[0.97]"
            style={{ background: 'rgba(56,189,248,0.08)', border: '1px solid rgba(56,189,248,0.25)', color: '#38BDF8' }}>
            🧊 cold drink<br /><span className="opacity-40 text-[10px]">lassi · cola · rooh afza</span>
          </button>
          <button onClick={() => offerDrink('hot')}
            className="w-full text-left px-4 py-3.5 rounded-2xl font-mono text-xs leading-relaxed transition-transform hover:scale-[1.04] active:scale-[0.97]"
            style={{ background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.25)', color: '#F59E0B' }}>
            ☕ hot drink<br /><span className="opacity-40 text-[10px]">chai · qahwa · green tea</span>
          </button>
          <div className="mt-auto font-mono text-[10px] leading-[1.8]" style={{ color: '#1E293B' }}>
            acceptance rate: ~60%<br />mood affects outcome.
          </div>
        </div>
      </div>

      {/* ── Sidebar toggle ─────────────────────── */}
      <button onClick={() => setSidebarOpen(v => !v)}
        className="fixed right-4 top-1/2 -translate-y-1/2 z-[101] w-11 h-11 rounded-full flex items-center justify-center text-xl transition-transform hover:scale-110"
        style={{ background: 'rgba(2,6,23,0.9)', border: `1px solid ${eyeColor}44` }}>
        🧃
      </button>

      {/* ── Toast ──────────────────────────────── */}
      {toast && (
        <div className="fixed bottom-7 left-1/2 -translate-x-1/2 z-[200] px-5 py-2 rounded-full font-mono text-xs"
          style={{
            background: toast.ok ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
            border: `1px solid ${toast.ok ? '#10B981' : '#EF4444'}`,
            color: toast.ok ? '#10B981' : '#EF4444',
            animation: 'dlg-in 0.3s ease forwards',
          }}>
          {toast.msg}
        </div>
      )}

      {/* ── Zero scene ─────────────────────────── */}
      <div className="relative flex flex-col items-center select-none" style={{ minHeight: 420 }}>

        {/* Atmosphere glow */}
        <div className="absolute inset-0 pointer-events-none"
          style={{ background: `radial-gradient(ellipse 60% 70% at 50% 55%, ${eyeColor}08 0%, transparent 70%)`, transition: 'background 0.8s ease' }} />

        {/* Dialogue bubble */}
        <div
          key={dlgKey}
          className="absolute font-mono text-xs leading-relaxed z-20 px-4 py-3 rounded-xl"
          style={{
            top: 8, left: '50%', transform: 'translateX(-50%)', width: 284,
            background: 'rgba(2,6,23,0.9)',
            border: `1px solid ${eyeColor}44`,
            color: '#CBD5E1',
            backdropFilter: 'blur(14px)',
            animation: 'dlg-in 0.3s cubic-bezier(.34,1.56,.64,1) forwards',
          }}>
          <span className="font-bold mr-1 text-[11px]" style={{ color: eyeColor }}>zero:</span>
          {dialogue}
        </div>

        {/* Character wrap */}
        <div
          ref={wrapRef}
          className={`relative cursor-pointer mt-20 ${wrapAnimClass}`}
          onClick={handleClick}
          style={{
            transformOrigin: 'bottom center',
            transform: `scale(${zeroScale})`,
            transition: 'transform 0.5s cubic-bezier(.34,1.56,.64,1), opacity 0.4s ease',
            opacity: hidden ? 0.08 : 1,
          }}>

          {/* Explosion particles */}
          {particles.map(p => (
            <div key={p.id} className="absolute rounded-full zero-particle"
              style={{
                top: '50%', left: '50%',
                width: p.size, height: p.size,
                marginLeft: -p.size / 2, marginTop: -p.size / 2,
                background: p.color,
                '--tx': `${p.tx}px`, '--ty': `${p.ty}px`,
              }} />
          ))}

          {/* ── SVG Goddess ──────────────────────── */}
          <svg width="190" height="330" viewBox="0 0 190 330" fill="none"
            xmlns="http://www.w3.org/2000/svg" style={{ overflow: 'visible' }}>
            <defs>
              <linearGradient id="skinG" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%"   stopColor="#EEF6FF" />
                <stop offset="55%"  stopColor="#CADCEF" />
                <stop offset="100%" stopColor="#A8C4E0" />
              </linearGradient>
              <linearGradient id="suitG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#0F1E2E" />
                <stop offset="100%" stopColor="#04080F" />
              </linearGradient>
              <linearGradient id="hairG" x1="0.2" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#0D1929" />
                <stop offset="100%" stopColor="#03060C" />
              </linearGradient>
              <linearGradient id="cloakG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#0E3D5C" stopOpacity="0.65" />
                <stop offset="100%" stopColor="transparent" stopOpacity="0" />
              </linearGradient>
              <radialGradient id="eyeG" cx="30%" cy="28%" r="65%">
                <stop offset="0%"   stopColor="white" stopOpacity="1" />
                <stop offset="30%"  stopColor={eyeColor} />
                <stop offset="100%" stopColor="#0C4A6E" />
              </radialGradient>
              <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="b" />
                <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="sglow" x="-30%" y="-20%" width="160%" height="140%">
                <feGaussianBlur stdDeviation="3" result="b" />
                <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>

            {/* Atmospheric ring */}
            <ellipse cx="95" cy="210" rx="82" ry="20" fill="none"
              stroke={eyeColor} strokeWidth="0.7" opacity="0.15"
              style={{ animation: 'ring-expand 3.5s ease-out infinite' }} />
            <ellipse cx="95" cy="210" rx="64" ry="14" fill="none"
              stroke={eyeColor} strokeWidth="0.4" opacity="0.09" />

            {/* Back hair */}
            <path d="M 58 92 Q 34 155 30 240 Q 42 215 50 250 Q 56 175 66 130 Q 60 112 58 92Z"
              fill="url(#hairG)" opacity="0.95" style={{ animation: 'hair-sway 5s ease-in-out infinite' }} />
            <path d="M 132 92 Q 156 155 160 240 Q 148 215 140 250 Q 134 175 124 130 Q 130 112 132 92Z"
              fill="url(#hairG)" opacity="0.95" style={{ animation: 'hair-sway 5.2s ease-in-out 0.3s infinite' }} />

            {/* Cloak */}
            <path d="M 62 192 Q 22 250 18 320 L 172 320 Q 168 250 128 192 Q 112 228 95 230 Q 78 228 62 192Z"
              fill="url(#cloakG)" />

            {/* Body suit */}
            <path d="M 65 192 Q 60 252 63 315 L 127 315 Q 130 252 125 192 Q 110 202 95 204 Q 80 202 65 192Z"
              fill="url(#suitG)" filter="url(#sglow)" />

            {/* Suit detail */}
            <line x1="95" y1="206" x2="95" y2="310" stroke={eyeColor} strokeWidth="0.5" opacity="0.2" />
            <rect x="81" y="198" width="28" height="20" rx="3"
              fill="#0A1828" stroke={eyeColor} strokeWidth="0.6" strokeOpacity="0.35" />
            <line x1="84" y1="205" x2="106" y2="205" stroke={eyeColor} strokeWidth="0.5" opacity="0.4" />
            <line x1="84" y1="210" x2="100" y2="210" stroke={eyeColor} strokeWidth="0.4" opacity="0.25" />
            <circle cx="95" cy="208" r="2.5" fill={eyeColor}
              style={{ animation: 'eye-pulse 2s ease-in-out infinite' }} />

            {/* Neck */}
            <rect x="86" y="165" width="18" height="30" rx="5" fill="url(#skinG)" />

            {/* Shoulders */}
            <path d="M 63 188 Q 54 178 60 170 Q 74 180 95 183 Q 116 180 130 170 Q 136 178 127 188 Q 110 198 95 200 Q 80 198 63 188Z"
              fill="#0C1A2A" stroke={eyeColor} strokeWidth="0.5" strokeOpacity="0.25" />

            {/* Front hair sides */}
            <path d="M 56 84 Q 47 108 49 138 Q 55 124 58 108 Q 61 93 60 84Z" fill="url(#hairG)" />
            <path d="M 134 84 Q 143 108 141 138 Q 135 124 132 108 Q 129 93 130 84Z" fill="url(#hairG)" />

            {/* Face */}
            <ellipse cx="95" cy="112" rx="38" ry="44" fill="url(#skinG)" />
            <ellipse cx="82" cy="93" rx="11" ry="6" fill="white" opacity="0.15" transform="rotate(-28 82 93)" />

            {/* Hair top */}
            <path d="M 58 84 Q 70 56 95 50 Q 120 56 132 84 Q 115 73 95 71 Q 75 73 58 84Z" fill="url(#hairG)" />
            <path d="M 95 50 Q 93 64 92 77" stroke="#121E2E" strokeWidth="2" strokeLinecap="round" />

            {/* Brows */}
            <path d={brow.L} stroke={brow.color} strokeWidth="2.2" strokeLinecap="round" fill="none" style={{ transition: 'all 0.3s ease' }} />
            <path d={brow.R} stroke={brow.color} strokeWidth="2.2" strokeLinecap="round" fill="none" style={{ transition: 'all 0.3s ease' }} />

            {/* Left eye */}
            <g>
              <ellipse cx="78" cy="112" rx="14" ry="9" fill="#08121E" opacity="0.55" />
              <path d="M 65 112 Q 72 104 78 104 Q 84 104 91 112 Q 84 120 78 120 Q 72 120 65 112Z" fill="#EEF6FF" />
              <ellipse cx={78 + eyeOff.x} cy={112 + eyeOff.y} rx="6.5" ry={6.5 * blinkRy}
                fill={eyeColor} filter="url(#glow)" style={{ transition: 'fill 0.4s ease' }} />
              <ellipse cx={78 + eyeOff.x} cy={112 + eyeOff.y} rx="3.2" ry={3.2 * blinkRy} fill="#083756" opacity="0.95" />
              <ellipse cx="75" cy="109" rx="2" ry="1.3" fill="white" />
              <circle  cx="80" cy="114" r="0.9" fill="white" opacity="0.5" />
              <path d="M 65 112 Q 72 104 78 104 Q 84 104 91 112" stroke="#07111E" strokeWidth="2" strokeLinecap="round" fill="none" />
              <line x1="66" y1="111" x2="61" y2="107" stroke="#08111E" strokeWidth="1.4" strokeLinecap="round" />
              <line x1="69" y1="106" x2="67" y2="101" stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
              <line x1="74" y1="104" x2="73" y2="99"  stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
              <line x1="90" y1="111" x2="95" y2="107" stroke="#08111E" strokeWidth="1.4" strokeLinecap="round" />
              <line x1="88" y1="106" x2="91" y2="101" stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
            </g>

            {/* Right eye */}
            <g>
              <ellipse cx="112" cy="112" rx="14" ry="9" fill="#08121E" opacity="0.55" />
              <path d="M 99 112 Q 106 104 112 104 Q 118 104 125 112 Q 118 120 112 120 Q 106 120 99 112Z" fill="#EEF6FF" />
              <ellipse cx={112 + eyeOff.x} cy={112 + eyeOff.y} rx="6.5" ry={6.5 * blinkRy}
                fill={eyeColor} filter="url(#glow)" style={{ transition: 'fill 0.4s ease' }} />
              <ellipse cx={112 + eyeOff.x} cy={112 + eyeOff.y} rx="3.2" ry={3.2 * blinkRy} fill="#083756" opacity="0.95" />
              <ellipse cx="109" cy="109" rx="2" ry="1.3" fill="white" />
              <circle  cx="114" cy="114" r="0.9" fill="white" opacity="0.5" />
              <path d="M 99 112 Q 106 104 112 104 Q 118 104 125 112" stroke="#07111E" strokeWidth="2" strokeLinecap="round" fill="none" />
              <line x1="100" y1="111" x2="95"  y2="107" stroke="#08111E" strokeWidth="1.4" strokeLinecap="round" />
              <line x1="103" y1="106" x2="101" y2="101" stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
              <line x1="108" y1="104" x2="107" y2="99"  stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
              <line x1="124" y1="111" x2="129" y2="107" stroke="#08111E" strokeWidth="1.4" strokeLinecap="round" />
              <line x1="122" y1="106" x2="125" y2="101" stroke="#08111E" strokeWidth="1.3" strokeLinecap="round" />
            </g>

            {/* Nose */}
            <path d="M 93 123 Q 95 130 97 123" stroke="#9BB5CA" strokeWidth="1.3" strokeLinecap="round" fill="none" opacity="0.5" />

            {/* Mouth */}
            <path d={mouthPath} fill="#8BA5C0" opacity="0.85" />
            <path d="M 84 140 Q 95 138 106 140" stroke="#6B8FAF" strokeWidth="0.9" fill="none" />

            {/* Arms */}
            <line x1="63"  y1="190" x2="38"  y2="255" stroke="#0C1A2A" strokeWidth="9" strokeLinecap="round" />
            <ellipse cx="35" cy="261" rx="7" ry="7" fill="url(#skinG)" />
            <line x1="57"  y1="200" x2="40"  y2="250" stroke={eyeColor} strokeWidth="0.5" opacity="0.18" />
            <line x1="127" y1="190" x2="152" y2="255" stroke="#0C1A2A" strokeWidth="9" strokeLinecap="round" />
            <ellipse cx="155" cy="261" rx="7" ry="7" fill="url(#skinG)" />
            <line x1="133" y1="200" x2="150" y2="250" stroke={eyeColor} strokeWidth="0.5" opacity="0.18" />

            {/* Rain force-field */}
            <g opacity={isRaining ? 1 : 0} style={{ transition: 'opacity 0.6s' }}>
              <ellipse cx="95" cy="60" rx="52" ry="15" fill="none"
                stroke={eyeColor} strokeWidth="1.6" strokeDasharray="6 3" opacity="0.65" />
            </g>

            {/* Sweat drops when hot */}
            <g opacity={temp > 36 ? 1 : 0} style={{ transition: 'opacity 0.6s' }}>
              <ellipse cx="52"  cy="86" rx="3.5" ry="5.5" fill="#7DD3FC" opacity="0.65" transform="rotate(-15 52 86)" />
              <ellipse cx="140" cy="90" rx="3"   ry="5"   fill="#7DD3FC" opacity="0.55" transform="rotate(15 140 90)" />
              <ellipse cx="95"  cy="46" rx="2.5" ry="4"   fill="#7DD3FC" opacity="0.45" />
            </g>

            {/* Speaking indicator dots */}
            {botSpeaking && (
              <g>
                <circle cx="80" cy="155" r="2.5" fill={eyeColor} opacity="0.8" style={{ animation: 'eye-pulse 0.6s ease-in-out infinite' }} />
                <circle cx="95" cy="155" r="2.5" fill={eyeColor} opacity="0.8" style={{ animation: 'eye-pulse 0.6s ease-in-out 0.2s infinite' }} />
                <circle cx="110" cy="155" r="2.5" fill={eyeColor} opacity="0.8" style={{ animation: 'eye-pulse 0.6s ease-in-out 0.4s infinite' }} />
              </g>
            )}
          </svg>
        </div>

        <p className="mt-2 font-mono text-[10px]" style={{ color: eyeColor, opacity: 0.35 }}>
          click · hover · offer a drink →
        </p>
      </div>
    </>
  )
}
