import { useState, useEffect, useRef } from 'react'

const BOT_STATE = {
  IDLE: 'IDLE',
  SCANNING: 'SCANNING',
  REACTING_RAIN: 'REACTING_RAIN',
  REACTING_WIND: 'REACTING_WIND',
  REACTING_HOT: 'REACTING_HOT',
  REACTING_COLD: 'REACTING_COLD',
  SPEAKING: 'SPEAKING',
  RADAR_MODE: 'RADAR_MODE',
  TIMELINE_DRAG: 'TIMELINE_DRAG',
  SLEEPING: 'SLEEPING',
  EXPLODING: 'EXPLODING',
}

export function useZero(temperature = 25) {
  const [state, setState] = useState(BOT_STATE.SCANNING)
  const [eyeColor, setEyeColor] = useState('#60a5fa')
  const [exploding, setExploding] = useState(false)
  const idleTimer = useRef(null)

  const scale = temperature < 20 ? 0.7 : temperature <= 30 ? 1.0 : temperature <= 38 ? 1.3 : 1.3

  const getEyeColor = (st, temp) => {
    if (st === BOT_STATE.REACTING_RAIN) return '#93c5fd'
    if (st === BOT_STATE.REACTING_HOT || temp > 35) return '#f97316'
    if (st === BOT_STATE.REACTING_COLD || temp < 10) return '#818cf8'
    if (st === BOT_STATE.SPEAKING) return '#34d399'
    if (st === BOT_STATE.RADAR_MODE) return '#06b6d4'
    if (st === BOT_STATE.SLEEPING) return '#6b7280'
    return '#60a5fa'
  }

  const triggerExplode = () => {
    setExploding(true)
    setState(BOT_STATE.EXPLODING)
    setTimeout(() => {
      setExploding(false)
      setState(BOT_STATE.IDLE)
    }, 1500)
  }

  const resetIdleTimer = () => {
    clearTimeout(idleTimer.current)
    idleTimer.current = setTimeout(() => setState(BOT_STATE.SLEEPING), 180000)
  }

  const setZeroState = (newState, temp) => {
    if (newState === BOT_STATE.REACTING_HOT && (temp || temperature) > 38) {
      triggerExplode()
      return
    }
    setState(newState)
    setEyeColor(getEyeColor(newState, temp || temperature))
    resetIdleTimer()
  }

  useEffect(() => {
    setEyeColor(getEyeColor(state, temperature))
  }, [state, temperature])

  useEffect(() => {
    const timer = setTimeout(() => setState(BOT_STATE.IDLE), 1200)
    resetIdleTimer()
    return () => { clearTimeout(timer); clearTimeout(idleTimer.current) }
  }, [])

  return { state, setState: setZeroState, eyeColor, scale, exploding, BOT_STATE }
}
