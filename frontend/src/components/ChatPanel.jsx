import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send } from 'lucide-react'

const SUGGESTIONS = [
  'Should I go outside today?',
  'Will it rain in Lahore?',
  'Cricket weather today?',
  'Chai time or too hot?',
  'Kaisa mausam hai aaj?',
]

export default function ChatPanel({ ws, messages, onMessage, isThinking, eyeColor }) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  const sendMessage = () => {
    if (!input.trim() || !ws) return
    ws.send(JSON.stringify({ message: input }))
    onMessage({ role: 'user', text: input })
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  return (
    <div
      className="flex flex-col rounded-[20px] overflow-hidden"
      style={{
        background: 'rgba(255,255,255,0.055)',
        backdropFilter: 'blur(32px)',
        border: '1px solid rgba(255,255,255,0.1)',
        height: '580px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.07)',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-5 py-4 shrink-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.08)' }}
      >
        <div
          className="w-2.5 h-2.5 rounded-full animate-glow-pulse"
          style={{ background: eyeColor, boxShadow: `0 0 8px ${eyeColor}` }}
        />
        <p className="font-body text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          Ask Zero
        </p>
        <span style={{ color: 'var(--gold)', fontSize: '0.75rem' }}>✦</span>
        <div className="ml-auto">
          <span className="font-urdu text-xs" style={{ color: 'var(--text-muted)' }}>زیرو</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3" style={{ scrollbarWidth: 'thin', scrollbarColor: `${eyeColor}33 transparent` }}>
        {messages.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="space-y-4 py-4"
          >
            <p className="font-body text-xs text-center" style={{ color: 'var(--text-muted)' }}>
              yaar, kuch poochh lo mujhse...
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {SUGGESTIONS.map((s, i) => (
                <motion.button
                  key={s}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.35 + i * 0.07 }}
                  onClick={() => setInput(s)}
                  className="font-body text-xs px-3 py-2 rounded-full transition-all duration-200"
                  style={{
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    color: 'var(--text-secondary)',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.background = `rgba(${hexToRgb(eyeColor) || '232,184,75'},0.1)`
                    e.currentTarget.style.borderColor = `${eyeColor || 'var(--gold)'}50`
                    e.currentTarget.style.color = 'var(--text-primary)'
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.background = 'rgba(255,255,255,0.06)'
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'
                    e.currentTarget.style.color = 'var(--text-secondary)'
                  }}
                >
                  {s}
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}

        <AnimatePresence initial={false}>
          {messages.map((msg, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 10, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.25 }}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'zero' ? (
                <div className="max-w-[85%] relative">
                  {/* Gold left border accent */}
                  <div
                    className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full"
                    style={{ background: eyeColor || 'var(--gold)' }}
                  />
                  <div
                    className="ml-3 px-4 py-3 rounded-2xl rounded-tl-sm"
                    style={{
                      background: 'rgba(255,255,255,0.07)',
                      border: '1px solid rgba(255,255,255,0.1)',
                    }}
                  >
                    <p className="font-body text-xs font-medium mb-1.5" style={{ color: eyeColor || 'var(--gold)' }}>
                      Zero ✦
                    </p>
                    <p className="font-body text-sm leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                      {msg.text}
                    </p>
                  </div>
                </div>
              ) : (
                <div
                  className="max-w-[80%] px-4 py-3 rounded-2xl rounded-tr-sm font-body text-sm leading-relaxed"
                  style={{
                    background: 'rgba(74,143,255,0.18)',
                    border: '1px solid rgba(74,143,255,0.25)',
                    color: 'var(--text-primary)',
                  }}
                >
                  {msg.text}
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Thinking indicator */}
        <AnimatePresence>
          {isThinking && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              className="flex justify-start"
            >
              <div className="ml-3 px-4 py-3 rounded-2xl rounded-tl-sm flex items-center gap-1.5"
                style={{ background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.1)' }}>
                {[0, 0.18, 0.36].map((delay, i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 rounded-full"
                    style={{ background: eyeColor || 'var(--gold)' }}
                    animate={{ opacity: [0.25, 1, 0.25], scale: [0.8, 1.1, 0.8] }}
                    transition={{ duration: 0.9, repeat: Infinity, delay }}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div
        className="px-4 pb-4 pt-3 shrink-0"
        style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}
      >
        <div className="flex gap-2 items-center">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="yaar, kaisa mausam hai aaj..."
            className="flex-1 font-body text-sm px-4 py-3 rounded-2xl transition-all duration-200"
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.08)',
              color: 'var(--text-primary)',
            }}
            onFocus={e => e.target.style.borderColor = `${eyeColor || 'var(--gold)'}50`}
            onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
          />
          <motion.button
            onClick={sendMessage}
            whileTap={{ scale: 0.88 }}
            whileHover={{ scale: 1.05 }}
            disabled={!input.trim()}
            className="p-3 rounded-2xl transition-all duration-200 shrink-0"
            style={{
              background: input.trim()
                ? `rgba(${hexToRgb(eyeColor) || '232,184,75'},0.2)`
                : 'rgba(255,255,255,0.05)',
              border: `1px solid ${input.trim() ? (eyeColor || 'var(--gold)') + '40' : 'rgba(255,255,255,0.08)'}`,
              color: input.trim() ? (eyeColor || 'var(--gold)') : 'var(--text-muted)',
            }}
          >
            <Send size={15} />
          </motion.button>
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
