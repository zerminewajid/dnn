import { useCallback, useRef } from 'react'

export function useVoice() {
  const audioRef     = useRef(null)
  const objectUrlRef = useRef(null)
  const speakingRef  = useRef(false)

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    speakingRef.current = false
  }, [])

  const speak = useCallback(async (text) => {
    if (!text?.trim()) return

    // Stop any in-progress playback before starting a new one.
    stop()

    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        // Non-2xx — nothing to revoke yet, just bail.
        return
      }

      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      objectUrlRef.current = url

      const audio = new Audio(url)
      audioRef.current    = audio
      speakingRef.current = true

      audio.onended = () => {
        URL.revokeObjectURL(url)
        objectUrlRef.current = null
        audioRef.current     = null
        speakingRef.current  = false
      }

      // Swallow Chrome autoplay NotAllowedError — user interaction may not
      // have happened yet; the audio will simply not play rather than throwing.
      audio.play().catch(() => {})

    } catch {
      // fetch failed or blob conversion failed — revoke URL if it was created.
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
        objectUrlRef.current = null
      }
      audioRef.current    = null
      speakingRef.current = false
    }
  }, [stop])

  return { speak, stop, isSpeaking: () => speakingRef.current }
}
