/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Georgia', 'serif'],
        body: ['"DM Sans"', 'sans-serif'],
        urdu: ['"Noto Nastaliq Urdu"', 'serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        navy: '#060d1f',
        storm: '#0f1f3d',
        gold: '#e8b84b',
        'rain-blue': '#4a8fff',
        sakura: '#f0a5bb',
        pearl: '#c8deff',
      },
      animation: {
        'rain': 'rain-fall linear infinite',
        'bob': 'bob 3.5s ease-in-out infinite',
        'shiver': 'shiver 0.15s ease-in-out infinite',
        'explode': 'explode 0.5s ease-out forwards',
        'shimmer-ring': 'shimmer-ring 1.8s ease-in-out infinite',
        'scan': 'scan 1.2s ease-out forwards',
        'pulse-led': 'pulse-led 0.8s ease-in-out infinite',
        'mesh-drift': 'mesh-drift 18s ease-in-out infinite',
        'mesh-drift-2': 'mesh-drift-2 24s ease-in-out infinite',
        'mesh-drift-3': 'mesh-drift-3 20s ease-in-out infinite',
        'float-in': 'float-in 0.6s ease-out forwards',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
      },
      keyframes: {
        'rain-fall': {
          '0%': { transform: 'translateY(-40px) translateX(0px)', opacity: '0' },
          '8%': { opacity: '1' },
          '92%': { opacity: '0.6' },
          '100%': { transform: 'translateY(110vh) translateX(-30px)', opacity: '0' },
        },
        bob: {
          '0%,100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-14px)' },
        },
        shiver: {
          '0%,100%': { transform: 'translateX(0px) rotate(0deg)' },
          '25%': { transform: 'translateX(-4px) rotate(-1deg)' },
          '75%': { transform: 'translateX(4px) rotate(1deg)' },
        },
        explode: {
          '0%': { transform: 'scale(1.3)', opacity: '1' },
          '40%': { transform: 'scale(2.5)', opacity: '0.6' },
          '100%': { transform: 'scale(0.8)', opacity: '1' },
        },
        'shimmer-ring': {
          '0%,100%': { opacity: '0.3', transform: 'scale(1)' },
          '50%': { opacity: '0.9', transform: 'scale(1.04)' },
        },
        scan: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'pulse-led': {
          '0%,100%': { transform: 'scaleY(1)', opacity: '1' },
          '50%': { transform: 'scaleY(0.15)', opacity: '0.7' },
        },
        'mesh-drift': {
          '0%,100%': { transform: 'translate(0,0) scale(1)' },
          '25%': { transform: 'translate(80px,-60px) scale(1.1)' },
          '50%': { transform: 'translate(-60px,80px) scale(0.9)' },
          '75%': { transform: 'translate(60px,60px) scale(1.05)' },
        },
        'mesh-drift-2': {
          '0%,100%': { transform: 'translate(0,0) scale(1)' },
          '33%': { transform: 'translate(-100px,50px) scale(1.1)' },
          '66%': { transform: 'translate(80px,-80px) scale(0.9)' },
        },
        'mesh-drift-3': {
          '0%,100%': { transform: 'translate(0,0) scale(1)' },
          '40%': { transform: 'translate(60px,80px) scale(1.08)' },
          '80%': { transform: 'translate(-80px,-40px) scale(0.95)' },
        },
        'float-in': {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'glow-pulse': {
          '0%,100%': { opacity: '0.4' },
          '50%': { opacity: '0.85' },
        },
      },
    },
  },
  plugins: [],
}
