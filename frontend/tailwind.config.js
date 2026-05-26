/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#0d1117',
        panel:   '#161b22',
        border:  '#30363d',
        accent:  '#00d4aa',
        danger:  '#f85149',
        warn:    '#e3b341',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
