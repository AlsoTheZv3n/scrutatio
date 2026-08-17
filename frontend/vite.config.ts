import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Tailwind v4 has no config file and no PostCSS step — the plugin below plus a
// single `@import "tailwindcss"` in index.css is the whole setup. Without it the
// utility classes are inert strings, which is what they were: tailwindcss was in
// package.json but nothing ever imported it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
})
