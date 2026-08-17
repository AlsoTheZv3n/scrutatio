// src/routes/routes.tsx
import type { ComponentType } from 'react'

const modules = import.meta.glob<{ default: ComponentType }>(
  '../pages/**/*.tsx',
  { eager: true },
)

/** Human labels for the nav. Falls back to the file name. */
const LABELS: Record<string, string> = {
  Home: 'Match',
  Corpus: 'Corpus',
  About: 'Limits',
}

/** Nav order. Anything unlisted sorts last, alphabetically. */
const ORDER = ['Home', 'Corpus', 'About']

export const routes = Object.entries(modules)
  .map(([file, module]) => {
    const name = file
      .replace('../pages/', '')
      .replace(/\.tsx$/, '')
      .replace(/Page$/, '')
      // "Home/Home" -> "Home". Without this the folder convention already in use
      // here produced `/home/home`, and the `name === 'Home'` check below could
      // never fire — so the landing page had no route at `/`.
      .replace(/^(.+)\/\1$/, '$1')

    return {
      key: name,
      path: name === 'Home' ? '/' : '/' + name.toLowerCase(),
      label: LABELS[name] ?? name.split('/').pop()!,
      element: <module.default />,
    }
  })
  .sort((a, b) => {
    const ai = ORDER.indexOf(a.key)
    const bi = ORDER.indexOf(b.key)
    return (ai < 0 ? ORDER.length : ai) - (bi < 0 ? ORDER.length : bi) ||
      a.key.localeCompare(b.key)
  })
