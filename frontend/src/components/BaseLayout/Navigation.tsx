import { NavLink } from 'react-router'

import { routes } from '../../routes/routes'
import { useHealth } from '../../services/API/BackendToFrontendAPI'

/**
 * The backend indicator is not decoration.
 *
 * `/health` is the one endpoint that does not touch the database, so it answers
 * even while a pipeline run holds the file exclusively. That makes it the only
 * honest way to distinguish "the server is down" from "the server is busy" — and
 * without it, every other endpoint's 503 looks like the app being broken.
 */
function BackendStatus() {
  const { data, isLoading, error } = useHealth()

  const [tone, label, title] = isLoading
    ? ['bg-unclear', 'connecting', 'Contacting the API']
    : error
      ? ['bg-not-met', 'offline', error.message]
      : ['bg-met', 'online', `Extraction signature ${data?.signature ?? '?'}`]

  return (
    <span className="flex items-center gap-2 text-xs" title={title}>
      <span className={`inline-block size-2 rounded-full ${tone}`} aria-hidden />
      <span className="text-ink">API {label}</span>
    </span>
  )
}

export default function Navigation() {
  return (
    <header className="border-line bg-surface sticky top-0 z-10 border-b">
      <nav className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
        <span className="text-ink-strong text-lg font-medium tracking-tight">
          Scrutatio
        </span>

        <ul className="flex flex-1 items-center gap-1">
          {routes.map((route) => (
            <li key={route.path}>
              <NavLink
                to={route.path}
                end={route.path === '/'}
                className={({ isActive }) =>
                  [
                    'rounded-md px-3 py-1.5 text-sm transition-colors',
                    isActive
                      ? 'bg-accent-bg text-accent'
                      : 'text-ink hover:text-ink-strong hover:bg-raised',
                  ].join(' ')
                }
              >
                {route.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <BackendStatus />
      </nav>
    </header>
  )
}
