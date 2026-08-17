// src/App.tsx
import { Route, Routes } from 'react-router'

import BaseLayout from './components/BaseLayout/BaseLayout'
import { routes } from './routes/routes'

export default function App() {
  return (
    <BaseLayout>
      <Routes>
        {routes.map((route) => (
          <Route key={route.path} path={route.path} element={route.element} />
        ))}
        <Route
          path="*"
          element={
            <p className="text-sm">
              No such page. Try <a className="text-accent underline" href="/">Match</a>.
            </p>
          }
        />
      </Routes>
    </BaseLayout>
  )
}
