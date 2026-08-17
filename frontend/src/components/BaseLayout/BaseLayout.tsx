import type { ReactNode } from 'react'

import Footer from './Footer'
import Navigation from './Navigation'

export default function BaseLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <Navigation />
      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
        {children}
      </main>
      <Footer />
    </>
  )
}
