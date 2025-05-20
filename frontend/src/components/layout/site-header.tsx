'use client';

import Link from 'next/link';
import { MobileNav } from './mobile-nav';

export default function SiteHeader() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white dark:bg-gray-950">
      <div className="container flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <Link href="/" className="flex items-center space-x-2">
            <span className="text-xl font-bold">AlignAI</span>
          </Link>
          <nav className="hidden gap-6 md:flex">
            <Link
              href="/sandbox"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Sandbox
            </Link>
            <Link
              href="#features"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Features
            </Link>
            <Link
              href="#how-it-works"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              How It Works
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <MobileNav />
        </div>
      </div>
    </header>
  );
}
