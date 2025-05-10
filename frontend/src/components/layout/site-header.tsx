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
              href="/analyzer"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Analyzer
            </Link>
            <Link
              href="/dashboard"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Dashboard
            </Link>
            <Link
              href="/policies"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Policy Library
            </Link>
            <Link
              href="/reports"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Reports
            </Link>
            <Link
              href="/sandbox"
              className="text-sm font-medium transition-colors hover:text-blue-600"
            >
              Sandbox
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <MobileNav />
          <div className="hidden md:flex">
            <Link
              href="/login"
              className="mr-2 rounded-md px-3 py-2 text-sm hover:bg-gray-100"
            >
              Sign in
            </Link>
            <Link
              href="/signup"
              className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700"
            >
              Sign up
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
}
