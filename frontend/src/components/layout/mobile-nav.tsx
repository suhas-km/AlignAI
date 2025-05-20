'use client';

import Link from 'next/link';
import { useState } from 'react';

export function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="md:hidden">
      <button
        className="flex h-10 w-10 items-center justify-center"
        onClick={() => setIsOpen(!isOpen)}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          className="h-6 w-6"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
        <span className="sr-only">Toggle Menu</span>
      </button>
      {isOpen && (
        <div 
          className="fixed inset-0 z-50 bg-gray-800/40" 
          onClick={() => setIsOpen(false)}
        >
          <div
            className="fixed inset-y-0 right-0 w-full max-w-sm bg-white p-6 dark:bg-gray-900"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <Link href="/" className="flex items-center space-x-2">
                <span className="text-xl font-bold">AlignAI</span>
              </Link>
              <button 
                className="h-10 w-10" 
                onClick={() => setIsOpen(false)}
                aria-label="Close menu"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  className="h-6 w-6"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
                <span className="sr-only">Close Menu</span>
              </button>
            </div>
            <nav className="mt-6 flex flex-col gap-2">
              <Link
                href="/analyzer"
                className="rounded-md px-4 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => setIsOpen(false)}
              >
                Analyzer
              </Link>
              <Link
                href="/sandbox"
                className="rounded-md px-4 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-800"
                onClick={() => setIsOpen(false)}
              >
                Try the Sandbox
              </Link>
            </nav>
          </div>
        </div>
      )}
    </div>
  );
}
