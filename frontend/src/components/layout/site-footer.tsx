'use client';

import Link from 'next/link';

export default function SiteFooter() {
  return (
    <footer className="border-t bg-white py-8 dark:bg-gray-950">
      <div className="container grid gap-8 md:grid-cols-2 lg:grid-cols-4">
        <div>
          <h3 className="mb-4 text-lg font-semibold">AlignAI</h3>
          <p className="mb-4 text-sm text-gray-600 dark:text-gray-400">
            AI alignment and ethical guardrails ensuring responsible, transparent, and safe
            interactions with AI.
          </p>
        </div>
        <div>
          <h3 className="mb-4 text-lg font-semibold">Features</h3>
          <ul className="space-y-2 text-sm">
            <li>
              <Link href="/analyzer" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Prompt Analyzer
              </Link>
            </li>
            <li>
              <Link href="/dashboard" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Compliance Dashboard
              </Link>
            </li>
            <li>
              <Link href="/policies" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Policy Library
              </Link>
            </li>
            <li>
              <Link href="/reports" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Compliance Reports
              </Link>
            </li>
            <li>
              <Link href="/sandbox" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Public Sandbox
              </Link>
            </li>
          </ul>
        </div>
        <div>
          <h3 className="mb-4 text-lg font-semibold">Resources</h3>
          <ul className="space-y-2 text-sm">
            <li>
              <Link href="/docs" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Documentation
              </Link>
            </li>
            <li>
              <Link href="/blog" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Blog
              </Link>
            </li>
            <li>
              <a 
                href="https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai"
                className="text-gray-600 hover:text-blue-600 dark:text-gray-400"
                target="_blank"
                rel="noopener noreferrer"
              >
                EU AI Act
              </a>
            </li>
          </ul>
        </div>
        <div>
          <h3 className="mb-4 text-lg font-semibold">Company</h3>
          <ul className="space-y-2 text-sm">
            <li>
              <Link href="/about" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                About
              </Link>
            </li>
            <li>
              <Link href="/contact" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Contact
              </Link>
            </li>
            <li>
              <Link href="/privacy" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Privacy Policy
              </Link>
            </li>
            <li>
              <Link href="/terms" className="text-gray-600 hover:text-blue-600 dark:text-gray-400">
                Terms of Service
              </Link>
            </li>
          </ul>
        </div>
      </div>
      <div className="container mt-8 border-t pt-4">
        <p className="text-center text-sm text-gray-600 dark:text-gray-400">
          © {new Date().getFullYear()} AlignAI. All rights reserved.
        </p>
      </div>
    </footer>
  );
}
