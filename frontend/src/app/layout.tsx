import '@/styles/globals.css';
import React from 'react';
import { Metadata } from 'next';
import { Inter } from 'next/font/google';

import SiteHeader from '@/components/layout/site-header';
import SiteFooter from '@/components/layout/site-footer';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'AlignAI - AI Alignment and Ethical Guardrails',
  description: 'Ensuring responsible, transparent, and safe interactions with AI',
  viewport: 'width=device-width, initial-scale=1',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="flex min-h-screen flex-col">
          <SiteHeader />
          <main className="flex-1">{children}</main>
          <SiteFooter />
        </div>
      </body>
    </html>
  );
}
