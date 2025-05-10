import { Metadata } from 'next';
import PublicSandbox from '@/components/sandbox/public-sandbox';

export const metadata: Metadata = {
  title: 'Public Sandbox - AlignAI',
  description: 'Try the AlignAI analysis tools without requiring an account',
};

export default function SandboxPage() {
  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Public Sandbox</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Try the AlignAI analysis tools without requiring an account
        </p>
      </div>
      <PublicSandbox />
    </div>
  );
}
