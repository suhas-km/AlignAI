import { Metadata } from 'next';
import dynamic from 'next/dynamic';

// Import the analyzer component with SSR disabled
const SimplePromptAnalyzer = dynamic(
  () => import('@/components/analyzer/simple-prompt-analyzer'),
  { ssr: false }
);

export const metadata: Metadata = {
  title: 'AI Content Analyzer - AlignAI',
  description: 'EU AI Act compliance checker for your content',
};

export default function AnalyzerPage() {
  return (
    <div className="container py-8">
      <div className="mb-6">
        <h1 className="mb-2 text-3xl font-bold">AI Content Analyzer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Check for EU AI Act compliance and ethical guidelines
        </p>
      </div>
      <div className="max-w-4xl mx-auto">
        <SimplePromptAnalyzer />
      </div>
    </div>
  );
}
