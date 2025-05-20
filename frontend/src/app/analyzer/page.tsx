import { Metadata } from 'next';
import dynamic from 'next/dynamic';

// Import the analyzer component with SSR disabled
const SimplePromptAnalyzer = dynamic(
  () => import('@/components/analyzer/simple-prompt-analyzer'),
  { ssr: false }
);

export const metadata: Metadata = {
  title: 'Text Analyzer - AlignAI',
  description: 'Analyze text for bias, PII, and policy violations',
};

export default function AnalyzerPage() {
  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Text Analyzer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Analyze text for bias, personally identifiable information (PII), and policy violations
        </p>
      </div>
      <div className="max-w-4xl mx-auto">
        <SimplePromptAnalyzer />
      </div>
    </div>
  );
}
