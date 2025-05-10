import { Metadata } from 'next';
import PromptAnalyzer from '@/components/analyzer/prompt-analyzer-fixed';

export const metadata: Metadata = {
  title: 'Prompt Analyzer - AlignAI',
  description: 'Analyze prompts for AI compliance risks in real-time',
};

export default function AnalyzerPage() {
  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Prompt Analyzer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Analyze your AI prompts for compliance risks in real-time
        </p>
      </div>
      <PromptAnalyzer />
    </div>
  );
}
