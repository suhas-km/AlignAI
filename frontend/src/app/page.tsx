import Link from 'next/link';
import { ArrowRightIcon } from '@/components/ui/icons';

export default function HomePage() {
  return (
    <div className="container flex flex-col items-center justify-center space-y-12 py-16 md:py-24">
      <div className="text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-6xl">
          AI Alignment and Ethical{' '}
          <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Guardrails
          </span>
        </h1>
        <p className="mb-8 text-xl text-gray-600 dark:text-gray-400">
          Ensure responsible, transparent, and safe interactions with AI
        </p>
        <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
          <Link
            href="/analyzer"
            className="flex items-center gap-2 rounded-full bg-blue-600 px-6 py-3 text-white hover:bg-blue-700"
          >
            Try the Analyzer <ArrowRightIcon className="h-4 w-4" />
          </Link>
          <Link
            href="/sandbox"
            className="flex items-center gap-2 rounded-full border border-gray-300 bg-transparent px-6 py-3 hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            Public Sandbox
          </Link>
        </div>
      </div>

      <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
        <FeatureCard
          title="Real-time Analysis"
          description="Analyze prompts and outputs in real-time with token-level risk highlighting"
          icon="🔎"
        />
        <FeatureCard
          title="EU AI Act Compliance"
          description="Map content against relevant EU AI Act regulations using semantic matching"
          icon="📜"
        />
        <FeatureCard
          title="Bias Detection"
          description="Identify potentially biased language in prompts and AI-generated content"
          icon="⚠️"
        />
        <FeatureCard
          title="PII Protection"
          description="Flag personally identifiable information to prevent data leakage"
          icon="🔒"
        />
        <FeatureCard
          title="Comprehensive Dashboard"
          description="Visualize compliance trends and risk factors across your organization"
          icon="📊"
        />
        <FeatureCard
          title="Audit-ready Reports"
          description="Generate detailed reports for regulatory compliance"
          icon="📝"
        />
      </div>
    </div>
  );
}

function FeatureCard({
  title,
  description,
  icon,
}: {
  title: string;
  description: string;
  icon: string;
}) {
  return (
    <div className="flex flex-col rounded-lg border border-gray-200 p-6 shadow-sm dark:border-gray-800">
      <div className="mb-4 text-3xl">{icon}</div>
      <h3 className="mb-2 text-xl font-bold">{title}</h3>
      <p className="text-gray-600 dark:text-gray-400">{description}</p>
    </div>
  );
}
