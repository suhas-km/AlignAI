import { Metadata } from 'next';
import DashboardOverview from '@/components/dashboard/dashboard-overview';

export const metadata: Metadata = {
  title: 'Compliance Dashboard - AlignAI',
  description: 'Visualize AI compliance metrics and insights',
};

export default function DashboardPage() {
  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Compliance Dashboard</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Monitor your AI compliance metrics and insights
        </p>
      </div>
      <DashboardOverview />
    </div>
  );
}
