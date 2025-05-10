import { Metadata } from 'next';
import PolicyLibrary from '@/components/policies/policy-library';

export const metadata: Metadata = {
  title: 'Policy Library - AlignAI',
  description: 'Browse and search through AI regulation policies',
};

export default function PoliciesPage() {
  return (
    <div className="container py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold">Policy Library</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Browse and search through the EU AI Act and related regulations
        </p>
      </div>
      <PolicyLibrary />
    </div>
  );
}
