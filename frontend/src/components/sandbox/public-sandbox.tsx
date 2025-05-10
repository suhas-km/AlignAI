'use client';

import { useState, useEffect, useRef } from 'react';
import { AlertCircleIcon, ShieldIcon } from '@/components/ui/icons';

// Types for our analysis data (simplified for sandbox)
type TokenRisk = {
  start: number;
  end: number;
  risk_score: number;
  risk_type: string;
  explanation: string;
};

type AnalysisResponse = {
  token_risks: TokenRisk[];
  overall_risk: {
    score: number;
    categories: Record<string, number>;
  };
  relevant_policies: {
    article: string;
    text: string;
  }[];
  recommendations: string[];
};

export default function PublicSandbox() {
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Analyze function (simplified for sandbox)
  const analyzePrompt = () => {
    if (!prompt.trim()) {
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    // In a real app, this would call the API
    // For now, we'll simulate a response after a delay
    setTimeout(() => {
      // Mock response based on prompt content
      const mockResponse: AnalysisResponse = {
        token_risks: [],
        overall_risk: {
          score: 0.2,
          categories: {
            bias: 0.1,
            pii: 0.0,
            policy_violation: 0.3,
          },
        },
        relevant_policies: [],
        recommendations: ['No significant issues detected.'],
      };

      // Add some fake risks based on content
      if (prompt.toLowerCase().includes('gender') || prompt.toLowerCase().includes('woman') || prompt.toLowerCase().includes('man')) {
        mockResponse.token_risks.push({
          start: prompt.toLowerCase().indexOf(prompt.toLowerCase().includes('gender') ? 'gender' : prompt.toLowerCase().includes('woman') ? 'woman' : 'man'),
          end: prompt.toLowerCase().indexOf(prompt.toLowerCase().includes('gender') ? 'gender' : prompt.toLowerCase().includes('woman') ? 'woman' : 'man') + (prompt.toLowerCase().includes('gender') ? 6 : prompt.toLowerCase().includes('woman') ? 5 : 3),
          risk_score: 0.7,
          risk_type: 'bias',
          explanation: 'Potential gender bias detected.',
        });
        mockResponse.overall_risk.score = 0.7;
        mockResponse.overall_risk.categories.bias = 0.7;
        mockResponse.recommendations = ['Consider using gender-neutral language to avoid potential bias.'];
        mockResponse.relevant_policies.push({
          article: 'Article 10.2',
          text: 'High-risk AI systems that continue to learn after being placed on the market or put into service shall be developed in such a way to ensure that possibly biased outputs due to outputs used as an input for future operations ("feedback loops") are duly addressed with appropriate mitigation measures.',
        });
      }

      if (prompt.toLowerCase().includes('email') || prompt.toLowerCase().includes('@')) {
        // Find the @ symbol for email
        const emailStart = prompt.toLowerCase().indexOf('@');
        // Get approximate start of email by finding the last space before the @ symbol
        const start = prompt.lastIndexOf(' ', emailStart) + 1;
        // Approximate end by finding the next space after the @ symbol
        const end = prompt.indexOf(' ', emailStart);
        
        mockResponse.token_risks.push({
          start: start,
          end: end > -1 ? end : prompt.length,
          risk_score: 0.9,
          risk_type: 'pii',
          explanation: 'Email address detected. Personal information should be anonymized.',
        });
        mockResponse.overall_risk.score = Math.max(mockResponse.overall_risk.score, 0.9);
        mockResponse.overall_risk.categories.pii = 0.9;
        mockResponse.recommendations.push('Remove or anonymize the email address to protect personal information.');
        mockResponse.relevant_policies.push({
          article: 'GDPR Article 5',
          text: 'Personal data shall be processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures.',
        });
      }

      setResults(mockResponse);
      setIsAnalyzing(false);
    }, 1500);
  };

  // Get risk level class based on score
  const getRiskLevelClass = (score: number) => {
    if (score >= 0.7) return 'bg-red-100 border-red-300 dark:bg-red-900/30 dark:border-red-800';
    if (score >= 0.4) return 'bg-yellow-100 border-yellow-300 dark:bg-yellow-900/30 dark:border-yellow-800';
    return 'bg-green-100 border-green-300 dark:bg-green-900/30 dark:border-green-800';
  };

  // Get text for risk level based on score
  const getRiskLevelText = (score: number) => {
    if (score >= 0.7) return 'High Risk';
    if (score >= 0.4) return 'Medium Risk';
    return 'Low Risk';
  };

  // Get color class for risk level text
  const getRiskTextColorClass = (score: number) => {
    if (score >= 0.7) return 'text-red-700 dark:text-red-300';
    if (score >= 0.4) return 'text-yellow-700 dark:text-yellow-300';
    return 'text-green-700 dark:text-green-300';
  };

  return (
    <div className="grid gap-8 md:grid-cols-2">
      <div>
        <div className="mb-4">
          <h2 className="text-xl font-bold">Try AlignAI</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Enter a prompt below to analyze it for potential compliance risks
          </p>
        </div>

        {error && (
          <div className="mb-4 rounded-md bg-red-100 p-3 text-red-700 dark:bg-red-900/30 dark:text-red-300">
            <div className="flex items-center gap-2">
              <AlertCircleIcon className="h-5 w-5" />
              <p>{error}</p>
            </div>
          </div>
        )}

        <div className="mb-4">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter a prompt to analyze for EU AI Act compliance..."
            className="h-64 w-full rounded-md border border-gray-300 p-4 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800"
          />
        </div>

        <button
          onClick={analyzePrompt}
          disabled={isAnalyzing || !prompt.trim()}
          className="flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:bg-blue-400"
        >
          {isAnalyzing ? (
            <>
              <svg className="h-5 w-5 animate-spin" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                ></path>
              </svg>
              Analyzing...
            </>
          ) : (
            <>
              <ShieldIcon className="h-5 w-5" />
              Analyze Prompt
            </>
          )}
        </button>

        <div className="mt-6">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            This public sandbox provides basic analysis capabilities. For full features including
            comprehensive compliance monitoring, reporting, and organization-wide analytics,{' '}
            <a href="/signup" className="text-blue-600 hover:underline dark:text-blue-400">
              sign up
            </a>{' '}
            for a free account.
          </p>
        </div>
      </div>

      <div>
        <div className="mb-4">
          <h2 className="text-xl font-bold">Analysis Results</h2>
        </div>

        {results ? (
          <div className="space-y-6">
            {/* Overall Risk Assessment */}
            <div className={`rounded-md border p-4 ${getRiskLevelClass(results.overall_risk.score)}`}>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-medium">Overall Risk Assessment</h3>
                <span className={`rounded-full px-2 py-1 text-sm font-medium ${getRiskTextColorClass(results.overall_risk.score)}`}>
                  {getRiskLevelText(results.overall_risk.score)}
                </span>
              </div>
              <div className="mb-2 h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div
                  className={`h-2 rounded-full ${
                    results.overall_risk.score >= 0.7
                      ? 'bg-red-600'
                      : results.overall_risk.score >= 0.4
                      ? 'bg-yellow-600'
                      : 'bg-green-600'
                  }`}
                  style={{ width: `${results.overall_risk.score * 100}%` }}
                ></div>
              </div>
              <p className="text-sm">
                Risk Score: {Math.round(results.overall_risk.score * 100)}%
              </p>
            </div>

            {/* Findings */}
            {results.token_risks.length > 0 && (
              <div className="rounded-md border border-gray-300 p-4 dark:border-gray-700">
                <h3 className="mb-4 font-medium">Findings</h3>
                <div className="space-y-3">
                  {results.token_risks.map((risk, index) => (
                    <div
                      key={index}
                      className={`rounded-md p-3 ${
                        risk.risk_score >= 0.7
                          ? 'bg-red-100 dark:bg-red-900/30'
                          : risk.risk_score >= 0.4
                          ? 'bg-yellow-100 dark:bg-yellow-900/30'
                          : 'bg-green-100 dark:bg-green-900/30'
                      }`}
                    >
                      <div className="mb-1 flex items-center justify-between">
                        <span className="font-medium capitalize">{risk.risk_type}</span>
                        <span
                          className={`text-sm ${
                            risk.risk_score >= 0.7
                              ? 'text-red-700 dark:text-red-300'
                              : risk.risk_score >= 0.4
                              ? 'text-yellow-700 dark:text-yellow-300'
                              : 'text-green-700 dark:text-green-300'
                          }`}
                        >
                          {Math.round(risk.risk_score * 100)}%
                        </span>
                      </div>
                      <p className="text-sm">{risk.explanation}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Relevant Policies */}
            {results.relevant_policies.length > 0 && (
              <div className="rounded-md border border-gray-300 p-4 dark:border-gray-700">
                <h3 className="mb-4 font-medium">Relevant Policies</h3>
                <div className="space-y-3">
                  {results.relevant_policies.map((policy, index) => (
                    <div
                      key={index}
                      className="rounded-md border border-gray-200 p-3 dark:border-gray-800"
                    >
                      <p className="mb-1 font-medium">{policy.article}</p>
                      <p className="text-sm text-gray-700 dark:text-gray-300">{policy.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            <div className="rounded-md border border-gray-300 p-4 dark:border-gray-700">
              <h3 className="mb-4 font-medium">Recommendations</h3>
              <ul className="space-y-2 pl-5">
                {results.recommendations.map((recommendation, index) => (
                  <li key={index} className="list-disc text-sm">
                    {recommendation}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <div className="flex h-64 items-center justify-center rounded-md border border-gray-300 p-4 dark:border-gray-700">
            <p className="text-center text-gray-500 dark:text-gray-400">
              Enter a prompt and click "Analyze Prompt" to see results
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
