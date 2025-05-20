'use client';

import React from 'react';

// Types for our analysis data
type TokenRisk = {
  start: number;
  end: number;
  risk_score: number;
  risk_type: string;
  explanation: string;
};

type PolicyMatch = {
  policy_id: number;
  article: string;
  similarity_score: number;
  text_snippet: string;
};

type OverallRisk = {
  score: number;
  categories: Record<string, number>;
};

type AnalysisResponse = {
  token_risks: TokenRisk[];
  policy_matches: PolicyMatch[];
  overall_risk: OverallRisk;
  recommendations: string[];
};

type AnalysisResultsProps = {
  results: AnalysisResponse;
};

export default function AnalysisResults({ results }: AnalysisResultsProps) {
  // Format risk score as percentage
  const formatRiskScore = (score: number) => `${Math.round(score * 100)}%`;
  
  // Get color class based on risk score
  const getRiskColorClass = (score: number) => {
    if (score >= 0.7) return 'text-red-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-green-600';
  };
  
  // Get background color class based on risk score
  const getRiskBgClass = (score: number) => {
    if (score >= 0.7) return 'bg-red-100 dark:bg-red-900/30';
    if (score >= 0.4) return 'bg-yellow-100 dark:bg-yellow-900/30';
    return 'bg-green-100 dark:bg-green-900/30';
  };

  return (
    <div className="space-y-6">
      {/* Overall Risk Score */}
      <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
        <h3 className="mb-3 text-lg font-semibold">Overall Risk Assessment</h3>
        <div className="mb-4 flex items-end justify-between">
          <span className="text-sm text-gray-500">Risk Score</span>
          <span className={`text-xl font-bold ${getRiskColorClass(results.overall_risk.score)}`}>
            {formatRiskScore(results.overall_risk.score)}
          </span>
        </div>
        <div className="mb-1 h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
          <div 
            className={`h-2 rounded-full ${getRiskBgClass(results.overall_risk.score)}`}
            style={{ width: `${results.overall_risk.score * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Category Breakdown */}
      {Object.keys(results.overall_risk.categories).length > 0 && (
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-3 text-lg font-semibold">Risk Categories</h3>
          <div className="space-y-3">
            {Object.entries(results.overall_risk.categories).map(([category, score]) => (
              <div key={category} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="capitalize text-sm">{category.replace('_', ' ')}</span>
                  <span className={`text-sm font-medium ${getRiskColorClass(score)}`}>
                    {formatRiskScore(score)}
                  </span>
                </div>
                <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                  <div 
                    className={`h-2 rounded-full ${getRiskBgClass(score)}`}
                    style={{ width: `${score * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Token-Level Risks */}
      {results.token_risks.length > 0 && (
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-3 text-lg font-semibold">Detailed Findings</h3>
          <div className="max-h-60 overflow-y-auto space-y-3">
            {results.token_risks.map((risk, index) => (
              <div 
                key={`risk-${index}`}
                className={`rounded-md p-3 ${getRiskBgClass(risk.risk_score)}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium capitalize">{risk.risk_type}</span>
                  <span className={`text-sm font-medium ${getRiskColorClass(risk.risk_score)}`}>
                    {formatRiskScore(risk.risk_score)}
                  </span>
                </div>
                <p className="text-sm">{risk.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Policy Matches */}
      {results.policy_matches.length > 0 && (
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-3 text-lg font-semibold">Relevant Policies</h3>
          <div className="max-h-60 overflow-y-auto space-y-3">
            {results.policy_matches.map((match) => (
              <div 
                key={`policy-${match.policy_id}`}
                className="rounded-md border border-gray-300 p-3 dark:border-gray-700"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium">{match.article}</span>
                  <span className="text-sm text-gray-500">
                    Match: {formatRiskScore(match.similarity_score)}
                  </span>
                </div>
                <p className="text-sm text-gray-700 dark:text-gray-300">
                  {match.text_snippet}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
        <h3 className="mb-3 text-lg font-semibold">Recommendations</h3>
        <ul className="space-y-2 text-sm">
          {results.recommendations && results.recommendations.length > 0 ? (
            // Display ML-generated recommendations from the backend
            results.recommendations.map((recommendation, index) => (
              <li key={`rec-${index}`} className="flex items-start gap-2">
                <span className="mt-0.5 text-blue-600">●</span>
                <span>{recommendation}</span>
              </li>
            ))
          ) : results.token_risks.length > 0 || results.policy_matches.length > 0 ? (
            // Fallback recommendations if ML recommendations are missing but issues were detected
            <>
              {results.token_risks.some(r => r.risk_type === 'bias') && (
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 text-yellow-600">●</span>
                  <span>Review text for potential bias and consider using more neutral language.</span>
                </li>
              )}
              {results.token_risks.some(r => r.risk_type === 'pii') && (
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 text-red-600">●</span>
                  <span>Remove personally identifiable information to ensure GDPR compliance.</span>
                </li>
              )}
              {results.policy_matches.length > 0 && (
                <li className="flex items-start gap-2">
                  <span className="mt-0.5 text-blue-600">●</span>
                  <span>Review relevant EU AI Act articles to ensure regulatory compliance.</span>
                </li>
              )}
            </>
          ) : (
            <li className="text-green-600">No significant issues detected.</li>
          )}
        </ul>
      </div>
    </div>
  );
}
