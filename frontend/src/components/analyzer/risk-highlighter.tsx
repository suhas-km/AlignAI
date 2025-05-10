'use client';

import React, { useRef, useEffect } from 'react';

type TokenRisk = {
  start: number;
  end: number;
  risk_score: number;
  risk_type: string;
  explanation: string;
};

type RiskHighlighterProps = {
  text: string;
  risks: TokenRisk[];
  onTextChange: (text: string) => void;
};

export default function RiskHighlighter({ text, risks, onTextChange }: RiskHighlighterProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sort risks by start position to ensure proper rendering
  const sortedRisks = [...risks].sort((a, b) => a.start - b.start);

  // Prepare highlighted content
  const getHighlightedContent = () => {
    let result = [];
    let lastIndex = 0;

    // Go through each risk and create spans with appropriate classes
    sortedRisks.forEach((risk) => {
      if (risk.start > lastIndex) {
        // Add text before this risk
        result.push(text.substring(lastIndex, risk.start));
      }

      // Add the risky text with appropriate class
      const riskLevel = risk.risk_score >= 0.7 ? 'high' : risk.risk_score >= 0.4 ? 'medium' : 'low';
      const riskText = text.substring(risk.start, risk.end);
      
      result.push(
        <span 
          key={`risk-${risk.start}-${risk.end}`}
          className={`risk-${riskLevel}`}
          data-tooltip={`${risk.risk_type}: ${risk.explanation}`}
        >
          {riskText}
        </span>
      );

      lastIndex = risk.end;
    });

    // Add any remaining text
    if (lastIndex < text.length) {
      result.push(text.substring(lastIndex));
    }

    return result;
  };

  return (
    <div className="relative">
      <div className="prose prose-sm mb-2 max-w-none rounded-md border border-gray-300 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
        {getHighlightedContent()}
      </div>
      <textarea
        ref={textareaRef}
        value={text}
        onChange={(e) => onTextChange(e.target.value)}
        className="h-64 w-full rounded-md border border-gray-300 p-4 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800"
      />
    </div>
  );
}
