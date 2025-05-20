export interface TokenRisk {
  start: number;
  end: number;
  risk_score: number;
  risk_type: string;
  explanation: string;
}

export interface AnalysisOptions {
  analyze_bias?: boolean;
  analyze_pii?: boolean;
  analyze_policy?: boolean;
  language?: string;
  threshold?: number;
  categories?: {
    bias?: boolean;
    pii?: boolean;
    policy_violation?: boolean;
  };
}

export interface AnalysisResult {
  text: string;
  analysis?: {
    bias?: Record<string, unknown> | null;
    pii?: Record<string, unknown> | null;
    policy?: Record<string, unknown> | null;
  };
  token_risks: TokenRisk[];
  overall_risk: {
    score: number;
    categories: {
      bias?: number;
      pii?: number;
      policy_violation?: number;
      [key: string]: number | undefined;
    };
  };
  relevant_policies: Array<{
    article: string;
    text: string;
  }>;
  recommendations: string[];
  warnings: string[];
  is_safe: boolean;
}

// No need for API_BASE_URL since we're using Next.js API routes directly

export const analyzeText = async (
  text: string,
  options: AnalysisOptions = {}
): Promise<AnalysisResult> => {
  try {
    // Prepare the request payload
    const payload = {
      text,
      options: {
        analyze_bias: options.analyze_bias ?? true,
        analyze_pii: options.analyze_pii ?? true,
        analyze_policy: options.analyze_policy ?? true,
        language: options.language ?? 'en',
        threshold: options.threshold ?? 0.7,
      },
    };

    // Call the Next.js API route directly
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to analyze text');
    }

    const result = await response.json();
    return result as AnalysisResult;

  } catch (error) {
    console.error('Error in analyzeText:', error);
    throw new Error(
      error instanceof Error ? error.message : 'Failed to analyze text'
    );
  }
};

export const healthCheck = async (): Promise<{ status: string; timestamp: string; sandbox?: boolean }> => {
  const response = await fetch('/api/analyze');
  
  if (!response.ok) {
    throw new Error('API service is unavailable');
  }
  
  return response.json();
};
