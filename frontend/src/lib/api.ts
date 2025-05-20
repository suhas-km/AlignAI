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

// Backend API URL - we'll use this to call the actual fine-tuned models
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

export const analyzeText = async (
  text: string,
  options: AnalysisOptions = {}
): Promise<AnalysisResult> => {
  try {
    console.log('Analyzing text:', text.substring(0, 30) + '...');
    console.log('Options:', options);
    
    // Try backend service first, with fallback to sandbox analysis
    let useBackend = true;
    let result;
    
    if (useBackend) {
      try {
        // Call the backend service directly (bypassing Next.js API route)
        const backendUrl = `${API_BASE_URL}${API_VERSION}/analyze/text`;
        console.log('Calling backend service at:', backendUrl);
        
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
        
        const response = await fetch(backendUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        
        if (!response.ok) {
          throw new Error(`Backend service error: ${response.status} ${response.statusText}`);
        }
        
        result = await response.json();
        console.log('Backend service response:', result);
      } catch (backendError) {
        console.error('Backend service failed:', backendError);
        throw backendError; // Re-throw the error since we don't want to use fallback
      }
    }
    
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
