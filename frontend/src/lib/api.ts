// Note: Local ONNX inference is disabled due to webpack bundling issues with onnxruntime-node
// The ONNX model is uploaded to HuggingFace at suhas-km/eu-ai-act-policy-model/onnx
// To enable client-side inference, a standalone web worker approach is needed
// See: https://github.com/xenova/transformers.js/issues/366

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

// Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = '/api/v1';

export const analyzeText = async (
  text: string,
  options: AnalysisOptions = {}
): Promise<AnalysisResult> => {
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
  
  return await response.json();
};

export const healthCheck = async (): Promise<{ status: string; timestamp: string; sandbox?: boolean }> => {
  const response = await fetch('/api/analyze');
  
  if (!response.ok) {
    throw new Error('API service is unavailable');
  }
  
  return response.json();
};
