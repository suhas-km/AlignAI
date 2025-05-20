import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Backend API URL for the fine-tuned models
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

// Helper function to handle API errors
function handleApiError(error: any, context: string) {
  console.error(`Error in ${context}:`, error);
  const status = error.status || 500;
  const message = error.message || 'Internal server error';
  return NextResponse.json(
    { error: message },
    { status }
  );
}

// Types for analysis
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

interface AnalysisRequest {
  text: string;
  options?: {
    analyze_bias?: boolean;
    analyze_pii?: boolean;
    analyze_policy?: boolean;
    language?: string;
    threshold?: number;
  };
}

export async function POST(request: NextRequest) {
  try {
    const { text, options = {} }: AnalysisRequest = await request.json();
    
    if (!text) {
      return NextResponse.json(
        { error: 'Text is required' },
        { status: 400 }
      );
    }
    
    // Check if at least one analysis option is selected
    if (options.analyze_bias === false && options.analyze_pii === false && options.analyze_policy === false) {
      return NextResponse.json(
        { error: 'At least one analysis type must be enabled' },
        { status: 400 }
      );
    }
    
    // Prepare the request for the fine-tuned models API
    const backendPayload = {
      text,
      options: {
        analyze_bias: options.analyze_bias ?? true,
        analyze_pii: options.analyze_pii ?? true,
        analyze_policy: options.analyze_policy ?? true,
        language: options.language || 'en',
        threshold: options.threshold ?? 0.7
      }
    };
    
    console.log('Calling backend API:', `${API_BASE_URL}/analyze/text`);
    const response = await fetch(`${API_BASE_URL}/analyze/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(backendPayload),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      const errorMessage = error.detail || 'Failed to analyze text with fine-tuned models';
      console.error('Backend API error:', errorMessage);
      return NextResponse.json(
        { error: errorMessage },
        { status: response.status || 500 }
      );
    }
    
    // Use only the fine-tuned model response
    const modelResponse = await response.json();
    console.log('Backend API response:', modelResponse);

    // Simply return the fine-tuned model response
    // No fallback or pattern matching - only using the fine-tuned models
    return NextResponse.json(modelResponse);
  } catch (error) {
    return handleApiError(error, 'analyze text');
  }
}

export async function GET() {
  // Return a mock health check response to match the sandbox approach
  return NextResponse.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    sandbox: true
  });
}
