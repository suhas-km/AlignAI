import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

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

    // Prepare the request payload with default options
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

    // Call the backend API
    const response = await fetch(`${API_BASE_URL}/analyze/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      return NextResponse.json(
        { 
          error: error.detail || 'Failed to analyze text',
          status: response.status,
        },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return handleApiError(error, 'analyze text');
  }
}

export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze/health`);
    
    if (!response.ok) {
      throw { 
        status: response.status,
        message: 'Backend service is unavailable'
      };
    }

    const data = await response.json();
    return NextResponse.json({
      status: data.status || 'ok',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return handleApiError(error, 'health check');
  }
}
