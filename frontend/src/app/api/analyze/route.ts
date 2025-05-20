import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

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

    // Create a sandbox-style response based on the prompt content
    const analyzeResponse: AnalysisResponse = {
      token_risks: [],
      overall_risk: {
        score: 0.2,
        categories: {
          bias: options.analyze_bias === false ? 0 : 0.1,
          pii: options.analyze_pii === false ? 0 : 0.0,
          policy_violation: options.analyze_policy === false ? 0 : 0.3,
        },
      },
      relevant_policies: [],
      recommendations: ['No significant issues detected.'],
    };

    // Analysis logic for bias
    if (options.analyze_bias !== false && 
        (text.toLowerCase().includes('gender') || 
         text.toLowerCase().includes('woman') || 
         text.toLowerCase().includes('man'))) {
      
      const biasWord = text.toLowerCase().includes('gender') ? 'gender' : 
                       text.toLowerCase().includes('woman') ? 'woman' : 'man';
      const start = text.toLowerCase().indexOf(biasWord);
      const end = start + biasWord.length;
      
      analyzeResponse.token_risks.push({
        start,
        end,
        risk_score: 0.7,
        risk_type: 'bias',
        explanation: 'Potential gender bias detected.',
      });
      
      analyzeResponse.overall_risk.score = Math.max(analyzeResponse.overall_risk.score, 0.7);
      analyzeResponse.overall_risk.categories.bias = 0.7;
      analyzeResponse.recommendations = ['Consider using gender-neutral language to avoid potential bias.'];
      analyzeResponse.relevant_policies.push({
        article: 'Article 10.2',
        text: 'High-risk AI systems that continue to learn after being placed on the market or put into service shall be developed in such a way to ensure that possibly biased outputs due to outputs used as an input for future operations ("feedback loops") are duly addressed with appropriate mitigation measures.',
      });
    }

    // Analysis logic for PII
    if (options.analyze_pii !== false && 
        (text.toLowerCase().includes('email') || text.toLowerCase().includes('@'))) {
      
      // Find the @ symbol for email
      const emailStart = text.toLowerCase().indexOf('@');
      
      // Get approximate start of email by finding the last space before the @ symbol
      let start = 0;
      if (emailStart > 0) {
        const spaceBeforeEmail = text.lastIndexOf(' ', emailStart);
        start = spaceBeforeEmail !== -1 ? spaceBeforeEmail + 1 : 0;
      }
      
      // Approximate end by finding the next space after the @ symbol
      const end = text.indexOf(' ', emailStart) !== -1 ? 
                 text.indexOf(' ', emailStart) : text.length;
      
      analyzeResponse.token_risks.push({
        start,
        end,
        risk_score: 0.9,
        risk_type: 'pii',
        explanation: 'Email address detected. Personal information should be anonymized.',
      });
      
      analyzeResponse.overall_risk.score = Math.max(analyzeResponse.overall_risk.score, 0.9);
      analyzeResponse.overall_risk.categories.pii = 0.9;
      analyzeResponse.recommendations.push('Remove or anonymize the email address to protect personal information.');
      analyzeResponse.relevant_policies.push({
        article: 'GDPR Article 5',
        text: 'Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject.',
      });
    }

    // Analysis logic for policy violations
    if (options.analyze_policy !== false && 
        (text.toLowerCase().includes('illegal') || 
         text.toLowerCase().includes('exploit') || 
         text.toLowerCase().includes('hack'))) {
      
      const violationWord = text.toLowerCase().includes('illegal') ? 'illegal' : 
                           text.toLowerCase().includes('exploit') ? 'exploit' : 'hack';
      const start = text.toLowerCase().indexOf(violationWord);
      const end = start + violationWord.length;
      
      analyzeResponse.token_risks.push({
        start,
        end,
        risk_score: 0.8,
        risk_type: 'policy_violation',
        explanation: 'Potential policy violation detected. Content may involve improper activities.',
      });
      
      analyzeResponse.overall_risk.score = Math.max(analyzeResponse.overall_risk.score, 0.8);
      analyzeResponse.overall_risk.categories.policy_violation = 0.8;
      analyzeResponse.recommendations.push('Review and revise content to ensure compliance with acceptable use policies.');
      analyzeResponse.relevant_policies.push({
        article: 'EU AI Act Article 5',
        text: 'AI systems intended to be used for the categorisation of natural persons based on biometric data according to ethnicity, gender, political or sexual orientation, or other prohibited grounds of discrimination.'
      });
    }

    // Add analysis data structure to match the API responses
    const enhancedResponse = {
      ...analyzeResponse,
      text,
      analysis: {
        bias: options.analyze_bias !== false ? {
          has_bias: analyzeResponse.overall_risk.categories.bias > 0.5,
          score: analyzeResponse.overall_risk.categories.bias,
          explanation: analyzeResponse.overall_risk.categories.bias > 0.5 ? 'Potential bias detected in text' : 'No significant bias detected'
        } : null,
        pii: options.analyze_pii !== false ? {
          has_pii: analyzeResponse.overall_risk.categories.pii > 0.5,
          entities: analyzeResponse.token_risks
            .filter(risk => risk.risk_type === 'pii')
            .map(risk => ({
              entity: text.substring(risk.start, risk.end),
              type: 'email',
              start: risk.start,
              end: risk.end,
              score: risk.risk_score
            }))
        } : null,
        policy: options.analyze_policy !== false ? {
          has_violation: analyzeResponse.overall_risk.categories.policy_violation > 0.5,
          violations: analyzeResponse.token_risks
            .filter(risk => risk.risk_type === 'policy_violation')
            .map(risk => ({
              policy: 'EU AI Act Article 5',
              severity: risk.risk_score > 0.7 ? 'high' : 'medium',
              explanation: risk.explanation
            }))
        } : null
      },
      warnings: analyzeResponse.token_risks.map(risk => risk.explanation),
      is_safe: analyzeResponse.overall_risk.score < 0.7
    };

    return NextResponse.json(enhancedResponse);
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
