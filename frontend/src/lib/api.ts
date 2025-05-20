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

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface ModelAnalysisResult {
  analysis: {
    bias?: {
      has_bias: boolean;
      score: number;
      explanation?: string;
    };
    pii?: {
      has_pii: boolean;
      entities: Array<{
        entity: string;
        type: string;
        start: number;
        end: number;
        score: number;
      }>;
    };
    policy?: {
      has_violation: boolean;
      violations: Array<{
        policy: string;
        severity: 'low' | 'medium' | 'high';
        explanation: string;
      }>;
    };
  };
  warnings: string[];
  is_safe: boolean;
}

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

    // Call the single analyze endpoint
    const response = await fetch(`${API_BASE_URL}/analyze/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || 'Failed to analyze text');
    }

    const data = await response.json();
    
    // Transform the response to match our frontend format
    const result: AnalysisResult = {
      text: data.text || text,
      token_risks: [],
      overall_risk: {
        score: 0,
        categories: {
          bias: 0,
          pii: 0,
          policy_violation: 0
        }
      },
      relevant_policies: [],
      recommendations: [],
      warnings: data.warnings || [],
      is_safe: data.is_safe !== false,
      analysis: {
        bias: data.analysis?.bias || null,
        pii: data.analysis?.pii || null,
        policy: data.analysis?.policy || null
      }
    };

    // Calculate overall risk score based on analysis results
    let totalScore = 0;
    let validScores = 0;

    // Process bias analysis
    if (data.analysis?.bias) {
      result.overall_risk.categories.bias = data.analysis.bias.score || 0;
      totalScore += data.analysis.bias.score || 0;
      validScores += data.analysis.bias.score ? 1 : 0;

      // Add bias risks to token risks
      if (data.analysis.bias.has_bias) {
        result.token_risks.push({
          start: 0,
          end: text.length,
          risk_score: data.analysis.bias.score || 0.5,
          risk_type: 'bias',
          explanation: data.analysis.bias.explanation || 'Potential bias detected'
        });
      }
    }

    // Process PII analysis
    if (data.analysis?.pii?.entities?.length) {
      // Calculate average PII score
      const piiScores = data.analysis.pii.entities.map((entity: { score?: number }) => entity.score || 0);
      const avgPiiScore = piiScores.length > 0 
        ? piiScores.reduce((acc: number, score: number) => acc + score, 0) / piiScores.length 
        : 0;
      
      result.overall_risk.categories.pii = avgPiiScore;
      totalScore += avgPiiScore;
      validScores += piiScores.length > 0 ? 1 : 0;

      // Add PII entities to token risks
      result.token_risks.push(
        ...data.analysis.pii.entities.map((entity: {
          start: number;
          end: number;
          score?: number;
          type: string;
          entity: string;
        }) => ({
          start: entity.start,
          end: entity.end,
          risk_score: entity.score || 0.5,
          risk_type: `pii_${entity.type}`,
          explanation: `Detected ${entity.type}: ${entity.entity}`
        }))
      );
    }

    // Process policy analysis
    if (data.analysis?.policy?.violations?.length) {
      // Calculate average policy violation score based on severity
      const severityScores = data.analysis.policy.violations.map((violation: { severity: string }) => 
        violation.severity === 'high' ? 1 : violation.severity === 'medium' ? 0.7 : 0.4
      );
      const avgPolicyScore = severityScores.length > 0
        ? severityScores.reduce((sum: number, score: number) => sum + score, 0) / severityScores.length
        : 0;
      
      result.overall_risk.categories.policy_violation = avgPolicyScore;
      totalScore += avgPolicyScore;
      validScores += severityScores.length > 0 ? 1 : 0;

      // Add policy violations to relevant policies
      result.relevant_policies = data.analysis.policy.violations.map((violation: {
        policy: string;
        explanation: string;
      }) => ({
        article: violation.policy,
        text: violation.explanation
      }));
    }

    // Calculate overall risk score
    if (validScores > 0) {
      result.overall_risk.score = totalScore / validScores;
    }

    return result;
  } catch (error) {
    console.error('Error in analyzeText:', error);
    throw new Error(
      error instanceof Error ? error.message : 'Failed to analyze text'
    );
  }
};

export const healthCheck = async (): Promise<{ status: string }> => {
  const response = await fetch('/api/analyze/health');
  if (!response.ok) {
    throw new Error('Service unavailable');
  }
  return response.json();
};
