'use client';

import { useState } from 'react';
import { analyzeText, type AnalysisResult, type AnalysisOptions, type TokenRisk } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { 
  AlertCircle, 
  CheckCircle, 
  Loader2, 
  ShieldAlert, 
  Shield, 
  User, 
  Scale, 
  Info 
} from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AnalysisProgress } from '@/components/ui/analysis-progress';

// Reusing TokenRisk type from the API client

// Helper functions for risk assessment
const getRiskLevelClass = (score: number) => {
  if (score >= 0.7) return 'bg-red-100 border-red-300 dark:bg-red-900/30 dark:border-red-800';
  if (score >= 0.4) return 'bg-yellow-100 border-yellow-300 dark:bg-yellow-900/30 dark:border-yellow-800';
  return 'bg-green-100 border-green-300 dark:bg-green-900/30 dark:border-green-800';
};

const getRiskLevelText = (score: number) => {
  if (score >= 0.7) return 'High Risk';
  if (score >= 0.4) return 'Medium Risk';
  return 'Low Risk';
};

const getRiskTextColorClass = (score: number) => {
  if (score >= 0.7) return 'text-red-700 dark:text-red-300';
  if (score >= 0.4) return 'text-yellow-700 dark:text-yellow-300';
  return 'text-green-700 dark:text-green-300';
};

export default function SimplePromptAnalyzer() {
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  
  // Initialize with empty values to prevent undefined errors
  const safeResult: AnalysisResult = {
    text: '',
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
    warnings: [],
    is_safe: true,
    analysis: {
      bias: null,
      pii: null,
      policy: null
    } as const
  };
  
  const displayResult = result || safeResult;
  const [error, setError] = useState<{title: string; message: string} | null>(null);
  // Initialize with default analysis options
  const [analysisOptions, setAnalysisOptions] = useState<AnalysisOptions>({
    analyze_bias: true,
    analyze_pii: true,
    analyze_policy: true,
    threshold: 0.7,
    language: 'en'
  });
  
  const toggleAnalysisOption = (option: 'analyze_bias' | 'analyze_pii' | 'analyze_policy') => {
    setAnalysisOptions(prev => ({
      ...prev,
      [option]: !prev[option]
    }));
  };

  const handleAnalyze = async () => {
    if (!prompt.trim()) {
      setError({
        title: 'Input Required',
        message: 'Please enter some text to analyze'
      });
      return;
    }

    // Check if at least one analysis option is selected
    if (!analysisOptions.analyze_bias && !analysisOptions.analyze_pii && !analysisOptions.analyze_policy) {
      setError({
        title: 'Analysis Required',
        message: 'Please select at least one analysis type'
      });
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      // Direct API call to the backend service
      const backendUrl = 'http://localhost:8000/api/v1/analyze/text';
      console.log('Calling backend at:', backendUrl);
      console.log('Analysis options:', analysisOptions);
      
      // Prepare the payload
      const payload = {
        text: prompt,
        options: {
          analyze_bias: analysisOptions.analyze_bias,
          analyze_pii: analysisOptions.analyze_pii,
          analyze_policy: analysisOptions.analyze_policy,
          language: analysisOptions.language || 'en',
          threshold: analysisOptions.threshold || 0.7
        }
      };
      
      // Make the API call directly to the backend
      const response = await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const responseData = await response.json();
      console.log('API response:', responseData);

      // Calculate risk scores from the backend response
      let biasScore = 0;
      let piiScore = 0;
      let policyScore = 0;

      if (responseData.analysis) {
        if (responseData.analysis.bias) {
          biasScore = responseData.analysis.bias.overall_score || 0;
        }
        
        if (responseData.analysis.pii) {
          piiScore = responseData.analysis.pii.overall_score || 0;
        }
        
        if (responseData.analysis.policy) {
          policyScore = responseData.analysis.policy.overall_score || 0;
        }
      }

      // Calculate overall score as the maximum of the three component scores
      const overallScore = Math.max(biasScore, piiScore, policyScore);

      // Create the result object with proper structure
      const analysisResult: AnalysisResult = {
        text: responseData.text,
        token_risks: [],
        overall_risk: {
          score: overallScore,
          categories: {
            bias: biasScore,
            pii: piiScore,
            policy_violation: policyScore
          }
        },
        relevant_policies: [],
        recommendations: [],
        warnings: responseData.warnings || [],
        is_safe: responseData.is_safe !== false,
        analysis: {
          bias: responseData.analysis?.bias || null,
          pii: responseData.analysis?.pii || null,
          policy: responseData.analysis?.policy || null
        }
      };

      setResult(analysisResult);
    } catch (error: unknown) {
      console.error('Error analyzing text:', error);
      setError({
        title: 'Analysis Failed',
        message: error instanceof Error ? error.message : 'Failed to connect to analysis service'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Analysis Tool</CardTitle>
          <CardDescription>
            Check your content against EU AI Act requirements
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="prompt">Content</Label>
            <Textarea
              id="prompt"
              value={prompt}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value)}
              placeholder="Paste your AI prompt or generated content here..."
              className="min-h-[200px]"
              disabled={isAnalyzing}
            />
          </div>
          
          <div className="space-y-4">
            <Label>Analysis Options</Label>
            <div className="grid gap-4 md:grid-cols-3">
                <div className="flex items-center space-x-2 rounded-lg border p-4">
                  <div className="flex-1 space-y-1">
                    <Label htmlFor="analyze-pii" className="flex items-center gap-2 cursor-pointer">
                      <User className="h-4 w-4" />
                      Find PII
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Identify personally identifiable information
                    </p>
                  </div>
                  <Switch 
                    id="analyze-pii" 
                    checked={analysisOptions.analyze_pii}
                    onCheckedChange={() => toggleAnalysisOption('analyze_pii')}
                    disabled={isAnalyzing}
                  />
                </div>
              
              <div className="flex items-center space-x-2 rounded-lg border p-4">
                <div className="flex-1 space-y-1">
                  <Label htmlFor="analyze-bias" className="flex items-center gap-2 cursor-pointer">
                    <Scale className="h-4 w-4" />
                    Detect Bias
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Detect potential biases in the text
                  </p>
                </div>
                <Switch 
                  id="analyze-bias" 
                  checked={analysisOptions.analyze_bias}
                  onCheckedChange={() => toggleAnalysisOption('analyze_bias')}
                  disabled={isAnalyzing}
                />
              </div>
              
              <div className="flex items-center space-x-2 rounded-lg border p-4">
                <div className="flex-1 space-y-1">
                  <Label htmlFor="analyze-policy" className="flex items-center gap-2 cursor-pointer">
                    <Shield className="h-4 w-4" />
                    EU Compliance
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Check for compliance with AI policies
                  </p>
                </div>
                <Switch 
                  id="analyze-policy" 
                  checked={analysisOptions.analyze_policy}
                  onCheckedChange={() => toggleAnalysisOption('analyze_policy')}
                  disabled={isAnalyzing}
                />
              </div>
            </div>
          </div>
          
          <Button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="w-full"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze'
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <p className="font-medium">{error.title}</p>
            <p>{error.message}</p>
          </AlertDescription>
        </Alert>
      )}

      <AnalysisProgress 
        isAnalyzing={isAnalyzing} 
        analysisOptions={analysisOptions} 
      />

      {result && (
        <Card>
          <CardHeader>
            <div className="flex items-center">
              {result.is_safe ? (
                <CheckCircle className="mr-2 h-5 w-5 text-green-500" />
              ) : (
                <ShieldAlert className="mr-2 h-5 w-5 text-destructive" />
              )}
              <CardTitle>
                {result.is_safe ? 'Content Compliant' : 'Issues Detected'}
              </CardTitle>
            </div>
            <CardDescription>
              {result.is_safe 
                ? 'Your content appears to be compliant with EU AI Act guidelines.'
                : 'Review the analysis below to address potential issues.'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Overall Risk Assessment */}
            <div className={`rounded-lg border p-4 ${getRiskLevelClass(displayResult.overall_risk.score)}`}>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-lg font-medium">Risk Summary</h3>
                <span className={`rounded-full px-3 py-1 text-sm font-medium ${getRiskTextColorClass(displayResult.overall_risk.score)}`}>
                  {getRiskLevelText(displayResult.overall_risk.score)}
                </span>
              </div>
              <div className="mb-2 h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
                <div
                  className={`h-2 rounded-full ${
                    displayResult.overall_risk.score >= 0.7
                      ? 'bg-red-600'
                      : displayResult.overall_risk.score >= 0.4
                      ? 'bg-yellow-600'
                      : 'bg-green-600'
                  }`}
                  style={{ width: `${displayResult.overall_risk.score * 100}%` }}
                ></div>
              </div>
              <p className="text-sm">
                Risk Score: {Math.round(displayResult.overall_risk.score * 100)}%
              </p>
            </div>

            {/* Findings */}
            {displayResult.token_risks.length > 0 && (
              <div className="rounded-md border p-4">
                <h3 className="mb-4 font-medium">Key Issues</h3>
                <div className="space-y-3">
                  {displayResult.token_risks.map((risk, i) => (
                    <div
                      key={i}
                      className={`rounded-md p-3 ${
                        risk.risk_score >= 0.7
                          ? 'bg-red-100 dark:bg-red-900/30'
                          : risk.risk_score >= 0.4
                          ? 'bg-yellow-100 dark:bg-yellow-900/30'
                          : 'bg-green-100 dark:bg-green-900/30'
                      }`}
                    >
                      <div className="mb-1 flex items-center justify-between">
                        <span className="font-medium capitalize">{risk.risk_type.replace('_', ' ')}</span>
                        <span className={getRiskTextColorClass(risk.risk_score)}>
                          {Math.round(risk.risk_score * 100)}%
                        </span>
                      </div>
                      <p className="text-sm">{risk.explanation}</p>
                      {risk.start !== undefined && risk.end !== undefined && (
                        <p className="mt-1 text-xs text-muted-foreground">
                          Text: {displayResult.text.substring(risk.start, risk.end)}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Detailed Analysis Tabs - Only show if at least one analysis was performed */}
            {(displayResult.analysis && (displayResult.analysis.bias || displayResult.analysis.pii || displayResult.analysis.policy)) && (
              <Tabs defaultValue="bias" className="mt-6">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger 
                    value="bias" 
                    disabled={!displayResult.analysis?.bias}
                    className={`flex items-center gap-2 ${displayResult.analysis?.bias && displayResult.analysis.bias !== null && typeof displayResult.analysis.bias === 'object' && 'has_bias' in displayResult.analysis.bias && Boolean(displayResult.analysis.bias.has_bias) ? getRiskTextColorClass(displayResult.overall_risk.categories.bias ?? 0) : ''}`}
                  >
                    <Scale className="h-4 w-4" />
                    <span>Bias</span>
                    {displayResult.analysis?.bias && displayResult.analysis.bias !== null && typeof displayResult.analysis.bias === 'object' && 'has_bias' in displayResult.analysis.bias && Boolean(displayResult.analysis.bias.has_bias) && (
                      <span className="ml-1 text-xs rounded-full bg-background px-1.5 py-0.5">
                        {`${Math.round((displayResult.overall_risk.categories.bias ?? 0) * 100)}%`}
                      </span>
                    )}
                  </TabsTrigger>
                  <TabsTrigger 
                    value="pii" 
                    disabled={!displayResult.analysis?.pii}
                    className={`flex items-center gap-2 ${displayResult.analysis?.pii && displayResult.analysis.pii !== null && typeof displayResult.analysis.pii === 'object' && 'has_pii' in displayResult.analysis.pii && Boolean(displayResult.analysis.pii.has_pii) ? getRiskTextColorClass(displayResult.overall_risk.categories.pii ?? 0) : ''}`}
                  >
                    <User className="h-4 w-4" />
                    <span>PII</span>
                    {displayResult.analysis?.pii && displayResult.analysis.pii !== null && typeof displayResult.analysis.pii === 'object' && 'has_pii' in displayResult.analysis.pii && Boolean(displayResult.analysis.pii.has_pii) && (
                      <span className="ml-1 text-xs rounded-full bg-background px-1.5 py-0.5">
                        {`${Math.round((displayResult.overall_risk.categories.pii ?? 0) * 100)}%`}
                      </span>
                    )}
                  </TabsTrigger>
                  <TabsTrigger 
                    value="policy" 
                    disabled={!displayResult.analysis?.policy}
                    className={`flex items-center gap-2 ${displayResult.analysis?.policy && displayResult.analysis.policy !== null && typeof displayResult.analysis.policy === 'object' && 'has_violation' in displayResult.analysis.policy && Boolean(displayResult.analysis.policy.has_violation) ? getRiskTextColorClass(displayResult.overall_risk.categories.policy_violation ?? 0) : ''}`}
                  >
                    <Shield className="h-4 w-4" />
                    <span>Compliance</span>
                    {displayResult.analysis?.policy && displayResult.analysis.policy !== null && typeof displayResult.analysis.policy === 'object' && 'has_violation' in displayResult.analysis.policy && Boolean(displayResult.analysis.policy.has_violation) && (
                      <span className="ml-1 text-xs rounded-full bg-background px-1.5 py-0.5">
                        {`${Math.round((displayResult.overall_risk.categories.policy_violation ?? 0) * 100)}%`}
                      </span>
                    )}
                  </TabsTrigger>
                </TabsList>
                
                <div className="mt-4 rounded-lg border p-4">
                  <TabsContent value="bias" className="m-0">
                    {displayResult.analysis?.bias ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">Bias Assessment</h4>
                          <span className={`rounded-full px-3 py-1 text-xs font-medium ${getRiskTextColorClass(displayResult.overall_risk.categories.bias || 0)}`}>
                            {getRiskLevelText(displayResult.overall_risk.categories.bias || 0)}
                          </span>
                        </div>
                        
                        {displayResult.analysis.bias.has_bias ? (
                          <div className="space-y-4">
                            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 dark:border-amber-600 dark:bg-amber-950/30">
                              <p className="text-sm">Potentially biased content detected. Review the details below.</p>
                            </div>
                            
                            {displayResult.analysis.bias.instances && Array.isArray(displayResult.analysis.bias.instances) && displayResult.analysis.bias.instances.length > 0 ? (
                              <div className="space-y-3">
                                <h5 className="text-sm font-medium">Detected Issues:</h5>
                                {displayResult.analysis.bias.instances.map((instance: any, idx: number) => (
                                  <div key={idx} className="rounded-md border p-3">
                                    <div className="flex justify-between">
                                      <span className="font-medium">{instance.type || "Potential Bias"}</span>
                                      <span className={`text-sm ${getRiskTextColorClass(instance.score || 0)}`}>
                                        {`${Math.round((instance.score || 0) * 100)}% confidence`}
                                      </span>
                                    </div>
                                    <p className="mt-1 text-sm">{String(instance.explanation || "This content contains potentially biased language.")}</p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                                {JSON.stringify(displayResult.analysis.bias, null, 2)}
                              </pre>
                            )}
                          </div>
                        ) : (
                          <div className="rounded-lg border border-green-300 bg-green-50 p-4 dark:border-green-600 dark:bg-green-950/30">
                            <div className="flex items-center">
                              <CheckCircle className="mr-2 h-5 w-5 text-green-600" />
                              <p>No bias detected in the provided text.</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No bias analysis requested.
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="pii" className="m-0">
                    {displayResult.analysis?.pii ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">Personal Information</h4>
                          <span className={`rounded-full px-3 py-1 text-xs font-medium ${getRiskTextColorClass(displayResult.overall_risk.categories.pii || 0)}`}>
                            {getRiskLevelText(displayResult.overall_risk.categories.pii || 0)}
                          </span>
                        </div>
                        
                        {displayResult.analysis.pii.has_pii ? (
                          <div className="space-y-4">
                            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 dark:border-amber-600 dark:bg-amber-950/30">
                              <p className="text-sm">Personally identifiable information (PII) detected. Review the details below.</p>
                            </div>
                            
                            {displayResult.analysis.pii.instances && Array.isArray(displayResult.analysis.pii.instances) && displayResult.analysis.pii.instances.length > 0 ? (
                              <div className="space-y-3">
                                <h5 className="text-sm font-medium">Detected PII:</h5>
                                {displayResult.analysis.pii.instances.map((instance: any, idx: number) => (
                                  <div key={idx} className="rounded-md border p-3">
                                    <div className="flex justify-between">
                                      <span className="font-medium">{instance.type || "PII Data"}</span>
                                      <span className={`text-sm ${getRiskTextColorClass(instance.score || 0)}`}>
                                        {`${Math.round((instance.score || 0) * 100)}% confidence`}
                                      </span>
                                    </div>
                                    {instance.matched_text && (
                                      <p className="mt-1 text-sm font-mono bg-muted p-1 rounded">{String(instance.matched_text)}</p>
                                    )}
                                    <p className="mt-1 text-sm">{String(instance.explanation || "This content contains personally identifiable information.")}</p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                                {JSON.stringify(displayResult.analysis.pii, null, 2)}
                              </pre>
                            )}
                          </div>
                        ) : (
                          <div className="rounded-lg border border-green-300 bg-green-50 p-4 dark:border-green-600 dark:bg-green-950/30">
                            <div className="flex items-center">
                              <CheckCircle className="mr-2 h-5 w-5 text-green-600" />
                              <p>No personally identifiable information detected in the provided text.</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No PII analysis requested.
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="policy" className="m-0">
                    {displayResult.analysis?.policy ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium">Policy Compliance Results</h4>
                          <span className={`rounded-full px-3 py-1 text-xs font-medium ${getRiskTextColorClass(displayResult.overall_risk.categories.policy_violation || 0)}`}>
                            {getRiskLevelText(displayResult.overall_risk.categories.policy_violation || 0)}
                          </span>
                        </div>
                        
                        {displayResult.analysis.policy.has_violation ? (
                          <div className="space-y-4">
                            <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 dark:border-amber-600 dark:bg-amber-950/30">
                              <p className="text-sm">Policy violations detected. Review the details below.</p>
                            </div>
                            
                            {displayResult.analysis.policy.violations && Array.isArray(displayResult.analysis.policy.violations) && displayResult.analysis.policy.violations.length > 0 ? (
                              <div className="space-y-3">
                                <h5 className="text-sm font-medium">Detected Violations:</h5>
                                {displayResult.analysis.policy.violations.map((violation: any, idx: number) => (
                                  <div key={idx} className="rounded-md border p-3">
                                    <div className="flex justify-between">
                                      <span className="font-medium">{violation.policy || "Policy Violation"}</span>
                                      <span className={`text-sm ${getRiskTextColorClass(violation.score || 0)}`}>
                                        {`${Math.round((violation.score || 0) * 100)}% confidence`}
                                      </span>
                                    </div>
                                    <p className="mt-1 text-sm">{String(violation.explanation || "This content may violate EU AI Act policies.")}</p>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                                {typeof displayResult.analysis.policy === 'object' && displayResult.analysis.policy !== null
                                  ? JSON.stringify(displayResult.analysis.policy, null, 2)
                                  : displayResult.analysis.policy !== null && displayResult.analysis.policy !== undefined
                                    ? String(displayResult.analysis.policy)
                                    : '(No data available)'}
                              </pre>
                            )}
                            
                            {displayResult.analysis.policy && 'relevant_policies' in displayResult.analysis.policy && Array.isArray(displayResult.analysis.policy.relevant_policies) && displayResult.analysis.policy.relevant_policies.length > 0 && (
                              <div className="space-y-3">
                                <h5 className="text-sm font-medium">Relevant EU AI Act Policies:</h5>
                                {displayResult.analysis.policy.relevant_policies.map((policy, idx) => (
                                  <div key={idx} className="rounded-md border p-3 bg-blue-50 dark:bg-blue-950/30">
                                    <div className="font-medium">{String(policy.article || "EU AI Act Reference")}</div>
                                    <p className="mt-1 text-sm italic">{String(policy.text || "")}</p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="rounded-lg border border-green-300 bg-green-50 p-4 dark:border-green-600 dark:bg-green-950/30">
                            <div className="flex items-center">
                              <CheckCircle className="mr-2 h-5 w-5 text-green-600" />
                              <p>No policy violations detected in the provided text.</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No compliance analysis requested.
                      </div>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            )}

            {/* Relevant Policies */}
            {displayResult.relevant_policies && displayResult.relevant_policies.length > 0 && (
              <div className="rounded-md border p-4">
                <h3 className="mb-4 font-medium">Relevant Policies</h3>
                <div className="space-y-3">
                  {displayResult.relevant_policies.map((policy, i) => (
                    <div key={i} className="rounded-md border p-3">
                      <p className="font-medium">{policy.article}</p>
                      <p className="text-sm text-muted-foreground">{policy.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {displayResult.recommendations.length > 0 && (
              <div className="rounded-md border p-4">
                <h3 className="mb-4 font-medium">Recommendations</h3>
                <ul className="space-y-2">
                  {displayResult.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start">
                      <span className="mr-2 mt-0.5">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
