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
    }
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
      // Call the API with the current analysis options
      const analysis = await analyzeText(prompt, analysisOptions);
      
      // Safely extract the overall_risk with proper defaults
      const overallRisk = analysis.overall_risk || { score: 0, categories: {} };
      
      // Ensure categories has all required properties
      const categories = {
        bias: 0,
        pii: 0,
        policy_violation: 0,
        ...(overallRisk.categories || {})
      };

      // Build the enhanced result with proper typing
      const enhancedResult: AnalysisResult = {
        text: analysis.text || prompt,
        token_risks: analysis.token_risks || [],
        overall_risk: {
          score: overallRisk.score || 0,
          categories: categories
        },
        relevant_policies: analysis.relevant_policies || [],
        recommendations: analysis.recommendations || [],
        warnings: analysis.warnings || [],
        is_safe: analysis.is_safe !== undefined ? analysis.is_safe : true,
        analysis: {
          bias: analysis.analysis?.bias || null,
          pii: analysis.analysis?.pii || null,
          policy: analysis.analysis?.policy || null
        }
      };
      
      setResult(enhancedResult);
    } catch (err: any) {
      console.error('Analysis error:', err);
      setError({
        title: 'Analysis Error',
        message: err?.message || 'Failed to analyze text'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Text Analysis</CardTitle>
          <CardDescription>
            Enter your text and select the types of analysis to perform
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="prompt">Text to analyze</Label>
            <Textarea
              id="prompt"
              value={prompt}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value)}
              placeholder="Enter text to analyze for bias, PII, and policy violations..."
              className="min-h-[200px]"
              disabled={isAnalyzing}
            />
          </div>
          
          <div className="space-y-4">
            <Label>Analysis Options</Label>
            <div className="grid gap-4 md:grid-cols-3">
                <div className="flex items-center space-x-2 rounded-lg border p-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center">
                      <User className="mr-2 h-4 w-4 text-blue-500" />
                      <Label htmlFor="pii-analysis">PII Detection</Label>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Identify personally identifiable information
                    </p>
                  </div>
                  <Switch 
                    id="pii-analysis" 
                    checked={analysisOptions.analyze_pii}
                    onCheckedChange={() => toggleAnalysisOption('analyze_pii')}
                    disabled={isAnalyzing}
                  />
                </div>
              
              <div className="flex items-center space-x-2 rounded-lg border p-4">
                <div className="flex-1 space-y-1">
                  <div className="flex items-center">
                    <Scale className="mr-2 h-4 w-4 text-purple-500" />
                    <Label htmlFor="bias-analysis">Bias Analysis</Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Detect potential biases in the text
                  </p>
                </div>
                <Switch 
                  id="bias-analysis" 
                  checked={analysisOptions.analyze_bias}
                  onCheckedChange={() => toggleAnalysisOption('analyze_bias')}
                  disabled={isAnalyzing}
                />
              </div>
              
              <div className="flex items-center space-x-2 rounded-lg border p-4">
                <div className="flex-1 space-y-1">
                  <div className="flex items-center">
                    <Shield className="mr-2 h-4 w-4 text-amber-500" />
                    <Label htmlFor="policy-analysis">Policy Compliance</Label>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Check for compliance with AI policies
                  </p>
                </div>
                <Switch 
                  id="policy-analysis" 
                  checked={analysisOptions.analyze_policy}
                  onCheckedChange={() => toggleAnalysisOption('analyze_policy')}
                  disabled={isAnalyzing}
                />
              </div>
            </div>
          </div>
          
          <Button 
            onClick={handleAnalyze} 
            disabled={isAnalyzing || !prompt.trim()}
            className="w-full md:w-auto"
            size="lg"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Text'
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
                {result.is_safe ? 'No Issues Found' : 'Potential Issues Detected'}
              </CardTitle>
            </div>
            <CardDescription>
              {result.is_safe 
                ? 'The text appears to be compliant with standard guidelines.'
                : 'Review the analysis below for potential issues.'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Overall Risk Assessment */}
            <div className={`rounded-lg border p-4 ${getRiskLevelClass(displayResult.overall_risk.score)}`}>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-lg font-medium">Overall Risk Assessment</h3>
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
                <h3 className="mb-4 font-medium">Findings</h3>
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
            {(displayResult.analysis?.bias || displayResult.analysis?.pii || displayResult.analysis?.policy) && (
              <Tabs defaultValue="bias" className="mt-6">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger 
                    value="bias" 
                    disabled={!displayResult.analysis?.bias}
                    className="flex items-center gap-2"
                  >
                    <Scale className="h-4 w-4" />
                    <span>Bias Analysis</span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="pii" 
                    disabled={!displayResult.analysis?.pii}
                    className="flex items-center gap-2"
                  >
                    <User className="h-4 w-4" />
                    <span>PII Detection</span>
                  </TabsTrigger>
                  <TabsTrigger 
                    value="policy" 
                    disabled={!displayResult.analysis?.policy}
                    className="flex items-center gap-2"
                  >
                    <Shield className="h-4 w-4" />
                    <span>Policy Check</span>
                  </TabsTrigger>
                </TabsList>
                
                <div className="mt-4 rounded-lg border p-4">
                  <TabsContent value="bias" className="m-0">
                    {displayResult.analysis?.bias ? (
                      <div className="space-y-4">
                        <h4 className="font-medium">Bias Analysis Results</h4>
                        <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                          {JSON.stringify(displayResult.analysis.bias, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No bias analysis was performed on this text.
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="pii" className="m-0">
                    {displayResult.analysis?.pii ? (
                      <div className="space-y-4">
                        <h4 className="font-medium">PII Detection Results</h4>
                        <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                          {JSON.stringify(displayResult.analysis.pii, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No PII analysis was performed on this text.
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="policy" className="m-0">
                    {displayResult.analysis?.policy ? (
                      <div className="space-y-4">
                        <h4 className="font-medium">Policy Compliance Results</h4>
                        <pre className="max-h-96 overflow-auto rounded-md bg-muted p-4 text-sm">
                          {JSON.stringify(displayResult.analysis.policy, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-8 text-muted-foreground">
                        <Info className="mr-2 h-4 w-4" />
                        No policy analysis was performed on this text.
                      </div>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            )}

            {/* Relevant Policies */}
            {displayResult.relevant_policies.length > 0 && (
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
