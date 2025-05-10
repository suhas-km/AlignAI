'use client';

import { useState, useEffect, useRef } from 'react';
import AnalysisResults from './analysis-results';
import RiskHighlighter from './risk-highlighter';
import { AlertCircleIcon, ShieldIcon } from '@/components/ui/icons';

// Types for our analysis data
type TokenRisk = {
  start: number;
  end: number;
  risk_score: number;
  risk_type: string;
  explanation: string;
};

type PolicyMatch = {
  policy_id: number;
  article: string;
  similarity_score: number;
  text_snippet: string;
};

type OverallRisk = {
  score: number;
  categories: Record<string, number>;
};

type AnalysisResponse = {
  token_risks: TokenRisk[];
  policy_matches: PolicyMatch[];
  overall_risk: OverallRisk;
};

type AnalysisOptions = {
  analyzeBias: boolean;
  analyzePII: boolean;
  analyzePolicy: boolean;
};

export default function PromptAnalyzer() {
  // State hooks
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState<AnalysisOptions>({
    analyzeBias: true,
    analyzePII: true,
    analyzePolicy: true,
  });
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [options, setOptions] = useState({
    analyzeBias: true,
    analyzePII: true,
    analyzePolicy: true,
  });
  const wsRef = useRef<WebSocket | null>(null);
  
  // Set up WebSocket connection
  useEffect(() => {
    // Define WebSocket URL with explicit protocol, host, and port
    // We're hardcoding it for testing since we know our backend is on port 8000
    const wsUrl = 'ws://localhost:8000/api/v1/analyze/prompt';
    console.log('Attempting to connect to WebSocket at:', wsUrl);
    
    const connectWs = () => {
      try {
        // Close existing connection if any
        if (wsRef.current) {
          wsRef.current.close();
        }
        
        // Create new connection
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('WebSocket connected successfully');
          setError(null);
        };
        
        ws.onmessage = (event) => {
          try {
            console.log('Received WebSocket message:', event.data);
            const data = JSON.parse(event.data) as AnalysisResponse;
            setAnalysisResults(data);
            setIsAnalyzing(false);
          } catch (e) {
            console.error('Error parsing WebSocket message:', e);
            setError('Failed to parse analysis results');
            setIsAnalyzing(false);
          }
        };
        
        ws.onerror = (e) => {
          console.error('WebSocket error:', e);
          setError('Connection error with WebSocket. We will use fallback HTTP API.');
          setIsAnalyzing(false);
          // Implement HTTP fallback for robustness
          useFallbackHttpApi();
        };
        
        ws.onclose = (e) => {
          console.log('WebSocket disconnected:', e.code, e.reason);
          if (e.code !== 1000) { // 1000 is normal closure
            setError('WebSocket connection closed unexpectedly. Using HTTP fallback.');
          }
        };
        
        wsRef.current = ws;
      } catch (error) {
        console.error('Error setting up WebSocket:', error);
        setError('Failed to create WebSocket connection. Using fallback HTTP API.');
      }
    };
    
    // Attempt to connect
    connectWs();
    
    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
      }
    };
  }, []);
  
  // Fallback HTTP API function
  const useFallbackHttpApi = () => {
    console.log('Using fallback HTTP API');
    // This would normally make a fetch request to the REST endpoint
    // For now, we'll just use mock data
  };
  
  // Analysis function
  const analyzePrompt = () => {
    if (!prompt.trim()) {
      return;
    }
    
    setIsAnalyzing(true);
    setError(null);
    console.log('Analyzing prompt:', prompt.substring(0, 50) + '...');
    
    // Use HTTP endpoint instead of WebSocket for now
    console.log('Using HTTP endpoint for analysis');
    
    // Prepare the request payload
    const payload = {
      text: prompt,
      options: {
        analyze_bias: options.analyzeBias,
        analyze_pii: options.analyzePII,
        analyze_policy: options.analyzePolicy,
      },
    };
    
    // Make a fetch request to the HTTP endpoint
    fetch('http://localhost:8000/api/v1/analyze/prompt', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Analysis response:', data);
        setAnalysisResults(data);
        setIsAnalyzing(false);
      })
      .catch(error => {
        console.error('Error with HTTP analysis:', error);
        setError(`HTTP request failed: ${error.message}`);
        setIsAnalyzing(false);
        
        // Fallback to mock data if HTTP fails
        console.log('Using mock data as final fallback');
        setTimeout(() => {
          const mockResponse: AnalysisResponse = {
            token_risks: [
              {
                start: 10,
                end: 15,
                risk_score: 0.8,
                risk_type: 'bias',
                explanation: 'Gender bias detected',
              },
            ],
            policy_matches: [
              {
                policy_id: 123,
                article: 'Article 10',
                similarity_score: 0.85,
                text_snippet: 'Data quality and transparency requirements...',
              },
            ],
            overall_risk: {
              score: 0.75,
              categories: {
                bias: 0.8,
                policy_violation: 0.7,
              },
            },
          };
          
          setAnalysisResults(mockResponse);
          setIsAnalyzing(false);
      });
  };

  return (
    <div className="w-full max-w-full">
      <div className="grid gap-8 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <h2 className="mb-4 text-xl font-semibold">Enter Prompt</h2>

          <div className="mb-4 space-y-4">
            <label className="block text-sm font-medium">Analysis Options</label>
            <div className="flex flex-wrap gap-4">
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={options.analyzeBias}
                  onChange={e => setOptions({ ...options, analyzeBias: e.target.checked })}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <span>Bias</span>
              </label>
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={options.analyzePII}
                  onChange={e => setOptions({ ...options, analyzePII: e.target.checked })}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <span>PII</span>
              </label>
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={options.analyzePolicy}
                  onChange={e => setOptions({ ...options, analyzePolicy: e.target.checked })}
                  className="h-4 w-4 rounded border-gray-300"
                />
                <span>Policy</span>
              </label>
            </div>
          </div>

          <div className="relative mb-4">
            {error && (
              <div className="mb-4 flex items-center gap-2 rounded-md bg-red-50 p-2 text-sm text-red-800 dark:bg-red-900/30 dark:text-red-200">
                <AlertCircleIcon className="h-4 w-4" />
                {error}
              </div>
            )}

            <div className="relative">
              {analysisResults && !isAnalyzing ? (
                <RiskHighlighter
                  text={prompt}
                  risks={analysisResults.token_risks}
                  onTextChange={newText => setPrompt(newText)}
                />
              ) : (
                <textarea
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  onPaste={e => {
                    const pastedText = e.clipboardData.getData('text');
                    setPrompt(prompt + pastedText);
                    e.preventDefault(); // Prevent default to handle paste manually
                  }}
                  placeholder="Enter your prompt here to analyze for compliance with EU AI Act..."
                  style={{ color: 'black', backgroundColor: 'white' }}
                  className="h-64 w-full rounded-md border border-gray-300 p-4 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:bg-gray-800"
                />
              )}
            </div>

            <div className="mt-4">
              <button
                onClick={analyzePrompt}
                disabled={isAnalyzing || !prompt.trim()}
                className="flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 disabled:bg-blue-400"
              >
                {isAnalyzing ? (
                  <>
                    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        fill="none"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <ShieldIcon className="h-4 w-4" />
                    Analyze Prompt
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2">
          <h2 className="mb-4 text-xl font-semibold">Analysis Results</h2>

          {analysisResults ? (
            <AnalysisResults results={analysisResults} />
          ) : (
            <div className="rounded-md border border-gray-300 p-6 text-center text-gray-500 dark:border-gray-700 dark:text-gray-400">
              <p>Enter a prompt and click "Analyze Prompt" to see compliance analysis results.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PromptAnalyzer;
