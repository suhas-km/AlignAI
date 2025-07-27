'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Loader2, Shield, User, Scale } from 'lucide-react';
import { Progress } from './progress';

interface AnalysisStep {
  id: string;
  label: string;
  icon: React.ReactNode;
  estimatedDuration: number; // in milliseconds
}

interface AnalysisProgressProps {
  isAnalyzing: boolean;
  analysisOptions: {
    analyze_bias?: boolean;
    analyze_pii?: boolean;
    analyze_policy?: boolean;
  };
}

export function AnalysisProgress({ isAnalyzing, analysisOptions }: AnalysisProgressProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(0);

  // Define analysis steps based on enabled options
  const allSteps: AnalysisStep[] = [
    {
      id: 'bias',
      label: 'Bias Detection',
      icon: <User className="h-4 w-4" />,
      estimatedDuration: 2000,
    },
    {
      id: 'pii',
      label: 'PII Scanning',
      icon: <Shield className="h-4 w-4" />,
      estimatedDuration: 1500,
    },
    {
      id: 'policy',
      label: 'Policy Compliance',
      icon: <Scale className="h-4 w-4" />,
      estimatedDuration: 2500,
    },
  ];

  // Filter steps based on analysis options
  const activeSteps = allSteps.filter(step => {
    if (step.id === 'bias') return analysisOptions.analyze_bias ?? false;
    if (step.id === 'pii') return analysisOptions.analyze_pii ?? false;
    if (step.id === 'policy') return analysisOptions.analyze_policy ?? false;
    return false;
  });

  const totalDuration = activeSteps.reduce((sum, step) => sum + step.estimatedDuration, 0);

  useEffect(() => {
    if (!isAnalyzing) {
      setCurrentStep(0);
      setProgress(0);
      setTimeRemaining(0);
      return;
    }

    let startTime = Date.now();
    let stepStartTime = startTime;
    let currentStepIndex = 0;

    const interval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const stepElapsed = Date.now() - stepStartTime;
      
      // Calculate overall progress
      const overallProgress = Math.min((elapsed / totalDuration) * 100, 95); // Cap at 95% until complete
      setProgress(overallProgress);
      
      // Calculate time remaining
      const remaining = Math.max(0, totalDuration - elapsed);
      setTimeRemaining(Math.ceil(remaining / 1000));
      
      // Move to next step if current step duration has passed
      if (currentStepIndex < activeSteps.length - 1 && 
          stepElapsed >= activeSteps[currentStepIndex].estimatedDuration) {
        currentStepIndex++;
        setCurrentStep(currentStepIndex);
        stepStartTime = Date.now();
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isAnalyzing, totalDuration, activeSteps]);

  if (!isAnalyzing) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="space-y-6 rounded-lg border bg-card p-6"
    >
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Analyzing Content</h3>
          <div className="text-sm text-muted-foreground">
            {timeRemaining > 0 && `~${timeRemaining}s remaining`}
          </div>
        </div>
        <Progress value={progress} className="h-2" />
      </div>

      <div className="space-y-3">
        {activeSteps.map((step, index) => {
          const isCompleted = index < currentStep;
          const isCurrent = index === currentStep;
          const isPending = index > currentStep;

          return (
            <motion.div
              key={step.id}
              className={`flex items-center space-x-3 rounded-md p-3 transition-colors ${
                isCompleted
                  ? 'bg-green-50 dark:bg-green-950/30'
                  : isCurrent
                  ? 'bg-blue-50 dark:bg-blue-950/30'
                  : 'bg-muted/50'
              }`}
              initial={{ opacity: 0.5 }}
              animate={{ 
                opacity: isPending ? 0.5 : 1,
                scale: isCurrent ? 1.02 : 1
              }}
              transition={{ duration: 0.2 }}
            >
              <div className={`flex h-8 w-8 items-center justify-center rounded-full ${
                isCompleted
                  ? 'bg-green-500 text-white'
                  : isCurrent
                  ? 'bg-blue-500 text-white'
                  : 'bg-muted text-muted-foreground'
              }`}>
                {isCompleted ? (
                  <CheckCircle className="h-4 w-4" />
                ) : isCurrent ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  step.icon
                )}
              </div>
              <div className="flex-1">
                <p className={`font-medium ${
                  isCompleted
                    ? 'text-green-700 dark:text-green-300'
                    : isCurrent
                    ? 'text-blue-700 dark:text-blue-300'
                    : 'text-muted-foreground'
                }`}>
                  {step.label}
                </p>
                <p className="text-xs text-muted-foreground">
                  {isCompleted
                    ? 'Complete'
                    : isCurrent
                    ? 'In progress...'
                    : 'Pending'
                  }
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Processing your content for compliance and safety...</span>
      </div>
    </motion.div>
  );
}
