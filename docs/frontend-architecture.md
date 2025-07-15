# AlignAI Frontend Architecture

## Overview

The AlignAI frontend is built with Next.js using the App Router architecture for optimal performance and developer experience. The frontend serves as the interface for users to interact with the compliance analysis engine, visualize risks, and manage compliance efforts.

## Technology Stack

- **Framework**: Next.js 14+ with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS with additional component libraries
- **State Management**: React Context for global state, Zustand for complex state
- **Data Fetching**: React Query for data management and caching
- **Real-time Communication**: WebSockets for live analysis updates

## Key Components

### 1. Prompt Analyzer Component

The core interactive component that allows users to input prompts and visualize compliance risks in real-time.

```tsx
// Simplified structure of the Prompt Analyzer Component
interface TokenRisk {
  start: number;
  end: number;
  riskScore: number;
  riskType: 'bias' | 'pii' | 'policy_violation';
  explanation: string;
}

const PromptAnalyzer: React.FC = () => {
  const [text, setText] = useState('');
  const [tokenRisks, setTokenRisks] = useState<TokenRisk[]>([]);
  const [overallRisk, setOverallRisk] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const socket = useWebSocket('/api/analyze/prompt');
  
  // Send text through WebSocket as user types or on submission
  const analyzePrompt = (text: string) => {
    setIsAnalyzing(true);
    socket.send(JSON.stringify({ text }));
  };
  
  // Listen for WebSocket responses
  useEffect(() => {
    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setTokenRisks(data.token_risks);
      setOverallRisk(data.overall_risk.score);
      setIsAnalyzing(false);
    };
  }, [socket]);
  
  return (
    <div className="prompt-analyzer">
      <div className="toolbar">{
        /* Analysis controls, options, etc. */
      }</div>
      <div className="editor-container">
        <RichTextEditor 
          text={text} 
          onTextChange={(newText) => {
            setText(newText);
            analyzePrompt(newText);
          }}
          tokenRisks={tokenRisks}
        />
      </div>
      <div className="risk-summary">
        {
          /* Risk summary panel showing overall score, 
             policy violations, and recommendations */
        }
      </div>
    </div>
  );
};
```

### 2. Heat Map Text Editor

Provides real-time visual feedback on compliance risks at the token level.

```tsx
interface RichTextEditorProps {
  text: string;
  onTextChange: (text: string) => void;
  tokenRisks: TokenRisk[];
}

const RichTextEditor: React.FC<RichTextEditorProps> = ({ text, onTextChange, tokenRisks }) => {
  const editor = useSlate();
  
  // Apply decorations based on tokenRisks
  const decorate = useCallback(
    (node) => {
      const ranges: Range[] = [];
      if (!Text.isText(node)) return ranges;
      
      // Map token risks to Slate ranges with styling
      tokenRisks.forEach((risk) => {
        ranges.push({
          anchor: { path: [], offset: risk.start },
          focus: { path: [], offset: risk.end },
          riskScore: risk.riskScore,
          riskType: risk.riskType,
          explanation: risk.explanation,
        });
      });
      
      return ranges;
    },
    [tokenRisks]
  );
  
  // Render leaf with appropriate styling based on risk score/type
  const renderLeaf = useCallback(({ attributes, children, leaf }) => {
    let style = {};
    
    if (leaf.riskScore) {
      const intensity = Math.min(Math.floor(leaf.riskScore * 100), 100);
      
      // Different colors for different risk types
      const colorMap = {
        bias: `rgba(255, 0, 0, ${intensity/100 * 0.5})`,
        pii: `rgba(255, 165, 0, ${intensity/100 * 0.5})`,
        policy_violation: `rgba(0, 0, 255, ${intensity/100 * 0.5})`
      };
      
      style = {
        backgroundColor: colorMap[leaf.riskType] || 'rgba(255, 0, 0, 0.2)',
        padding: '0 1px',
        borderBottom: '2px solid currentColor',
      };
    }
    
    return (
      <span 
        {...attributes}
        style={style}
        data-tooltip={leaf.explanation}
      >
        {children}
      </span>
    );
  }, []);
  
  return (
    <Slate
      editor={editor}
      value={text}
      onChange={(value) => onTextChange(value)}
    >
      <Editable
        decorate={decorate}
        renderLeaf={renderLeaf}
        placeholder="Enter your prompt here..."
      />
    </Slate>
  );
};
```

### 3. Risk Summary Panel

Displays consolidated information about detected risks and provides recommendations.

```tsx
interface RiskSummaryProps {
  overallRisk: number;
  tokenRisks: TokenRisk[];
  policyMatches: PolicyMatch[];
}

const RiskSummary: React.FC<RiskSummaryProps> = ({ overallRisk, tokenRisks, policyMatches }) => {
  const riskLevel = overallRisk < 0.3 ? 'Low' : overallRisk < 0.7 ? 'Medium' : 'High';
  
  return (
    <div className="risk-summary">
      <div className="overall-score">
        <h3>Overall Risk: {riskLevel}</h3>
        <div className="risk-meter">
          <div 
            className="risk-fill"
            style={{ width: `${overallRisk * 100}%` }}
          />
        </div>
        <div className="score-value">{Math.round(overallRisk * 100)}%</div>
      </div>
      
      <div className="risk-categories">
        <h4>Risk Breakdown</h4>
        <ul>
          {Object.entries(groupBy(tokenRisks, 'riskType')).map(([type, risks]) => (
            <li key={type}>
              <span className="risk-type">{formatRiskType(type)}</span>
              <span className="risk-count">{risks.length}</span>
            </li>
          ))}
        </ul>
      </div>
      
      <div className="policy-references">
        <h4>Policy References</h4>
        <ul>
          {policyMatches.map((policy) => (
            <li key={policy.id}>
              <Link href={`/policies/${policy.id}`}>
                {policy.article}: {truncate(policy.text_snippet, 100)}
              </Link>
              <span className="similarity">{Math.round(policy.similarity_score * 100)}%</span>
            </li>
          ))}
        </ul>
      </div>
      
      <div className="recommendations">
        <h4>Recommendations</h4>
        <ul>
          {generateRecommendations(tokenRisks, policyMatches).map((rec, i) => (
            <li key={i}>{rec}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};
```

## Page Structure

Using Next.js App Router architecture, the site is organized into the following routes:

### 1. Home Page (`/app/page.tsx`)

Landing page with product introduction and call to action.

### 2. Dashboard (`/app/dashboard/page.tsx`)

Admin dashboard showing compliance analytics and organization-wide metrics.

### 3. Prompt Analyzer (`/app/analyzer/page.tsx`)

The main tool for analyzing and improving prompts in real-time.

### 4. Policy Library (`/app/policies/page.tsx`)

Browsable library of EU AI Act regulations and guidelines.

### 5. Reports (`/app/reports/page.tsx`)

Access and generate compliance reports for audit purposes.

### 6. Sandbox (`/app/sandbox/page.tsx`)

Public demonstration of the analysis capabilities with limited features.

## State Management

State is managed using a combination of local component state, React Context for shared state within feature boundaries, and Zustand for global application state.

```tsx
// Example of Zustand store for global analysis settings
import { create } from 'zustand';

interface AnalysisSettings {
  analyzeBias: boolean;
  analyzePii: boolean;
  analyzePolicy: boolean;
  setAnalyzeBias: (analyze: boolean) => void;
  setAnalyzePii: (analyze: boolean) => void;
  setAnalyzePolicy: (analyze: boolean) => void;
}

export const useAnalysisSettingsStore = create<AnalysisSettings>((set) => ({
  analyzeBias: true,
  analyzePii: true,
  analyzePolicy: true,
  setAnalyzeBias: (analyzeBias) => set({ analyzeBias }),
  setAnalyzePii: (analyzePii) => set({ analyzePii }),
  setAnalyzePolicy: (analyzePolicy) => set({ analyzePolicy }),
}));
```

## Authentication and Authorization

Authentication is handled through Supabase Auth with client-side integration using the Supabase JavaScript client. Protected routes are wrapped with authentication checks.

```tsx
// Example of auth wrapper component
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useSupabaseClient } from '@supabase/auth-helpers-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const supabase = useSupabaseClient();
  const router = useRouter();
  
  useEffect(() => {
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.push('/login');
      }
    };
    
    checkAuth();
  }, [router, supabase.auth]);
  
  return <>{children}</>;
};
```

## Performance Optimizations

1. **React Server Components** for static content and reduced JavaScript payload
2. **Dynamic Imports** for code-splitting and lazy-loading of complex components
3. **Edge Caching** for policy content and static assets
4. **WebSocket Debouncing** to minimize analysis requests during typing
5. **Incremental Static Regeneration** for policy library pages

## Accessibility Considerations

1. Semantic HTML structure for screen readers
2. ARIA attributes for interactive components
3. Keyboard navigation support
4. Color contrast compliance (especially important for the heat map visualization)
5. Focus management for dynamic content

## Responsive Design

The UI is built with a mobile-first approach using Tailwind CSS breakpoints. Key components like the Prompt Analyzer adapt to different screen sizes:

- Mobile: Stacked layout with collapsible panels
- Tablet: Side-by-side editor and risk panel
- Desktop: Full three-panel layout with additional analytics

## Testing Strategy

1. **Unit Tests**: Component-level tests with React Testing Library
2. **Integration Tests**: Page-level tests for key user flows
3. **E2E Tests**: Full application testing with Playwright
4. **Visual Regression Tests**: Using Storybook and Chromatic
5. **Accessibility Tests**: Automated a11y testing with axe-core