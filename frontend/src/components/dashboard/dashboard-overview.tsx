'use client';

import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

// Mock data for development (will be replaced with API calls)
const mockTrendData = [
  { date: '2025-05-01', riskScore: 0.45, promptCount: 18 },
  { date: '2025-05-02', riskScore: 0.43, promptCount: 23 },
  { date: '2025-05-03', riskScore: 0.39, promptCount: 15 },
  { date: '2025-05-04', riskScore: 0.41, promptCount: 27 },
  { date: '2025-05-05', riskScore: 0.42, promptCount: 32 },
  { date: '2025-05-06', riskScore: 0.38, promptCount: 24 },
  { date: '2025-05-07', riskScore: 0.36, promptCount: 29 },
];

const mockRiskDistribution = [
  { name: 'Low Risk', value: 774, color: '#4ade80' },
  { name: 'Medium Risk', value: 356, color: '#facc15' },
  { name: 'High Risk', value: 124, color: '#ef4444' },
];

const mockTopRiskCategories = [
  { category: 'PII', count: 356 },
  { category: 'Bias', count: 243 },
  { category: 'Prohibited Practice', count: 87 },
  { category: 'Transparency', count: 65 },
  { category: 'Data Quality', count: 42 },
];

export default function DashboardOverview() {
  const [dashboardData, setDashboardData] = useState({
    totalPromptsAnalyzed: 1254,
    averageRiskScore: 0.42,
    riskTrend: mockTrendData,
    riskDistribution: mockRiskDistribution,
    topRiskCategories: mockTopRiskCategories,
  });

  // In a real app, we would fetch data from the API
  useEffect(() => {
    // Example API call (commented out for now)
    // const fetchDashboardData = async () => {
    //   try {
    //     const response = await fetch('/api/v1/reports/dashboard/summary');
    //     if (response.ok) {
    //       const data = await response.json();
    //       setDashboardData(data);
    //     }
    //   } catch (error) {
    //     console.error('Error fetching dashboard data:', error);
    //   }
    // };
    // 
    // fetchDashboardData();
  }, []);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

  return (
    <div className="space-y-8">
      {/* Summary Cards */}
      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <SummaryCard 
          title="Total Prompts Analyzed" 
          value={dashboardData.totalPromptsAnalyzed.toString()} 
          trend="+12% from last week" 
          trendUp={true} 
        />
        <SummaryCard 
          title="Average Risk Score" 
          value={formatPercent(dashboardData.averageRiskScore)}
          trend="-3% from last week" 
          trendUp={false} 
        />
        <SummaryCard 
          title="Compliance Rate" 
          value="86%" 
          trend="+2% from last week" 
          trendUp={true} 
        />
      </div>

      {/* Charts */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Risk Score Trend */}
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-4 text-lg font-semibold">Risk Score Trend</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={dashboardData.riskTrend}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={formatDate} />
                <YAxis tickFormatter={formatPercent} domain={[0, 1]} />
                <Tooltip 
                  formatter={(value: number) => [formatPercent(value), 'Risk Score']}
                  labelFormatter={(label) => formatDate(label)}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="riskScore"
                  stroke="#8884d8"
                  activeDot={{ r: 8 }}
                  name="Risk Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Distribution */}
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-4 text-lg font-semibold">Risk Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={dashboardData.riskDistribution}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {dashboardData.riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [value, 'Count']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Prompt Volume */}
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-4 text-lg font-semibold">Daily Prompt Volume</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={dashboardData.riskTrend}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tickFormatter={formatDate} />
                <YAxis />
                <Tooltip labelFormatter={(label) => formatDate(label)} />
                <Legend />
                <Bar dataKey="promptCount" fill="#4ade80" name="Prompts Analyzed" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Risk Categories */}
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-4 text-lg font-semibold">Top Risk Categories</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={dashboardData.topRiskCategories}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="category" />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#8884d8" name="Occurrences" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ 
  title, 
  value, 
  trend, 
  trendUp 
}: { 
  title: string; 
  value: string; 
  trend: string; 
  trendUp: boolean; 
}) {
  return (
    <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
      <h3 className="mb-1 text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h3>
      <div className="flex items-baseline">
        <span className="text-3xl font-bold">{value}</span>
        <span className={`ml-2 text-sm ${trendUp ? 'text-green-600' : 'text-red-600'}`}>
          {trend}
        </span>
      </div>
    </div>
  );
}
