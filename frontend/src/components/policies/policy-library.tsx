'use client';

import { useState, useEffect } from 'react';
import { SearchIcon } from '@/components/ui/icons';

// Mock data for development
const mockPolicies = [
  {
    id: 1,
    act_title: 'EU AI Act',
    document_type: 'Regulation',
    chapter_identifier: 'Chapter 2',
    chapter_title: 'High-Risk AI Systems',
    article_number: 'Article 10',
    section_identifier: 'Section 1',
    paragraph_identifier: '1',
    policy_text: 'High-risk AI systems shall be designed and developed in such a way that they achieve, in the light of their intended purpose, an appropriate level of accuracy, robustness and cybersecurity, and perform consistently in those respects throughout their lifecycle.',
    category: ['Quality', 'Safety'],
    publication_date: '2023-12-08',
    version: '1.0',
  },
  {
    id: 2,
    act_title: 'EU AI Act',
    document_type: 'Regulation',
    chapter_identifier: 'Chapter 2',
    chapter_title: 'High-Risk AI Systems',
    article_number: 'Article 10',
    section_identifier: 'Section 2',
    paragraph_identifier: '2',
    policy_text: 'High-risk AI systems that continue to learn after being placed on the market or put into service shall be developed in such a way to ensure that possibly biased outputs due to outputs used as an input for future operations ("feedback loops") are duly addressed with appropriate mitigation measures.',
    category: ['Bias', 'Quality'],
    publication_date: '2023-12-08',
    version: '1.0',
  },
  {
    id: 3,
    act_title: 'EU AI Act',
    document_type: 'Regulation',
    chapter_identifier: 'Chapter 3',
    chapter_title: 'Transparency and User Information',
    article_number: 'Article 13',
    section_identifier: 'Section 1',
    paragraph_identifier: '1',
    policy_text: 'High-risk AI systems shall be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable users to interpret the system's output and use it appropriately.',
    category: ['Transparency'],
    publication_date: '2023-12-08',
    version: '1.0',
  },
  {
    id: 4,
    act_title: 'EU AI Act',
    document_type: 'Regulation',
    chapter_identifier: 'Chapter 3',
    chapter_title: 'Transparency and User Information',
    article_number: 'Article 14',
    section_identifier: 'Section 1',
    paragraph_identifier: '1',
    policy_text: 'Providers shall ensure that high-risk AI systems are accompanied by instructions for use that include concise, complete, correct and clear information that is relevant, accessible and comprehensible to users.',
    category: ['Transparency', 'Documentation'],
    publication_date: '2023-12-08',
    version: '1.0',
  },
  {
    id: 5,
    act_title: 'EU AI Act',
    document_type: 'Regulation',
    chapter_identifier: 'Chapter 4',
    chapter_title: 'Prohibited AI Practices',
    article_number: 'Article 5',
    section_identifier: 'Section 1',
    paragraph_identifier: '1',
    point_identifier: 'a',
    policy_text: 'The placing on the market, putting into service or use of an AI system that deploys subliminal techniques beyond a person's consciousness in order to materially distort a person's behaviour in a manner that causes or is likely to cause that person or another person physical or psychological harm is prohibited.',
    category: ['Prohibited Practice'],
    publication_date: '2023-12-08',
    version: '1.0',
  },
];

// Available categories for filtering
const allCategories = [
  'Bias', 
  'Quality', 
  'Transparency', 
  'Safety', 
  'Documentation', 
  'Prohibited Practice', 
  'PII'
];

export default function PolicyLibrary() {
  const [policies, setPolicies] = useState(mockPolicies);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredPolicies, setFilteredPolicies] = useState(mockPolicies);
  const [selectedArticle, setSelectedArticle] = useState<number | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  
  // Fetch policies from API (would be implemented in a real app)
  useEffect(() => {
    // Example API call (commented out for now)
    // const fetchPolicies = async () => {
    //   try {
    //     const response = await fetch('/api/v1/policies');
    //     if (response.ok) {
    //       const data = await response.json();
    //       setPolicies(data);
    //       setFilteredPolicies(data);
    //     }
    //   } catch (error) {
    //     console.error('Error fetching policies:', error);
    //   }
    // };
    // 
    // fetchPolicies();
  }, []);
  
  // Filter policies based on search query and selected categories
  useEffect(() => {
    let results = policies;
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      results = results.filter(
        (policy) => 
          policy.policy_text.toLowerCase().includes(query) ||
          policy.article_number.toLowerCase().includes(query) ||
          policy.chapter_title.toLowerCase().includes(query)
      );
    }
    
    // Filter by selected categories
    if (selectedCategories.length > 0) {
      results = results.filter((policy) => 
        policy.category.some(cat => selectedCategories.includes(cat))
      );
    }
    
    setFilteredPolicies(results);
  }, [searchQuery, selectedCategories, policies]);
  
  // Toggle a category selection
  const toggleCategory = (category: string) => {
    setSelectedCategories((prev) => 
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    );
  };
  
  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };
  
  // Get unique articles from filtered policies for article filter
  const uniqueArticles = Array.from(
    new Set(filteredPolicies.map((policy) => policy.article_number))
  );

  return (
    <div className="grid gap-6 lg:grid-cols-4">
      {/* Sidebar Filters */}
      <div className="lg:col-span-1">
        <div className="rounded-lg border border-gray-200 p-4 dark:border-gray-800">
          <h3 className="mb-4 text-lg font-semibold">Filters</h3>
          
          {/* Search */}
          <div className="mb-6">
            <label htmlFor="search" className="mb-2 block text-sm font-medium">
              Search
            </label>
            <div className="relative">
              <input
                type="text"
                id="search"
                className="w-full rounded-md border border-gray-300 p-2 pl-8 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-700 dark:bg-gray-800"
                placeholder="Search policies..."
                value={searchQuery}
                onChange={handleSearchChange}
              />
              <SearchIcon className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
            </div>
          </div>
          
          {/* Categories */}
          <div className="mb-6">
            <h4 className="mb-2 text-sm font-medium">Categories</h4>
            <div className="space-y-2">
              {allCategories.map((category) => (
                <label key={category} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedCategories.includes(category)}
                    onChange={() => toggleCategory(category)}
                    className="h-4 w-4 rounded border-gray-300"
                  />
                  <span className="text-sm">{category}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </div>
      
      {/* Policy List */}
      <div className="lg:col-span-3">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            {filteredPolicies.length} {filteredPolicies.length === 1 ? 'Policy' : 'Policies'}
          </h3>
        </div>
        
        <div className="space-y-4">
          {filteredPolicies.length > 0 ? (
            filteredPolicies.map((policy) => (
              <div
                key={policy.id}
                className={`rounded-lg border border-gray-200 p-4 transition hover:bg-gray-50 dark:border-gray-800 dark:hover:bg-gray-900 ${
                  selectedArticle === policy.id ? 'ring-2 ring-blue-500' : ''
                }`}
                onClick={() => setSelectedArticle(policy.id)}
              >
                <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <h4 className="font-medium">
                      {policy.article_number} - {policy.chapter_title}
                    </h4>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {policy.act_title} | {policy.document_type} | {policy.publication_date}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {policy.category.map((cat) => (
                      <span
                        key={cat}
                        className="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                      >
                        {cat}
                      </span>
                    ))}
                  </div>
                </div>
                <p className="text-sm">{policy.policy_text}</p>
              </div>
            ))
          ) : (
            <div className="rounded-lg border border-gray-200 p-6 text-center dark:border-gray-800">
              <p className="text-gray-500 dark:text-gray-400">No policies found matching your criteria.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
