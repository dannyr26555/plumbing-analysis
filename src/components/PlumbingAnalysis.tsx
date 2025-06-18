'use client';

import { useState } from 'react';

interface PlumbingAnalysisProps {
  analysis: string;
}

interface AnalysisSection {
  title: string;
  content: string;
}

export default function PlumbingAnalysis({ analysis }: PlumbingAnalysisProps) {
  const [expandedSections, setExpandedSections] = useState<{ [key: string]: boolean }>({});

  // Parse the analysis text into sections
  const parseAnalysis = (text: string): AnalysisSection[] => {
    const sections: AnalysisSection[] = [];
    const lines = text.split('\n');
    let currentSection: AnalysisSection | null = null;

    lines.forEach(line => {
      if (line.startsWith('### ')) {
        if (currentSection) {
          sections.push(currentSection);
        }
        currentSection = {
          title: line.replace('### ', '').trim(),
          content: ''
        };
      } else if (currentSection) {
        currentSection.content += line + '\n';
      }
    });

    if (currentSection) {
      sections.push(currentSection);
    }

    return sections;
  };

  const toggleSection = (title: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [title]: !prev[title]
    }));
  };

  const renderContent = (section: AnalysisSection) => {
    if (section.title === 'Materials and Quantities') {
      // Parse table content
      const lines = section.content.trim().split('\n');
      const tableRows = lines.filter(line => line.includes('|'));
      
      if (tableRows.length > 0) {
        return (
          <div className="overflow-x-auto">
            <table className="w-full table-fixed border-collapse">
              <thead>
                <tr className="bg-gray-900">
                  {tableRows[0].split('|').filter(cell => cell.trim()).map((header, index) => (
                    <th
                      key={index}
                      className={`px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 ${
                        index === 0 ? 'w-2/3' : 'w-1/6'
                      }`}
                    >
                      {header.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {tableRows.slice(2).map((row, rowIndex) => (
                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-gray-800' : 'bg-gray-900'}>
                    {row.split('|').filter(cell => cell.trim()).map((cell, cellIndex) => (
                      <td
                        key={cellIndex}
                        className={`px-4 py-3 text-sm text-white border-b border-gray-600 ${
                          cellIndex === 0 ? 'w-2/3' : 'w-1/6'
                        }`}
                      >
                        {cell.trim()}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
    }
    
    // Default rendering for other sections
    return (
      <div className="text-gray-300 whitespace-pre-wrap prose prose-invert max-w-none">
        {section.content}
      </div>
    );
  };

  const sections = parseAnalysis(analysis);

  return (
    <div className="w-[65%] p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
      <h2 className="text-2xl font-semibold text-white mb-4">Analysis Results</h2>
      <div className="space-y-4">
        {sections.map((section, index) => (
          <div key={index} className="border border-gray-700 rounded-lg overflow-hidden">
            <button
              onClick={() => toggleSection(section.title)}
              className="w-full px-4 py-3 text-left bg-gray-700 hover:bg-gray-600 transition-colors duration-200 flex justify-between items-center"
            >
              <h3 className="text-lg font-medium text-white">{section.title}</h3>
              <svg
                className={`w-5 h-5 text-gray-300 transform transition-transform duration-200 ${
                  expandedSections[section.title] ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {expandedSections[section.title] && (
              <div className="p-4 bg-gray-800">
                {renderContent(section)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 