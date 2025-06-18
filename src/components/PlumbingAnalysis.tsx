'use client';

interface PlumbingAnalysisProps {
  analysis: string;
}

export default function PlumbingAnalysis({ analysis }: PlumbingAnalysisProps) {
  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
      <h2 className="text-2xl font-semibold text-white mb-4">Analysis Results</h2>
      <div className="text-gray-300 whitespace-pre-wrap">
        {analysis}
      </div>
    </div>
  );
} 