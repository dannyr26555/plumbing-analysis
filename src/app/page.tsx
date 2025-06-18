'use client';

import { useState } from 'react';
import PdfUploader from '@/components/PdfUploader';
import PdfViewer from '@/components/PdfViewer';
import PlumbingAnalysis from '@/components/PlumbingAnalysis';

export default function Home() {
  const [pdfUrl, setPdfUrl] = useState<string>('');
  const [analysis, setAnalysis] = useState<string>('');

  const handleUploadSuccess = (url: string, analysis: string) => {
    setPdfUrl(url);
    setAnalysis(analysis);
  };

  return (
    <main className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {!pdfUrl ? (
          <PdfUploader onUploadSuccess={handleUploadSuccess} />
        ) : (
          <div className="flex gap-6 justify-center">
            <PlumbingAnalysis analysis={analysis} />
            <PdfViewer url={pdfUrl} />
          </div>
        )}
      </div>
    </main>
  );
}
