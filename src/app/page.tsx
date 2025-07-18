'use client';

import { useState } from 'react';
import PdfUploader from '@/components/PdfUploader';
import PdfViewer from '@/components/PdfViewer';
import PlumbingAnalysis from '@/components/PlumbingAnalysis';

export default function Home() {
  const [pdfUrl, setPdfUrl] = useState<string>('');
  const [analysis, setAnalysis] = useState<any>(null); // Now handles both structured and string formats
  const [context, setContext] = useState<string>('');

  const handleUploadSuccess = (url: string, analysisData: any, contextData: string | null) => {
    setPdfUrl(url);
    setAnalysis(analysisData);
    setContext(contextData || ''); // Convert null to empty string
  };

  return (
    <main className="min-h-screen bg-gray-900 py-8">
      <div className="max-w-[95vw] mx-auto px-4">
        {!pdfUrl || !analysis ? (
          <PdfUploader onUploadSuccess={handleUploadSuccess} />
        ) : (
          <div className="flex gap-8 justify-center">
            <PlumbingAnalysis analysis={analysis} context={context} />
            <PdfViewer url={pdfUrl} />
          </div>
        )}
      </div>
    </main>
  );
}
