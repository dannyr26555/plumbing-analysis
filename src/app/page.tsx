'use client';

import { useState } from 'react';
import PdfUploader from '@/components/PdfUploader';
import PdfViewer from '@/components/PdfViewer';
import PlumbingAnalysis from '@/components/PlumbingAnalysis';

export default function Home() {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [isUploaded, setIsUploaded] = useState(false);

  const handleUploadComplete = (url: string, analysisContent: string) => {
    setPdfUrl(url);
    setAnalysis(analysisContent);
    setIsUploaded(true);
  };

  return (
    <main className="min-h-screen bg-gray-900 py-12">
      <div className="container mx-auto px-4">
        <h1 className="text-4xl font-bold text-center text-white mb-8">
          Plumbing Analysis
        </h1>
        {!isUploaded ? (
          <PdfUploader onUploadComplete={handleUploadComplete} />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="order-2 lg:order-1">
              {analysis && <PlumbingAnalysis analysis={analysis} />}
            </div>
            <div className="order-1 lg:order-2">
              {pdfUrl && <PdfViewer pdfUrl={pdfUrl} />}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
