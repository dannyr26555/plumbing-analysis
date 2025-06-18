'use client';

interface PdfViewerProps {
  pdfUrl: string;
}

export default function PdfViewer({ pdfUrl }: PdfViewerProps) {
  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700 mb-8">
      <h2 className="text-2xl font-semibold text-white mb-4">PDF Document</h2>
      <div className="w-full h-[600px] bg-gray-700 rounded-lg overflow-hidden">
        <iframe
          src={pdfUrl}
          className="w-full h-full"
          title="PDF Viewer"
        />
      </div>
    </div>
  );
} 