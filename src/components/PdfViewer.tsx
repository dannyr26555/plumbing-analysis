'use client';

import { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

interface PdfViewerProps {
  url: string;
}

export default function PdfViewer({ url }: PdfViewerProps) {
  return (
    <div className="w-[35%] sticky top-8 z-10">
      <div className="p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
        <div className="flex flex-col items-center">
          <iframe
            src={url}
            className="w-full h-[calc(100vh-12rem)] border-0 rounded-lg"
            title="PDF Viewer"
          />
        </div>
      </div>
    </div>
  );
} 