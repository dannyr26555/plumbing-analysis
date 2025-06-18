'use client';

import { useState } from 'react';

interface PdfUploaderProps {
  onUploadSuccess: (url: string, analysis: string) => void;
}

export default function PdfUploader({ onUploadSuccess }: PdfUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file');
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to analyze PDF');
      }

      const data = await response.json();
      onUploadSuccess(data.pdfUrl, data.analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
      <h2 className="text-2xl font-semibold text-white mb-6">Upload PDF</h2>
      
      <div className="space-y-4">
        {/* File Input */}
        <div>
          <label htmlFor="pdf-file" className="block text-sm font-medium text-gray-300 mb-2">
            Select PDF File
          </label>
          <input
            id="pdf-file"
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            disabled={isUploading}
            className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-500 file:text-white hover:file:bg-blue-600 file:cursor-pointer cursor-pointer bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        {/* Selected File Display */}
        {selectedFile && (
          <div className="p-3 bg-gray-700 rounded-lg border border-gray-600">
            <p className="text-sm text-gray-300">
              <span className="font-medium">Selected file:</span> {selectedFile.name}
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="w-full px-4 py-3 bg-blue-500 text-white font-medium rounded-lg hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-50 transition-colors duration-200"
        >
          {isUploading ? 'Analyzing PDF...' : 'Upload & Analyze'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-900 border border-red-700 rounded-lg">
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
} 