'use client';

import { useState } from 'react';

interface PdfUploaderProps {
  onUploadComplete: (url: string, analysis: string) => void;
}

export default function PdfUploader({ onUploadComplete }: PdfUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload file');
      }

      const data = await response.json();
      
      // Create a URL for the selected file
      const fileUrl = URL.createObjectURL(selectedFile);
      
      // Pass both the file URL and analysis to the parent component
      onUploadComplete(fileUrl, data.data.analysis);
    } catch (error) {
      console.error('Error uploading file:', error);
      setError(error instanceof Error ? error.message : 'Failed to upload file');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
      <div className="mb-4">
        <label className="block text-gray-200 text-sm font-semibold mb-2" htmlFor="pdf-upload">
          Select PDF File
        </label>
        <div className="relative">
          <input
            id="pdf-upload"
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-200
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-md file:border-0
                     file:text-sm file:font-semibold
                     file:bg-blue-500 file:text-white
                     hover:file:bg-blue-600
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        {error && (
          <p className="mt-2 text-red-500 text-sm">{error}</p>
        )}
      </div>
      <button
        onClick={handleUpload}
        disabled={!selectedFile || isUploading}
        className={`w-full py-2 px-4 rounded-lg text-white font-semibold transition-colors duration-200 ${
          selectedFile && !isUploading
            ? 'bg-blue-500 hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800'
            : 'bg-gray-600 cursor-not-allowed'
        }`}
      >
        {isUploading ? 'Uploading...' : 'Upload'}
      </button>
    </div>
  );
} 