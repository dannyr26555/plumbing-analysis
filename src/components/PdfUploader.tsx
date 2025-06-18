'use client';

import { useState, useEffect, useRef } from 'react';

interface PdfUploaderProps {
  onUploadSuccess: (url: string, analysis: string) => void;
}

interface ProgressState {
  current: number;
  total: number;
  stage: 'uploading' | 'processing' | 'analyzing' | 'complete';
  message: string;
}

export default function PdfUploader({ onUploadSuccess }: PdfUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

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
    setProgress(null);
  };

  const startProgressStream = (taskId: string, onComplete: () => void) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const eventSource = new EventSource(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/progress/${taskId}/stream`);
    eventSourceRef.current = eventSource;
    eventSource.onmessage = (event) => {
      try {
        const progressData = JSON.parse(event.data);
        setProgress({
          current: progressData.current,
          total: progressData.total,
          stage: progressData.stage,
          message: progressData.message
        });
        if (progressData.stage === 'complete') {
          eventSource.close();
          eventSourceRef.current = null;
          onComplete();
        }
      } catch (error) {
        console.error('Error parsing progress data:', error);
      }
    };
    eventSource.onerror = (error) => {
      console.error('EventSource error:', error);
      eventSource.close();
      eventSourceRef.current = null;
    };
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }
    setIsUploading(true);
    setError(null);
    setProgress(null);
    // Step 1: Upload file and get taskId/filename
    const formData = new FormData();
    formData.append('file', selectedFile);
    let taskId = '';
    let filename = '';
    try {
      const uploadRes = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!uploadRes.ok) throw new Error('Failed to upload PDF');
      const uploadData = await uploadRes.json();
      taskId = uploadData.taskId;
      filename = uploadData.filename;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during upload');
      setIsUploading(false);
      return;
    }
    // Step 2: Start progress stream
    startProgressStream(taskId, async () => {
      // Step 4: When complete, fetch result from /api/result/{taskId}
      try {
        const resultRes = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/result/${taskId}`);
        if (!resultRes.ok) throw new Error('Failed to fetch analysis result');
        const resultData = await resultRes.json();
        onUploadSuccess(resultData.pdfUrl, resultData.analysis);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred during analysis');
      } finally {
        setIsUploading(false);
      }
    });
    // Step 3: Trigger analysis (in background)
    fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: new URLSearchParams({ taskId, filename }),
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
  };

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const getProgressColor = (stage: string) => {
    switch (stage) {
      case 'uploading':
        return 'bg-blue-500';
      case 'processing':
        return 'bg-yellow-500';
      case 'analyzing':
        return 'bg-green-500';
      case 'complete':
        return 'bg-green-600';
      default:
        return 'bg-blue-500';
    }
  };

  const getStageIcon = (stage: string) => {
    switch (stage) {
      case 'uploading':
        return '📤';
      case 'processing':
        return '⚙️';
      case 'analyzing':
        return '🔍';
      case 'complete':
        return '✅';
      default:
        return '📤';
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
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
        {/* Progress Bar */}
        {progress && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-sm text-gray-300">
              <span className="flex items-center gap-2">
                <span>{getStageIcon(progress.stage)}</span>
                <span>{progress.message}</span>
              </span>
              <span>{Math.round((progress.current / progress.total) * 100)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ease-out ${getProgressColor(progress.stage)}`}
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          className="w-full px-4 py-3 bg-blue-500 text-white font-medium rounded-lg hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-50 transition-colors duration-200"
        >
          {isUploading ? 'Processing...' : 'Upload'}
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