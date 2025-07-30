'use client';

import { useState } from 'react';
import * as XLSX from 'xlsx';

interface MaterialItem {
  item_name: string;
  quantity: number;
  unit: string;
  reference_sheet?: string;
  zone?: string;
  size?: string;
  specification?: string;
  confidence?: number;
  notes?: string;
}

interface ContextResult {
  sheet_metadata: {
    sheet_id?: string;
    discipline?: string;
    floor?: string;
    title?: string;
  };
  legend: Array<{
    symbol: string;
    description: string;
  }>;
  drawing_types: string[];
  technical_standards: string[];
}

interface PlumbingResult {
  materials: MaterialItem[];
  special_requirements: string[];
  potential_issues: string[];
  summary?: string;
}

interface ProcessingError {
  agent: string;
  sheet_id?: string;
  error_type: string;
  error_message: string;
  timestamp: string;
}

interface AnalysisData {
  task_id: string;
  sheets_processed: string[];
  context_results: ContextResult[];
  plumbing_results: PlumbingResult[];
  consolidated_materials: MaterialItem[];
  processing_status: {
    stage: string;
    progress: number;
    message: string;
    errors: ProcessingError[];
    total_errors: number;
    agent_error_summary: { [key: string]: number };
    sheet_error_summary: { [key: string]: number };
  };
  metadata: {
    filename: string;
    total_pages: number;
    workflow: string;
  };
}

interface PlumbingAnalysisProps {
  analysis: AnalysisData | string; // Support both old and new formats
  context: any; // Legacy prop, may not be used anymore
}

interface AnalysisSection {
  title: string;
  content: string;
}

export default function PlumbingAnalysis({ analysis, context }: PlumbingAnalysisProps) {
  const [expandedSections, setExpandedSections] = useState<{ [key: string]: boolean }>({
    'Materials Summary': true, // Expand materials by default
  });
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [exportFormat, setExportFormat] = useState<'csv' | 'xlsx' | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportSuccess, setExportSuccess] = useState(false);

  // Check if analysis is structured data or legacy string
  const isStructuredData = typeof analysis === 'object' && analysis !== null;
  const analysisData = isStructuredData ? analysis as AnalysisData : null;

  // Legacy parsing for old string format
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

  const handleSort = (columnKey: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig && sortConfig.key === columnKey && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key: columnKey, direction });
  };

  // Export functions with save dialog
  const exportToCSV = async (materials: MaterialItem[], defaultFilename: string = 'consolidated_materials.csv') => {
    const headers = ['Item Name', 'Quantity', 'Unit', 'Size', 'Reference Sheet', 'Zone', 'Specification', 'Confidence', 'Notes'];
    const csvContent = [
      headers.join(','),
      ...materials.map(material => [
        `"${material.item_name || ''}"`,
        material.quantity || '',
        `"${material.unit || ''}"`,
        `"${material.size || ''}"`,
        `"${material.reference_sheet || ''}"`,
        `"${material.zone || ''}"`,
        `"${material.specification || ''}"`,
        material.confidence ? (material.confidence * 100).toFixed(0) + '%' : '',
        `"${material.notes || ''}"`
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });

    // Try to use the modern File System Access API
    if ('showSaveFilePicker' in window) {
      try {
        const fileHandle = await (window as any).showSaveFilePicker({
          suggestedName: defaultFilename,
          types: [
            {
              description: 'CSV files',
              accept: {
                'text/csv': ['.csv'],
              },
            },
          ],
        });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        return;
      } catch (err) {
        // User cancelled or API not supported, fall back to download
        if ((err as Error).name !== 'AbortError') {
          console.warn('Save dialog not available:', err);
        }
      }
    }

    // Fallback: traditional download with suggested filename
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', defaultFilename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const exportToXLSX = async (materials: MaterialItem[], defaultFilename: string = 'consolidated_materials.xlsx') => {
    const worksheet = XLSX.utils.json_to_sheet(
      materials.map(material => ({
        'Item Name': material.item_name || '',
        'Quantity': material.quantity || '',
        'Unit': material.unit || '',
        'Size': material.size || '',
        'Reference Sheet': material.reference_sheet || '',
        'Zone': material.zone || '',
        'Specification': material.specification || '',
        'Confidence': material.confidence ? (material.confidence * 100).toFixed(0) + '%' : '',
        'Notes': material.notes || ''
      }))
    );

    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'Consolidated Materials');
    
    // Try to use the modern File System Access API
    if ('showSaveFilePicker' in window) {
      try {
        const fileHandle = await (window as any).showSaveFilePicker({
          suggestedName: defaultFilename,
          types: [
            {
              description: 'Excel files',
              accept: {
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
              },
            },
          ],
        });
        
        // Generate the Excel file as a blob
        const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
        const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
        
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        return;
      } catch (err) {
        // User cancelled or API not supported, fall back to download
        if ((err as Error).name !== 'AbortError') {
          console.warn('Save dialog not available:', err);
        }
      }
    }

    // Fallback: use XLSX's built-in download with suggested filename
    XLSX.writeFile(workbook, defaultFilename);
  };

  const handleExport = async () => {
    if (!exportFormat) {
      alert('Please select an export format (CSV or XLSX)');
      return;
    }

    if (!isStructuredData || !analysisData) {
      alert('No data available for export');
      return;
    }

    setIsExporting(true);
    setExportSuccess(false);
    const filename = `consolidated_materials_${new Date().toISOString().split('T')[0]}`;
    
    try {
      if (exportFormat === 'csv') {
        await exportToCSV(analysisData.consolidated_materials, `${filename}.csv`);
      } else if (exportFormat === 'xlsx') {
        await exportToXLSX(analysisData.consolidated_materials, `${filename}.xlsx`);
      }
      
      // Show success message
      setExportSuccess(true);
      
      // Hide success message after 2 seconds
      setTimeout(() => {
        setExportSuccess(false);
      }, 2000);
      
    } catch (error) {
      console.error('Export failed:', error);
      alert('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  const sortMaterials = (materials: MaterialItem[], key: string, direction: 'asc' | 'desc') => {
    return [...materials].sort((a, b) => {
      let aVal = (a as any)[key] || '';
      let bVal = (b as any)[key] || '';
      
      // Handle numbers
      if (key === 'quantity' || key === 'confidence') {
        aVal = Number(aVal) || 0;
        bVal = Number(bVal) || 0;
        return direction === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      // Handle strings
      aVal = String(aVal).toLowerCase();
      bVal = String(bVal).toLowerCase();
      
      if (direction === 'asc') {
        return aVal.localeCompare(bVal);
      } else {
        return bVal.localeCompare(aVal);
      }
    });
  };

  const getSortIcon = (columnKey: string) => {
    if (sortConfig?.key === columnKey) {
      return sortConfig.direction === 'asc' ? '↑' : '↓';
    }
    return '↕';
  };

  const renderMaterialsTable = (materials: MaterialItem[], title: string) => {
    if (!materials || materials.length === 0) {
      return (
        <div className="text-gray-400 text-center py-4">
          No materials found for {title}
        </div>
      );
    }

    let sortedMaterials = materials;
    if (sortConfig) {
      sortedMaterials = sortMaterials(materials, sortConfig.key, sortConfig.direction);
    }

    return (
      <div className="overflow-x-auto">
        <table className="min-w-full table-auto border-collapse">
          <thead>
            <tr className="bg-gray-900">
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('item_name')}
              >
                <div className="flex items-center justify-between">
                  <span>Item Name</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('item_name')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('quantity')}
              >
                <div className="flex items-center justify-between">
                  <span>Quantity</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('quantity')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('unit')}
              >
                <div className="flex items-center justify-between">
                  <span>Unit</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('unit')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('size')}
              >
                <div className="flex items-center justify-between">
                  <span>Size</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('size')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('reference_sheet')}
              >
                <div className="flex items-center justify-between">
                  <span>Sheet</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('reference_sheet')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('zone')}
              >
                <div className="flex items-center justify-between">
                  <span>Zone</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('zone')}</span>
                </div>
              </th>
              <th
                className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600 cursor-pointer hover:bg-gray-700 transition-colors duration-200"
                onClick={() => handleSort('confidence')}
              >
                <div className="flex items-center justify-between">
                  <span>Confidence</span>
                  <span className="ml-1 text-gray-400">{getSortIcon('confidence')}</span>
                </div>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {sortedMaterials.map((material, index) => (
              <tr key={index} className={index % 2 === 0 ? 'bg-gray-800' : 'bg-gray-900'}>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  <div>
                    <div className="font-medium">{material.item_name}</div>
                    {material.specification && (
                      <div className="text-xs text-gray-400">{material.specification}</div>
                    )}
                    {material.notes && (
                      <div className="text-xs text-gray-400 mt-1">{material.notes}</div>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600 font-mono">
                  {material.quantity}
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  {material.unit}
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  {material.size || '-'}
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  {material.reference_sheet || '-'}
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  {material.zone || '-'}
                </td>
                <td className="px-4 py-3 text-sm text-white border-b border-gray-600">
                  {material.confidence ? (
                    <span className={`px-2 py-1 rounded text-xs ${
                      material.confidence >= 0.8 ? 'bg-green-600' :
                      material.confidence >= 0.6 ? 'bg-yellow-600' : 'bg-red-600'
                    }`}>
                      {(material.confidence * 100).toFixed(0)}%
                    </span>
                  ) : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // Render structured data format
  if (isStructuredData && analysisData) {
    return (
      <div className="w-[65%] p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
        <h2 className="text-2xl font-semibold text-white mb-4">Analysis Results</h2>
        
        {/* Analysis Summary */}
        <div className="mb-6 p-4 bg-gray-900 rounded-lg">
          <h3 className="text-lg font-medium text-white mb-2">Analysis Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Total Sheets:</span>
              <div className="text-white font-medium">{analysisData.sheets_processed.length}</div>
            </div>
            <div>
              <span className="text-gray-400">Total Materials:</span>
              <div className="text-white font-medium">{analysisData.consolidated_materials.length}</div>
            </div>
            <div>
              <span className="text-gray-400">Workflow:</span>
              <div className="text-white font-medium">{analysisData.metadata.workflow}</div>
            </div>
            <div>
              <span className="text-gray-400">Status:</span>
              <div className={`font-medium flex items-center gap-2 ${
                analysisData.processing_status.total_errors > 0 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {analysisData.processing_status.message}
                {analysisData.processing_status.total_errors > 0 && (
                  <span className="text-red-400 text-sm">
                    ({analysisData.processing_status.total_errors} error{analysisData.processing_status.total_errors !== 1 ? 's' : ''})
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {/* Consolidated Materials */}
          <div className="border rounded-lg overflow-hidden border-green-500 bg-green-900/20">
            <div className="w-full px-4 py-3 bg-green-700 flex justify-between items-center">
              <button
                onClick={() => toggleSection('Materials Summary')}
                className="flex items-center gap-2 text-left transition-colors duration-200 hover:text-green-200"
              >
                <h3 className="text-lg font-medium flex items-center gap-2 text-green-100">
                  <span>🔧</span>
                  Consolidated Materials ({analysisData.consolidated_materials.length} items)
                </h3>
                <svg
                  className={`w-5 h-5 text-gray-300 transform transition-transform duration-200 ${
                    expandedSections['Materials Summary'] ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {/* Export Controls */}
              <div className="flex items-center gap-3">
                {/* Radio buttons for format selection */}
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-1 text-green-100 text-sm">
                    <input
                      type="radio"
                      name="exportFormat"
                      value="csv"
                      checked={exportFormat === 'csv'}
                      onChange={(e) => setExportFormat(e.target.value as 'csv')}
                      className="w-3 h-3 text-green-600 bg-gray-100 border-gray-300 focus:ring-green-500"
                    />
                    CSV
                  </label>
                  <label className="flex items-center gap-1 text-green-100 text-sm">
                    <input
                      type="radio"
                      name="exportFormat"
                      value="xlsx"
                      checked={exportFormat === 'xlsx'}
                      onChange={(e) => setExportFormat(e.target.value as 'xlsx')}
                      className="w-3 h-3 text-green-600 bg-gray-100 border-gray-300 focus:ring-green-500"
                    />
                    XLSX
                  </label>
                </div>
                
                                {/* Export button */}
                <button
                  onClick={handleExport}
                  disabled={!exportFormat || isExporting}
                  className={`px-4 py-2 rounded-md text-sm font-semibold transition-all duration-200 flex items-center gap-2 shadow-lg ${
                    exportFormat && !isExporting
                      ? 'bg-blue-600 hover:bg-blue-700 text-white border border-blue-500 hover:shadow-xl transform hover:scale-105'
                      : 'bg-gray-600 text-gray-300 cursor-not-allowed border border-gray-500'
                  }`}
                  title={
                    isExporting 
                      ? 'Exporting...' 
                      : !exportFormat 
                        ? 'Please select an export format first' 
                        : 'Export consolidated materials'
                  }
                >
                  {isExporting && (
                    <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {exportSuccess && (
                    <svg className="h-3 w-3 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  {isExporting ? 'Exporting...' : exportSuccess ? 'Saved' : 'Export'}
                </button>
              </div>
            </div>
            {expandedSections['Materials Summary'] && (
              <div className="p-4 bg-green-900/10">
                {renderMaterialsTable(analysisData.consolidated_materials, 'All Sheets')}
              </div>
            )}
          </div>

          {/* Context Results */}
          {analysisData.context_results.map((contextResult, index) => (
            <div key={index} className="border rounded-lg overflow-hidden border-blue-500 bg-blue-900/20">
              <button
                onClick={() => toggleSection(`Context ${index + 1}`)}
                className="w-full px-4 py-3 text-left transition-colors duration-200 flex justify-between items-center bg-blue-700 hover:bg-blue-600"
              >
                <h3 className="text-lg font-medium flex items-center gap-2 text-blue-100">
                  <span>📋</span>
                  Sheet Context: {contextResult.sheet_metadata.sheet_id || `Page ${index + 1}`}
                </h3>
                <svg
                  className={`w-5 h-5 text-gray-300 transform transition-transform duration-200 ${
                    expandedSections[`Context ${index + 1}`] ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {expandedSections[`Context ${index + 1}`] && (
                <div className="p-4 bg-blue-900/10">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    <div>
                      <h4 className="font-medium text-white mb-2">Sheet Metadata</h4>
                      <div className="text-sm space-y-1">
                        <div><span className="text-gray-400">Sheet ID:</span> {contextResult.sheet_metadata.sheet_id || 'N/A'}</div>
                        <div><span className="text-gray-400">Discipline:</span> {contextResult.sheet_metadata.discipline || 'N/A'}</div>
                        <div><span className="text-gray-400">Floor:</span> {contextResult.sheet_metadata.floor || 'N/A'}</div>
                        <div><span className="text-gray-400">Title:</span> {contextResult.sheet_metadata.title || 'N/A'}</div>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium text-white mb-2">Drawing Types</h4>
                      <div className="flex flex-wrap gap-2">
                        {contextResult.drawing_types.map((type, typeIndex) => (
                          <span key={typeIndex} className="px-2 py-1 bg-blue-600 text-blue-100 rounded text-xs">
                            {type}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  {contextResult.legend.length > 0 && (
                    <div>
                      <h4 className="font-medium text-white mb-2">Legend ({contextResult.legend.length} symbols)</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                        {contextResult.legend.map((legendItem, legendIndex) => (
                          <div key={legendIndex} className="flex items-center gap-2 p-2 bg-gray-800 rounded">
                            <span className="font-mono bg-gray-700 px-2 py-1 rounded text-xs">{legendItem.symbol}</span>
                            <span className="text-gray-300">{legendItem.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {/* Plumbing Results per Sheet */}
          {analysisData.plumbing_results.map((plumbingResult, index) => (
            <div key={index} className="border rounded-lg overflow-hidden border-orange-500 bg-orange-900/20">
              <button
                onClick={() => toggleSection(`Plumbing ${index + 1}`)}
                className="w-full px-4 py-3 text-left transition-colors duration-200 flex justify-between items-center bg-orange-700 hover:bg-orange-600"
              >
                <h3 className="text-lg font-medium flex items-center gap-2 text-orange-100">
                  <span>🔧</span>
                  Sheet Analysis: {analysisData.sheets_processed[index] || `Sheet ${index + 1}`} ({plumbingResult.materials.length} materials)
                </h3>
                <svg
                  className={`w-5 h-5 text-gray-300 transform transition-transform duration-200 ${
                    expandedSections[`Plumbing ${index + 1}`] ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {expandedSections[`Plumbing ${index + 1}`] && (
                <div className="p-4 bg-orange-900/10">
                  {renderMaterialsTable(plumbingResult.materials, `Sheet ${index + 1}`)}
                  
                  {plumbingResult.special_requirements.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-medium text-white mb-2">Special Requirements</h4>
                      <ul className="space-y-1 text-sm">
                        {plumbingResult.special_requirements.map((req, reqIndex) => (
                          <li key={reqIndex} className="text-gray-300">• {req}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {plumbingResult.potential_issues.length > 0 && (
                    <div className="mt-4">
                      <h4 className="font-medium text-white mb-2">Potential Issues</h4>
                      <ul className="space-y-1 text-sm">
                        {plumbingResult.potential_issues.map((issue, issueIndex) => (
                          <li key={issueIndex} className="text-red-300">⚠ {issue}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Error Display Section - Moved to Bottom */}
        {analysisData.processing_status.total_errors > 0 && (
          <div className="mt-6 p-4 bg-red-900/20 border border-red-500 rounded-lg">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-red-400 text-xl">⚠️</span>
              <h3 className="text-lg font-medium text-red-400">
                Analysis Errors ({analysisData.processing_status.total_errors})
              </h3>
            </div>
            
            <div className="mb-4">
              <h4 className="text-sm font-medium text-red-300 mb-2">Error Summary:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">By Agent:</span>
                  <div className="text-red-200">
                    {Object.entries(analysisData.processing_status.agent_error_summary).map(([agent, count]) => (
                      <div key={agent} className="ml-2">• {agent}: {count} error{count !== 1 ? 's' : ''}</div>
                    ))}
                  </div>
                </div>
                {Object.keys(analysisData.processing_status.sheet_error_summary).length > 0 && (
                  <div>
                    <span className="text-gray-400">By Sheet:</span>
                    <div className="text-red-200">
                      {Object.entries(analysisData.processing_status.sheet_error_summary).map(([sheet, count]) => (
                        <div key={sheet} className="ml-2">• Sheet {sheet}: {count} error{count !== 1 ? 's' : ''}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-medium text-red-300">Detailed Errors:</h4>
              <div className="max-h-40 overflow-y-auto space-y-2">
                {analysisData.processing_status.errors.map((error, index) => (
                  <div key={index} className="p-3 bg-red-900/30 border border-red-700 rounded text-sm">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-medium text-red-300">
                        {error.agent} {error.sheet_id && `(Sheet ${error.sheet_id})`}
                      </span>
                      <span className="text-xs text-red-400">
                        {new Date(error.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-red-200 mb-1">
                      <span className="text-red-400 font-medium">Type:</span> {error.error_type}
                    </div>
                    <div className="text-red-100">
                      {error.error_message}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Legacy fallback for old string format
  const renderContent = (section: AnalysisSection) => {
    if (section.title === 'Materials and Quantities') {
      // Parse table content from legacy format
      const lines = section.content.trim().split('\n');
      const tableRows = lines.filter(line => line.includes('|'));
      
      if (tableRows.length > 0) {
        const headers = tableRows[0].split('|').filter(cell => cell.trim());
        const dataRows = tableRows.slice(2); // Skip header and separator
        
        return (
          <div className="overflow-x-auto">
            <table className="min-w-full table-auto border-collapse">
              <thead>
                <tr className="bg-gray-900">
                  {headers.map((header, index) => (
                    <th key={index} className="px-4 py-3 text-left text-xs font-medium text-white uppercase tracking-wider border-b border-gray-600">
                      {header.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {dataRows.map((row, rowIndex) => (
                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-gray-800' : 'bg-gray-900'}>
                    {row.split('|').filter(cell => cell.trim()).map((cell, cellIndex) => (
                      <td key={cellIndex} className="px-4 py-3 text-sm text-white border-b border-gray-600">
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
    
    return (
      <div className="text-gray-300 whitespace-pre-wrap prose prose-invert max-w-none">
        {section.content}
      </div>
    );
  };

  // Legacy rendering for old string format
  const sections = parseAnalysis(typeof analysis === 'string' ? analysis : '');
  const allSections = context ? [
    { title: 'Document Context', content: context },
    ...sections
  ] : sections;

  return (
    <div className="w-[65%] p-6 bg-gray-800 rounded-xl shadow-lg border border-gray-700">
      <h2 className="text-2xl font-semibold text-white mb-4">Analysis Results (Legacy Format)</h2>
      <div className="space-y-4">
        {allSections.map((section, index) => (
          <div key={index} className={`border rounded-lg overflow-hidden ${
            section.title === 'Document Context' 
              ? 'border-blue-500 bg-blue-900/20' 
              : 'border-gray-700'
          }`}>
            <button
              onClick={() => toggleSection(section.title)}
              className={`w-full px-4 py-3 text-left transition-colors duration-200 flex justify-between items-center ${
                section.title === 'Document Context'
                  ? 'bg-blue-700 hover:bg-blue-600'
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              <h3 className={`text-lg font-medium flex items-center gap-2 ${
                section.title === 'Document Context' ? 'text-blue-100' : 'text-white'
              }`}>
                {section.title === 'Document Context' && <span>📋</span>}
                {section.title}
              </h3>
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
              <div className={`p-4 ${
                section.title === 'Document Context' 
                  ? 'bg-blue-900/10' 
                  : 'bg-gray-800'
              }`}>
                {renderContent(section)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
} 