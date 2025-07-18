#!/usr/bin/env python3
"""
Test script to validate the analysis improvements
"""

import requests
import json
import time
import sys
import os

def test_analysis_improvements(pdf_file_path, backend_url="http://localhost:8000"):
    """Test the improved analysis pipeline"""
    
    print("Testing Analysis Improvements")
    print("=" * 50)
    
    if not os.path.exists(pdf_file_path):
        print(f"Error: PDF file not found: {pdf_file_path}")
        return False
    
    print(f"Testing with PDF: {pdf_file_path}")
    print(f"Backend URL: {backend_url}")
    
    # Step 1: Upload PDF
    print("\n1. Uploading PDF...")
    with open(pdf_file_path, 'rb') as f:
        files = {'file': f}
        upload_response = requests.post(f"{backend_url}/api/upload", files=files)
    
    if upload_response.status_code != 200:
        print(f"Upload failed: {upload_response.status_code}")
        return False
    
    upload_data = upload_response.json()
    task_id = upload_data['taskId']
    filename = upload_data['filename']
    print(f"Upload successful. Task ID: {task_id}")
    
    # Step 2: Start Analysis
    print("\n2. Starting analysis...")
    analysis_data = {
        'taskId': task_id,
        'filename': filename
    }
    analysis_response = requests.post(f"{backend_url}/api/analyze", data=analysis_data)
    
    if analysis_response.status_code != 200:
        print(f"Analysis failed: {analysis_response.status_code}")
        return False
    
    print("Analysis started successfully")
    
    # Step 3: Wait for completion and check progress
    print("\n3. Monitoring progress...")
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        progress_response = requests.get(f"{backend_url}/api/progress/{task_id}")
        if progress_response.status_code == 200:
            progress = progress_response.json()
            print(f"Progress: {progress.get('progress', 0)}% - {progress.get('message', 'Processing...')}")
            
            if progress.get('stage') == 'complete':
                break
                
            if progress.get('stage') == 'failed':
                print("Analysis failed!")
                return False
        
        time.sleep(5)
    
    # Step 4: Get Results
    print("\n4. Retrieving results...")
    results_response = requests.get(f"{backend_url}/api/result/{task_id}")
    
    if results_response.status_code != 200:
        print(f"Failed to get results: {results_response.status_code}")
        return False
    
    results = results_response.json()
    analysis = results.get('analysis', {})
    
    # Step 5: Check Improvements
    print("\n5. Checking Improvements:")
    print("-" * 30)
    
    # Check filtering
    metadata = analysis.get('metadata', {})
    materials_before = metadata.get('materials_before_filtering', 0)
    materials_after = metadata.get('total_materials', 0)
    confidence_threshold = metadata.get('confidence_threshold', 0.6)
    
    print(f"Materials before filtering: {materials_before}")
    print(f"Materials after filtering: {materials_after}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Materials filtered out: {materials_before - materials_after}")
    
    # Check confidence distribution
    materials = analysis.get('consolidated_materials', [])
    confidence_stats = {
        'total': len(materials),
        'with_confidence': 0,
        'high_confidence': 0,
        'low_confidence': 0,
        'avg_confidence': 0
    }
    
    total_confidence = 0
    for material in materials:
        confidence = material.get('confidence')
        if confidence is not None:
            confidence_stats['with_confidence'] += 1
            total_confidence += confidence
            if confidence >= 0.7:
                confidence_stats['high_confidence'] += 1
            elif confidence < 0.5:
                confidence_stats['low_confidence'] += 1
    
    if confidence_stats['with_confidence'] > 0:
        confidence_stats['avg_confidence'] = total_confidence / confidence_stats['with_confidence']
    
    print(f"\nConfidence Statistics:")
    print(f"  Total materials: {confidence_stats['total']}")
    print(f"  With confidence scores: {confidence_stats['with_confidence']}")
    print(f"  High confidence (>=0.7): {confidence_stats['high_confidence']}")
    print(f"  Low confidence (<0.5): {confidence_stats['low_confidence']}")
    print(f"  Average confidence: {confidence_stats['avg_confidence']:.2f}")
    
    # Step 6: Get Debug Information
    print("\n6. Debug Information:")
    print("-" * 30)
    
    # Text extraction debug
    text_debug_response = requests.get(f"{backend_url}/api/debug/text-extraction/{task_id}")
    if text_debug_response.status_code == 200:
        text_debug = text_debug_response.json()
        sheets = text_debug.get('text_extraction_debug', {})
        print(f"Sheets processed: {len(sheets)}")
        for sheet_id, data in sheets.items():
            if 'error' not in data:
                print(f"  Sheet {sheet_id}: {data.get('legend_count', 0)} legend entries, {data.get('text_blocks_count', 0)} text blocks")
            else:
                print(f"  Sheet {sheet_id}: Error - {data['error']}")
    
    # Material validation debug
    validation_debug_response = requests.get(f"{backend_url}/api/debug/material-validation/{task_id}")
    if validation_debug_response.status_code == 200:
        validation_debug = validation_debug_response.json()
        validation_info = validation_debug.get('validation_debug', {})
        confidence_dist = validation_info.get('confidence_distribution', {})
        print(f"\nConfidence Distribution:")
        for range_name, count in confidence_dist.items():
            print(f"  {range_name}: {count}")
    
    # Step 7: Sample Materials
    print("\n7. Sample Materials (first 5):")
    print("-" * 30)
    for i, material in enumerate(materials[:5]):
        print(f"{i+1}. {material.get('item_name', 'Unknown')}")
        print(f"   Quantity: {material.get('quantity', 'null')} {material.get('unit', '')}")
        print(f"   Confidence: {material.get('confidence', 'null')}")
        print(f"   Sheet: {material.get('reference_sheet', 'Unknown')}")
        notes = material.get('notes', '')
        if notes:
            print(f"   Notes: {notes}")
        print()
    
    print("Test completed successfully!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_improvements.py <pdf_file_path> [backend_url]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    backend_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    success = test_analysis_improvements(pdf_file, backend_url)
    sys.exit(0 if success else 1) 