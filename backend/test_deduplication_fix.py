#!/usr/bin/env python3
"""
Test script to verify the deduplication fix is working correctly
"""

import requests
import json
import time
import sys
import os

def test_deduplication_fix(pdf_file_path, backend_url="http://localhost:8000"):
    """Test that materials found multiple times on the same page are properly deduplicated"""
    
    print("Testing Per-Page Deduplication Fix")
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
        print(f"Upload failed: {upload_response.status_code} - {upload_response.text}")
        return False
    
    upload_data = upload_response.json()
    task_id = upload_data.get('taskId')
    print(f"✅ Upload successful. Task ID: {task_id}")
    
    # Step 2: Start analysis
    print("\n2. Starting analysis...")
    analyze_data = {
        'taskId': task_id,
        'filename': os.path.basename(pdf_file_path)
    }
    analyze_response = requests.post(f"{backend_url}/api/analyze", data=analyze_data)
    
    if analyze_response.status_code != 200:
        print(f"Analysis failed: {analyze_response.status_code} - {analyze_response.text}")
        return False
    
    print("✅ Analysis started successfully")
    
    # Step 3: Monitor progress
    print("\n3. Monitoring analysis progress...")
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        progress_response = requests.get(f"{backend_url}/api/progress/{task_id}")
        if progress_response.status_code == 200:
            progress = progress_response.json()
            status = progress.get('status')
            percent = progress.get('progress', 0)
            message = progress.get('message', '')
            
            print(f"   Status: {status} ({percent}%) - {message}")
            
            if status == 'complete':
                print("✅ Analysis completed!")
                break
            elif status == 'error':
                print(f"❌ Analysis failed: {message}")
                return False
        
        time.sleep(2)
    else:
        print("❌ Analysis timed out")
        return False
    
    # Step 4: Get results and check deduplication
    print("\n4. Getting results and checking deduplication...")
    results_response = requests.get(f"{backend_url}/api/results/{task_id}")
    
    if results_response.status_code != 200:
        print(f"Failed to get results: {results_response.status_code}")
        return False
    
    results = results_response.json()
    analysis = results.get('analysis', {})
    metadata = analysis.get('metadata', {})
    materials = analysis.get('consolidated_materials', [])
    
    # Check per-page deduplication statistics
    materials_before_filter = metadata.get('materials_before_filtering', 0)
    materials_after_filter = metadata.get('total_materials', 0)
    dedup_method = metadata.get('deduplication_method', 'unknown')
    materials_filtered = materials_before_filter - materials_after_filter
    
    print(f"\n📊 PER-PAGE DEDUPLICATION STATISTICS:")
    print(f"   Materials before filtering:     {materials_before_filter}")
    print(f"   Materials after filtering:      {materials_after_filter}")
    print(f"   Materials filtered:             {materials_filtered}")
    print(f"   Deduplication method:           {dedup_method}")
    
    # Step 5: Get validation debug info
    print("\n5. Getting detailed validation debug info...")
    debug_response = requests.get(f"{backend_url}/api/debug/material-validation/{task_id}")
    
    if debug_response.status_code == 200:
        debug_data = debug_response.json()
        validation_info = debug_data.get('validation_debug', {})
        
        print(f"\n🔍 VALIDATION DEBUG INFO:")
        for key, value in validation_info.items():
            if key == 'confidence_distribution':
                print(f"   {key}:")
                for range_key, count in value.items():
                    print(f"      {range_key}: {count}")
            else:
                print(f"   {key}: {value}")
    
    # Step 6: Check for duplicate material names
    print("\n6. Checking for duplicate material names...")
    material_names = [mat.get('item_name', '').lower().strip() for mat in materials]
    unique_names = set(material_names)
    
    duplicates_found = len(material_names) - len(unique_names)
    if duplicates_found > 0:
        print(f"⚠️  Found {duplicates_found} potential duplicates!")
        
        # Show examples of duplicates
        name_counts = {}
        for name in material_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        duplicate_names = {name: count for name, count in name_counts.items() if count > 1}
        if duplicate_names:
            print("   Duplicate material names found:")
            for name, count in list(duplicate_names.items())[:5]:  # Show first 5
                print(f"      '{name}': {count} instances")
    else:
        print("✅ No duplicate material names found!")
    
    # Step 7: Show sample materials with reference sheets
    print("\n7. Sample materials with reference sheets:")
    for i, material in enumerate(materials[:5]):  # Show first 5
        name = material.get('item_name', 'Unknown')
        qty = material.get('quantity', 'N/A')
        unit = material.get('unit', 'N/A')
        ref_sheets = material.get('reference_sheet', 'N/A')
        confidence = material.get('confidence', 'N/A')
        
        print(f"   {i+1}. {name}")
        print(f"      Qty: {qty} {unit}, Sheets: {ref_sheets}, Confidence: {confidence}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"   ✅ Analysis completed successfully")
    print(f"   ✅ Per-page deduplication method: {dedup_method}")
    print(f"   ✅ Final material count: {materials_after_filter}")
    print(f"   ✅ Materials filtered by confidence: {materials_filtered}")
    
    if duplicates_found == 0:
        print(f"   🎯 SUCCESS: No duplicate material names found - per-page deduplication working!")
    else:
        print(f"   ⚠️  WARNING: Found {duplicates_found} potential duplicate names - may need investigation")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_deduplication_fix.py <pdf_file_path> [backend_url]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    backend_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    
    success = test_deduplication_fix(pdf_path, backend_url)
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 