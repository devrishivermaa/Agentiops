#!/usr/bin/env python3
# view_reports.py
"""
List and optionally open generated analysis reports.
"""

import os
import sys
import subprocess
from datetime import datetime

def list_reports(output_dir="output"):
    """List all generated reports."""
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return []
    
    # Find all report files
    pdf_files = []
    json_files = []
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('_analysis.pdf') or 'analysis' in file and file.endswith('.pdf'):
                pdf_files.append(full_path)
            elif file.endswith('_results.json') or 'results' in file and file.endswith('.json'):
                json_files.append(full_path)
    
    return sorted(pdf_files, key=os.path.getmtime, reverse=True), sorted(json_files, key=os.path.getmtime, reverse=True)


def format_file_info(filepath):
    """Format file information for display."""
    stat = os.stat(filepath)
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    
    return {
        'path': filepath,
        'name': os.path.basename(filepath),
        'size': f"{size_mb:.2f} MB",
        'modified': mtime.strftime('%Y-%m-%d %H:%M:%S')
    }


def open_file(filepath):
    """Open file with system default application."""
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', filepath])
        elif sys.platform == 'win32':  # Windows
            os.startfile(filepath)
        else:  # Linux
            subprocess.run(['xdg-open', filepath])
        
        return True
    except Exception as e:
        print(f"❌ Failed to open file: {e}")
        return False


def main():
    print("=" * 80)
    print("📊 ANALYSIS REPORTS")
    print("=" * 80)
    
    pdf_files, json_files = list_reports()
    
    if not pdf_files and not json_files:
        print("\n⚠️  No reports found in 'output' directory.")
        print("   Run 'python3 main.py' to generate reports.")
        print("=" * 80)
        return
    
    # Display PDF reports
    if pdf_files:
        print(f"\n📄 PDF Reports ({len(pdf_files)} found):\n")
        for idx, pdf_file in enumerate(pdf_files, 1):
            info = format_file_info(pdf_file)
            print(f"{idx}. {info['name']}")
            print(f"   Size: {info['size']} | Modified: {info['modified']}")
            print(f"   Path: {info['path']}\n")
    
    # Display JSON files
    if json_files:
        print(f"📊 JSON Data Files ({len(json_files)} found):\n")
        for idx, json_file in enumerate(json_files, 1):
            info = format_file_info(json_file)
            print(f"{idx}. {info['name']}")
            print(f"   Size: {info['size']} | Modified: {info['modified']}")
            print(f"   Path: {info['path']}\n")
    
    print("=" * 80)
    
    # Offer to open latest PDF
    if pdf_files:
        print(f"\n💡 Latest report: {os.path.basename(pdf_files[0])}")
        
        try:
            choice = input("   Open latest PDF report? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                print("   Opening...")
                if open_file(pdf_files[0]):
                    print("   ✅ Opened successfully")
                else:
                    print(f"   ℹ️  Manually open: {pdf_files[0]}")
        except (EOFError, KeyboardInterrupt):
            print("\n   Skipped")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
