#!/usr/bin/env python3
"""
Open-MSI: Open-Source GUI Tool for Mass Spectrometry Analysis

Main entry point for the Open-MSI application.
"""

import sys
import os

def main():
    """Main entry point for Open-MSI application."""
    # Add the src directory to the Python path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print("=" * 50)
    print("Open-MSI: Mass Spectrometry Analysis Tool")
    print("=" * 50)
    print("Loading application...")
    
    try:
        # Import and run the GUI application
        from GuiCode.ToFMSGui import *
        print("✓ Application loaded successfully!")
        print("✓ GUI interface should now be available")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Please install required packages with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()