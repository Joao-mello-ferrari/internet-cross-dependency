#!/usr/bin/env python3
"""
Dependency verification script for Internet Cross-Dependency Analysis project.
Run this script to verify that all required Python packages are installed correctly.
"""

import sys
import importlib
from pathlib import Path

# List of required packages to verify
REQUIRED_PACKAGES = [
    'numpy',
    'pandas', 
    'matplotlib',
    'seaborn',
    'scipy',
    'requests',
    'httpx',
    'aiohttp',
    'tqdm',
    'openai',
    'dotenv',
    'ipwhois',
    'ping3',
    'psycopg2',
    'google.cloud.bigquery',
    'tldextract',
    'drawsvg',
]

def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    required = (3, 12, 4)
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= required:
        print("✅ Python version requirement met")
        return True
    else:
        print(f"❌ Python version {required[0]}.{required[1]}.{required[2]} or higher required")
        return False

def check_package(package_name, optional=False):
    """Check if a package can be imported."""
    try:
        importlib.import_module(package_name)
        status = "✅" if not optional else "✅ (optional)"
        print(f"{status} {package_name}")
        return True
    except ImportError as e:
        status = "❌" if not optional else "⚠️ (optional)"
        print(f"{status} {package_name} - {str(e)}")
        return False

def check_node_version():
    """Check if Node.js is available and meets version requirements."""
    try:
        import subprocess
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_str = result.stdout.strip().replace('v', '')
            version_parts = version_str.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            print(f"Node.js version: {version_str}")
            
            if major >= 22 and minor >= 17:
                print("✅ Node.js version requirement met")
                return True
            else:
                print("❌ Node.js 22.17.0 or higher required")
                return False
        else:
            print("❌ Node.js not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Node.js: {e}")
        return False

def check_js_dependencies():
    """Check if JavaScript dependencies are installed."""
    js_project_path = Path("src/steps/locality/locedge/classify_headers")
    package_json = js_project_path / "package.json"
    node_modules = js_project_path / "node_modules"
    
    if not package_json.exists():
        print("❌ JavaScript package.json not found")
        return False
    
    if not node_modules.exists():
        print("❌ JavaScript dependencies not installed (node_modules missing)")
        print("   Run: cd src/steps/locality/locedge/classify_headers && npm install")
        return False
    
    print("✅ JavaScript dependencies appear to be installed")
    return True

def main():
    """Main verification function."""
    print("🔍 Verifying Internet Cross-Dependency Analysis Dependencies\n")
    
    all_good = True
    
    # Check Python version
    print("📋 Checking Python version...")
    if not check_python_version():
        all_good = False
    print()
    
    # Check required Python packages
    print("📦 Checking required Python packages...")
    for package in REQUIRED_PACKAGES:
        if not check_package(package):
            all_good = False
    print()
    
    # Check Node.js
    print("🟢 Checking Node.js version...")
    if not check_node_version():
        all_good = False
    print()
    
    # Check JavaScript dependencies
    print("📦 Checking JavaScript dependencies...")
    if not check_js_dependencies():
        all_good = False
    print()
    
    # Final result
    if all_good:
        print("🎉 All required dependencies are installed correctly!")
        print("You can now run the analysis pipeline.")
    else:
        print("❌ Some required dependencies are missing.")
        print("Please install missing packages and run this script again.")
        print("\nTo install Python dependencies: pip install -r requirements.txt")
        print("To install JavaScript dependencies: cd src/steps/locality/locedge/classify_headers && npm install")
        sys.exit(1)

if __name__ == "__main__":
    main()