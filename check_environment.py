#!/usr/bin/env python
"""
Environment checker and auto-fixer for Risk Model Pipeline
Automatically detects and fixes common dependency issues
"""

import sys
import subprocess
import importlib
from packaging import version
import json

class EnvironmentChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        
    def check_python_version(self):
        """Check Python version"""
        py_version = sys.version_info
        if py_version.major != 3 or py_version.minor < 8:
            self.issues.append(f"Python 3.8+ required, found {py_version.major}.{py_version.minor}")
        else:
            print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    def check_package(self, package_name, min_version=None, max_version=None):
        """Check if a package is installed with correct version"""
        try:
            module = importlib.import_module(package_name.replace('-', '_'))
            installed_version = getattr(module, '__version__', 'unknown')
            
            if min_version and installed_version != 'unknown':
                if version.parse(installed_version) < version.parse(min_version):
                    self.issues.append(f"{package_name} version {installed_version} < {min_version}")
                    return False
            
            if max_version and installed_version != 'unknown':
                if version.parse(installed_version) > version.parse(max_version):
                    self.issues.append(f"{package_name} version {installed_version} > {max_version}")
                    return False
            
            print(f"✓ {package_name} {installed_version}")
            return True
            
        except ImportError:
            self.issues.append(f"{package_name} not installed")
            return False
    
    def auto_fix(self):
        """Attempt to auto-fix issues"""
        print("\n🔧 Attempting to fix issues...")
        
        for issue in self.issues:
            if "not installed" in issue:
                package = issue.split()[0]
                print(f"Installing {package}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                 check=True, capture_output=True)
                    self.fixes_applied.append(f"Installed {package}")
                except:
                    print(f"Failed to install {package}")
            
            elif "version" in issue:
                # Extract package and version from issue
                parts = issue.split()
                package = parts[0]
                
                # Find appropriate version from requirements
                req_version = self.get_requirement_version(package)
                if req_version:
                    print(f"Installing {package}=={req_version}...")
                    try:
                        subprocess.run([sys.executable, "-m", "pip", "install", 
                                      f"{package}=={req_version}"], 
                                     check=True, capture_output=True)
                        self.fixes_applied.append(f"Fixed {package} version")
                    except:
                        print(f"Failed to fix {package} version")
    
    def get_requirement_version(self, package):
        """Get recommended version from requirements file"""
        requirements = {
            'pandas': '1.4.4',
            'numpy': '1.21.5',
            'scikit-learn': '1.0.2',
            'matplotlib': '3.5.3',
            'seaborn': '0.12.2',
        }
        return requirements.get(package)
    
    def run_full_check(self):
        """Run complete environment check"""
        print("🔍 Checking environment for Risk Model Pipeline")
        print("=" * 60)
        
        # Check Python
        self.check_python_version()
        
        # Check core dependencies
        print("\n📦 Core Dependencies:")
        core_deps = [
            ('pandas', '1.3.0', '2.0.0'),
            ('numpy', '1.20.0', '1.25.0'),
            ('scikit-learn', '1.0.0', '1.3.0'),
            ('joblib', '1.0.0', None),
            ('openpyxl', '3.0.0', None),
        ]
        
        for dep in core_deps:
            name = dep[0]
            min_v = dep[1] if len(dep) > 1 else None
            max_v = dep[2] if len(dep) > 2 else None
            self.check_package(name, min_v, max_v)
        
        # Check optional dependencies
        print("\n📦 Optional Dependencies:")
        optional_deps = [
            ('matplotlib', None, None),
            ('seaborn', None, None),
            ('shap', None, None),
            ('optuna', None, None),
        ]
        
        for dep in optional_deps:
            name = dep[0]
            if not self.check_package(name):
                self.warnings.append(f"{name} not available (optional)")
        
        # Summary
        print("\n" + "=" * 60)
        if self.issues:
            print(f"❌ Found {len(self.issues)} issues:")
            for issue in self.issues:
                print(f"   - {issue}")
            
            # Offer auto-fix
            response = input("\nAttempt to auto-fix issues? (y/n): ")
            if response.lower() == 'y':
                self.auto_fix()
                
                if self.fixes_applied:
                    print("\n✅ Fixes applied:")
                    for fix in self.fixes_applied:
                        print(f"   - {fix}")
                else:
                    print("\n⚠️ No fixes could be applied automatically")
        else:
            print("✅ Environment is ready!")
        
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        # Save report
        report = {
            'issues': self.issues,
            'warnings': self.warnings,
            'fixes_applied': self.fixes_applied,
            'status': 'ready' if not self.issues else 'issues_found'
        }
        
        with open('environment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📄 Report saved to environment_report.json")
        
        return len(self.issues) == 0

if __name__ == "__main__":
    checker = EnvironmentChecker()
    success = checker.run_full_check()
    sys.exit(0 if success else 1)