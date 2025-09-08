"""Fix matplotlib installation issues"""

import subprocess
import sys
import os

print("Fixing matplotlib installation...")
print("=" * 60)

# Step 1: Try to remove broken matplotlib
print("\n1. Removing broken matplotlib files...")
try:
    # Force reinstall
    subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "matplotlib==3.5.3"], 
                   capture_output=True, text=True, timeout=300)
    print("   [OK] Matplotlib 3.5.3 installed")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Step 2: Install dependencies
print("\n2. Installing matplotlib dependencies...")
dependencies = [
    "cycler>=0.10",
    "kiwisolver>=1.0.1", 
    "pillow>=6.2.0",
    "pyparsing>=2.2.1",
    "python-dateutil>=2.7"
]

for dep in dependencies:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                      capture_output=True, text=True, timeout=60)
        print(f"   [OK] {dep} installed")
    except Exception as e:
        print(f"   [ERROR] {dep} failed: {e}")

# Step 3: Test matplotlib
print("\n3. Testing matplotlib...")
try:
    import matplotlib
    print(f"   [OK] Matplotlib {matplotlib.__version__} imported successfully!")
    
    import matplotlib.pyplot as plt
    print("   [OK] pyplot imported successfully!")
    
    # Simple test plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    plt.close()
    print("   [OK] Test plot created successfully!")
    
except ImportError as e:
    print(f"   [ERROR] Import error: {e}")
    print("\nAlternative: Use conda instead:")
    print("   conda update matplotlib")
    print("   OR")
    print("   conda install matplotlib=3.5.3")
    
except Exception as e:
    print(f"   [ERROR] Test failed: {e}")

print("\n" + "=" * 60)
print("Fix attempt completed!")
print("\nIf matplotlib still doesn't work, try:")
print("1. Restart your Python kernel/Jupyter")
print("2. Use conda: conda install matplotlib")
print("3. The notebook will work without plots (text-only mode)")