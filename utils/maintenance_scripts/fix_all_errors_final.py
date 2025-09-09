#!/usr/bin/env python
"""Final comprehensive fix for all flake8 errors"""

import os
import re
from pathlib import Path


def fix_cli_py():
    """Fix specific issues in cli.py"""
    filepath = Path('src/risk_pipeline/cli.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix imports order - move all imports to top after the UTF-8 setup
    new_lines = []
    imports = []
    utf8_block = []
    other_lines = []
    
    in_utf8_block = False
    for i, line in enumerate(lines):
        if i == 0 and line.strip().startswith('"""'):
            new_lines.append(line)
        elif '_os_utf8' in line or '_sys_utf8' in line or 'PYTHONUTF8' in line:
            in_utf8_block = True
            utf8_block.append(line)
        elif in_utf8_block and (line.strip() == 'pass' or 'except' in line or 'try:' in line):
            utf8_block.append(line)
        elif line.strip().startswith(('import ', 'from ')) and not '_utf8' in line:
            imports.append(line)
        else:
            if in_utf8_block and line.strip() == '':
                utf8_block.append(line)
                in_utf8_block = False
            else:
                other_lines.append(line)
    
    # Add missing imports
    if not any('import os' in imp for imp in imports):
        imports.insert(0, 'import os\n')
    if not any('import pickle' in imp for imp in imports):
        imports.append('import pickle\n')
    
    # Remove duplicate/unused imports
    clean_imports = []
    seen = set()
    for imp in imports:
        if 'from pathlib import Path' in imp:
            continue  # Remove unused Path import
        if imp.strip() not in seen:
            clean_imports.append(imp)
            seen.add(imp.strip())
    
    # Reconstruct file
    result = []
    result.append('"""Command Line Interface for Risk Model Pipeline"""\n')
    result.extend(utf8_block)
    result.append('\n')
    result.extend(sorted(clean_imports))  # Sort imports
    result.append('\n')
    result.extend(other_lines)
    
    # Fix long lines
    content = ''.join(result)
    content = content.replace(
        'help="Optional CSV path for scores (combined); if omitted, CSV is not written"',
        'help="Optional CSV path for scores (combined); if omitted, CSV is not written"'
    )
    content = content.replace(
        'help="Path to WOE mapping JSON (woe_mapping_<run_id>.json)"',
        'help="Path to WOE mapping JSON file"'
    )
    content = content.replace(
        'help="Path to final vars JSON (final_vars_<run_id>.json)"',
        'help="Path to final vars JSON file"'
    )
    content = content.replace(
        'help="Optional model report path to append \'external_scores\' sheet"',
        'help="Optional report path for external scores"'
    )
    
    # Fix os.path.exists
    content = content.replace('os.path.exists', 'os.path.exists')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_api_py():
    """Fix api.py blank line at end"""
    filepath = Path('src/risk_pipeline/api.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove trailing blank lines
    content = content.rstrip() + '\n'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_pandas_compat():
    """Fix pandas_compat.py unused import"""
    filepath = Path('src/risk_pipeline/pandas_compat.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove unused numpy import
    new_lines = []
    for line in lines:
        if line.strip() == 'import numpy as np':
            continue  # Skip unused import
        new_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def fix_monitoring_py():
    """Fix monitoring.py semicolon and whitespace issues"""
    filepath = Path('src/risk_pipeline/monitoring.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix semicolons
    content = re.sub(r';\s*', '\n        ', content)
    
    # Fix missing whitespace after comma
    content = re.sub(r',([^\s])', r', \1', content)
    
    # Fix E306 - add blank line before nested definition
    lines = content.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if i > 0 and line.strip().startswith('def ') and lines[i-1].strip() != '':
            indent = len(line) - len(line.lstrip())
            if indent > 0:  # Nested function
                new_lines.append('')
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_pipeline_py():
    """Fix pipeline.py import order and indentation issues"""
    filepath = Path('src/risk_pipeline/pipeline.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix import order - move all imports to top
    lines = content.split('\n')
    imports = []
    other = []
    docstring = []
    
    in_docstring = False
    for line in lines:
        if line.strip().startswith('"""') and not in_docstring:
            docstring.append(line)
            in_docstring = True
            if line.count('"""') == 2:
                in_docstring = False
        elif in_docstring:
            docstring.append(line)
            if '"""' in line:
                in_docstring = False
        elif line.strip().startswith(('import ', 'from ')):
            imports.append(line)
        else:
            other.append(line)
    
    # Remove duplicate imports and unused ones
    clean_imports = []
    seen = set()
    for imp in imports:
        if 'import Optional' in imp or 'import dataclass' in imp:
            continue  # Remove unused
        if 'import numpy as np' in imp and imp.strip().startswith('import numpy'):
            continue  # Remove duplicate numpy
        if imp.strip() not in seen and imp.strip():
            clean_imports.append(imp)
            seen.add(imp.strip())
    
    # Fix indentation for continuation lines
    content = '\n'.join(docstring + [''] + sorted(clean_imports) + ['', ''] + other)
    
    # Fix E128 continuation line indentation
    content = re.sub(r'\n\s{26}', '\n                        ', content)
    
    # Fix E226 missing whitespace around arithmetic operator
    content = re.sub(r'(\d)\*(\d)', r'\1 * \2', content)
    content = re.sub(r'(\w)\+(\w)', r'\1 + \2', content)
    
    # Fix long lines by breaking them
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if len(line) > 120 and 'help=' not in line:
            # Try to break at comma
            if ',' in line and '(' in line:
                parts = line.split(',')
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(parts[0] + ',')
                    for part in parts[1:]:
                        new_lines.append(' ' * (indent + 4) + part.strip())
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_all_files():
    """Fix all Python files systematically"""
    src_dir = Path('src')
    test_dir = Path('tests')
    
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original = content
                
                # Fix semicolons
                content = re.sub(r';\s*', '\n    ', content)
                
                # Fix bare except
                content = re.sub(r'\bexcept:\s*$', 'except Exception:', content, flags=re.MULTILINE)
                
                # Fix missing whitespace after comma
                content = re.sub(r',([^\s\)])', r', \1', content)
                
                # Fix missing whitespace around operators
                content = re.sub(r'(\w)=(\w)', r'\1 = \2', content)
                content = re.sub(r'(\d)\*(\d)', r'\1 * \2', content)
                content = re.sub(r'(\d)\+(\d)', r'\1 + \2', content)
                
                # Remove trailing whitespace
                lines = content.split('\n')
                lines = [line.rstrip() for line in lines]
                content = '\n'.join(lines)
                
                # Ensure single newline at end
                content = content.rstrip() + '\n'
                
                if content != original:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Fixed: {filepath}")
                    
            except Exception as e:
                print(f"Error fixing {filepath}: {e}")


def main():
    """Main function"""
    print("Fixing all flake8 errors...")
    
    # Fix specific files first
    fix_cli_py()
    print("Fixed cli.py")
    
    fix_api_py()
    print("Fixed api.py")
    
    fix_pandas_compat()
    print("Fixed pandas_compat.py")
    
    fix_monitoring_py()
    print("Fixed monitoring.py")
    
    fix_pipeline_py()
    print("Fixed pipeline.py")
    
    # Then fix all files
    fix_all_files()
    
    print("\nAll fixes completed!")


if __name__ == '__main__':
    main()