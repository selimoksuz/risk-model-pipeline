#!/usr/bin/env python
"""Fix indentation errors and other issues"""

import os
import re
from pathlib import Path


def fix_file_structure(filepath):
    """Fix file structure issues"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip empty files
    if not lines:
        return
    
    fixed_lines = []
    i = 0
    
    # Check for docstring at beginning
    if lines and lines[0].strip().startswith('"""'):
        # Keep the docstring
        fixed_lines.append(lines[0])
        i = 1
        # Find end of docstring
        while i < len(lines):
            fixed_lines.append(lines[i])
            if '"""' in lines[i] and i > 0:
                i += 1
                break
            i += 1
    
    # Process rest of file
    prev_was_import = False
    prev_was_blank = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Fix indentation issues at start of line
        if line and line[0] == ' ' and not line[1:].lstrip():
            # Line with only spaces - make it blank
            fixed_lines.append('\n')
        elif line.startswith('    ') and i == 0:
            # Remove leading spaces at start of file
            fixed_lines.append(line.lstrip())
        else:
            # Keep the line
            fixed_lines.append(line)
        
        prev_was_import = stripped.startswith(('import ', 'from '))
        prev_was_blank = not stripped
        i += 1
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_cli_string_error(filepath):
    """Fix the string error in cli.py"""
    if 'cli.py' not in str(filepath):
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the broken string literal
    content = content.replace(
        'help="Optional CSV path for scores (combined)\n    if omitted',
        'help="Optional CSV path for scores (combined); if omitted'
    )
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def remove_leading_spaces(filepath):
    """Remove unexpected leading spaces from import lines"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Check if line starts with spaces followed by import/from
        if line.startswith('    ') and line.lstrip().startswith(('import ', 'from ', '"""')):
            # Remove leading spaces
            fixed_lines.append(line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def main():
    """Main function"""
    src_dir = Path('src')
    test_dir = Path('tests')
    
    # First fix CLI string error
    cli_path = src_dir / 'risk_pipeline' / 'cli.py'
    if cli_path.exists():
        fix_cli_string_error(cli_path)
    
    # Fix all Python files
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                remove_leading_spaces(filepath)
                fix_file_structure(filepath)
            except Exception as e:
                print(f"Error fixing {filepath}: {e}")
    
    print("Fixed indentation issues")


if __name__ == '__main__':
    main()