#!/usr/bin/env python
"""Fix remaining flake8 errors more carefully"""

import os
import re
from pathlib import Path


def fix_blank_lines_carefully(filepath):
    """Fix E302, E305, E306 blank line errors more carefully"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    inside_class = False
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Track if we're inside a class
        if stripped.startswith('class '):
            inside_class = True
            # Ensure 2 blank lines before class (unless at start of file)
            if i > 0 and fixed_lines:
                # Count existing blank lines
                blank_count = 0
                j = len(fixed_lines) - 1
                while j >= 0 and fixed_lines[j].strip() == '':
                    blank_count += 1
                    j -= 1
                
                # Add blank lines if needed
                while blank_count < 2:
                    fixed_lines.append('\n')
                    blank_count += 1
        
        elif stripped.startswith('def '):
            # Check indentation to see if it's a method or function
            indent = len(line) - len(line.lstrip())
            if indent > 0:  # Method inside class
                # Ensure 1 blank line before method
                if i > 0 and fixed_lines and fixed_lines[-1].strip() != '':
                    fixed_lines.append('\n')
            else:  # Top-level function
                inside_class = False
                # Ensure 2 blank lines before function
                if i > 0 and fixed_lines:
                    blank_count = 0
                    j = len(fixed_lines) - 1
                    while j >= 0 and fixed_lines[j].strip() == '':
                        blank_count += 1
                        j -= 1
                    
                    while blank_count < 2:
                        fixed_lines.append('\n')
                        blank_count += 1
        
        fixed_lines.append(line)
        i += 1
    
    # Remove trailing blank lines
    while fixed_lines and fixed_lines[-1].strip() == '':
        fixed_lines.pop()
    
    # Add single newline at end if missing
    if fixed_lines and not fixed_lines[-1].endswith('\n'):
        fixed_lines[-1] += '\n'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_unused_imports(filepath):
    """Remove unused imports (F401)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find all imports
    imports_to_check = []
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')):
            imports_to_check.append((i, line))
    
    # Check which are used
    code_without_imports = []
    for i, line in enumerate(lines):
        if not line.strip().startswith(('import ', 'from ')):
            code_without_imports.append(line)
    code_text = '\n'.join(code_without_imports)
    
    lines_to_remove = set()
    for line_num, import_line in imports_to_check:
        # Extract what was imported
        if 'from ' in import_line and ' import ' in import_line:
            # from X import Y, Z
            parts = import_line.split(' import ')[-1]
            names = [n.strip().split(' as ')[0] for n in parts.split(',')]
            
            # Check if any are used
            used = False
            for name in names:
                if name in ['Path', 'Optional', 'Dict', 'Any', 'Tuple', 'List']:
                    # Keep type hints
                    used = True
                    break
                if re.search(r'\b' + re.escape(name) + r'\b', code_text):
                    used = True
                    break
            
            if not used and 'Path' not in import_line:
                lines_to_remove.add(line_num)
        
        elif ' import ' in import_line and ' as ' not in import_line:
            # import X
            module = import_line.replace('import ', '').strip()
            if not re.search(r'\b' + re.escape(module.split('.')[0]) + r'\b', code_text):
                if module not in ['os', 'sys', 'typing']:
                    lines_to_remove.add(line_num)
    
    # Remove unused imports
    new_lines = []
    for i, line in enumerate(lines):
        if i not in lines_to_remove:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def fix_semicolons(filepath):
    """Fix E702 multiple statements on one line (semicolon)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line in lines:
        # Replace semicolons with newlines, preserving indentation
        if ';' in line and not line.strip().startswith('#'):
            indent = len(line) - len(line.lstrip())
            parts = line.split(';')
            fixed_lines.append(parts[0] + '\n')
            for part in parts[1:]:
                if part.strip():
                    fixed_lines.append(' ' * indent + part.strip() + '\n')
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_bare_except(filepath):
    """Fix E722 bare except"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'\bexcept:\s*$', 'except Exception:', content, flags=re.MULTILINE)
    content = re.sub(r'\bexcept:\s*#', 'except Exception:  #', content, flags=re.MULTILINE)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    """Main function to fix all errors"""
    src_dir = Path('src')
    test_dir = Path('tests')
    
    print("Fixing blank line errors...")
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                fix_blank_lines_carefully(filepath)
            except Exception as e:
                print(f"Error fixing blank lines in {filepath}: {e}")
    
    print("Fixing unused imports...")
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                fix_unused_imports(filepath)
            except Exception as e:
                print(f"Error fixing imports in {filepath}: {e}")
    
    print("Fixing semicolons...")
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                fix_semicolons(filepath)
            except Exception as e:
                print(f"Error fixing semicolons in {filepath}: {e}")
    
    print("Fixing bare except...")
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            try:
                fix_bare_except(filepath)
            except Exception as e:
                print(f"Error fixing bare except in {filepath}: {e}")
    
    print("Done!")


if __name__ == '__main__':
    main()