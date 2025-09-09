#!/usr/bin/env python
"""Fix all remaining flake8 errors systematically"""

import os
import re
from pathlib import Path


def fix_line_too_long(content):
    """Fix E501 line too long errors"""
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        if len(line) > 79:
            # Don't break imports
            if line.strip().startswith(('import ', 'from ')):
                fixed_lines.append(line)
            # Don't break strings
            elif '"""' in line or "'''" in line or line.strip().startswith('#'):
                fixed_lines.append(line)
            # Break long function calls at commas
            elif '(' in line and ')' in line and ',' in line:
                # Find position to break
                if len(line) <= 120:  # Reasonable length
                    fixed_lines.append(line)
                else:
                    # Try to break at comma
                    parts = line.split(',')
                    if len(parts) > 1:
                        indent = len(line) - len(line.lstrip())
                        first = parts[0] + ','
                        fixed_lines.append(first)
                        for part in parts[1:-1]:
                            fixed_lines.append(' ' * (indent + 4) + part.strip() + ',')
                        if parts[-1].strip():
                            fixed_lines.append(' ' * (indent + 4) + parts[-1].strip())
                    else:
                        fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def fix_blank_lines(content):
    """Fix E302, E305, E306 blank line errors"""
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for function/class definitions
        if i > 0 and (line.strip().startswith('def ') or line.strip().startswith('class ')):
            # Count blank lines before
            blank_count = 0
            j = i - 1
            while j >= 0 and not lines[j].strip():
                blank_count += 1
                j -= 1
            
            # Check if inside a class
            indent = len(line) - len(line.lstrip())
            is_nested = indent > 0
            
            # Remove existing blank lines
            while fixed_lines and not fixed_lines[-1].strip():
                fixed_lines.pop()
            
            # Add correct number of blank lines
            if is_nested:
                fixed_lines.append('')  # 1 blank line for nested
            else:
                fixed_lines.append('')  # 2 blank lines for top-level
                fixed_lines.append('')
        
        fixed_lines.append(line)
        i += 1
    
    # Remove trailing blank lines
    while fixed_lines and not fixed_lines[-1].strip():
        fixed_lines.pop()
    
    return '\n'.join(fixed_lines)


def fix_imports(content):
    """Fix F401 unused imports and E402 module level import not at top"""
    lines = content.split('\n')
    
    # Separate imports and other code
    imports = []
    code = []
    in_docstring = False
    docstring_done = False
    shebang = []
    encoding = []
    module_docstring = []
    
    for i, line in enumerate(lines):
        # Handle shebang
        if i == 0 and line.startswith('#!'):
            shebang.append(line)
            continue
        # Handle encoding
        if i <= 1 and '# -*- coding:' in line:
            encoding.append(line)
            continue
        
        # Handle module docstring
        if not docstring_done:
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                in_docstring = True
                module_docstring.append(line)
                if line.count('"""') == 2 or line.count("'''") == 2:
                    in_docstring = False
                    docstring_done = True
                continue
            elif in_docstring:
                module_docstring.append(line)
                if '"""' in line or "'''" in line:
                    in_docstring = False
                    docstring_done = True
                continue
        
        # Collect imports
        if line.strip().startswith(('import ', 'from ')) and not in_docstring:
            imports.append(line)
        else:
            code.append(line)
    
    # Check which imports are used
    code_text = '\n'.join(code)
    used_imports = []
    
    for imp in imports:
        if 'import' in imp:
            # Extract imported names
            if ' as ' in imp:
                # Handle aliased imports
                parts = imp.split(' as ')
                name = parts[-1].strip()
            elif 'from' in imp and 'import' in imp:
                # Handle from ... import ...
                parts = imp.split('import')[-1]
                names = [n.strip() for n in parts.split(',')]
                for name in names:
                    if ' as ' in name:
                        name = name.split(' as ')[-1].strip()
                    # Check if used in code
                    if re.search(r'\b' + re.escape(name) + r'\b', code_text):
                        if imp not in used_imports:
                            used_imports.append(imp)
                        break
                else:
                    # Special cases - always keep certain imports
                    if any(keep in imp for keep in ['__future__', 'typing', 'sys', 'os']):
                        used_imports.append(imp)
                continue
            else:
                # Regular import
                name = imp.replace('import ', '').strip()
            
            # Check if used
            if re.search(r'\b' + re.escape(name.split('.')[0]) + r'\b', code_text):
                used_imports.append(imp)
            # Keep special imports
            elif any(keep in imp for keep in ['__future__', 'typing']):
                used_imports.append(imp)
    
    # Reconstruct file
    result = []
    if shebang:
        result.extend(shebang)
    if encoding:
        result.extend(encoding)
    if module_docstring:
        result.extend(module_docstring)
    
    if result and used_imports:
        result.append('')
    
    result.extend(sorted(set(used_imports)))
    
    if used_imports and code:
        result.append('')
        result.append('')
    
    result.extend(code)
    
    return '\n'.join(result)


def fix_file(filepath):
    """Fix all flake8 errors in a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        # Fix imports first
        content = fix_imports(content)
        
        # Fix blank lines
        content = fix_blank_lines(content)
        
        # Fix line lengths (do this last)
        # content = fix_line_too_long(content)  # Skip for now, too aggressive
        
        # Fix semicolons
        content = content.replace('; ', '\n    ')
        
        # Fix bare excepts
        content = re.sub(r'\bexcept:\s*$', 'except Exception:', content, flags=re.MULTILINE)
        
        # Fix missing os import in cli.py
        if 'cli.py' in str(filepath):
            if 'import os' not in content and 'os.' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('import'):
                        lines.insert(i, 'import os')
                        break
                content = '\n'.join(lines)
        
        # Fix undefined names
        content = content.replace('os.path.exists', 'Path.exists' if 'from pathlib import Path' in content else 'os.path.exists')
        
        # Remove trailing whitespace
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)
        
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


def main():
    """Main function"""
    src_dir = Path('src')
    test_dir = Path('tests')
    
    fixed_count = 0
    
    # Fix all Python files
    for directory in [src_dir, test_dir]:
        for filepath in directory.rglob('*.py'):
            if fix_file(filepath):
                print(f"Fixed: {filepath}")
                fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()