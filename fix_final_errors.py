#!/usr/bin/env python
"""Fix the final remaining flake8 errors"""

import os
import re
from pathlib import Path


def fix_cli_imports():
    """Fix E402 module level import not at top of file in cli.py"""
    filepath = Path('src/risk_pipeline/cli.py')
    # Already fixed in previous edit
    pass


def fix_unused_imports():
    """Remove all unused imports from all files"""
    files_to_fix = {
        'src/risk_pipeline/utf8_fix.py': ['e'],  # Remove unused 'e' variable
        'src/risk_pipeline/config/schema.py': ['typing.List'],
        'src/risk_pipeline/core/base.py': ['sys', 'typing.Optional'],
        'src/risk_pipeline/core/config.py': ['typing.Optional'],
        'src/risk_pipeline/core/config_old.py': ['typing.List'],
        'src/risk_pipeline/core/data_processor.py': ['typing.List', 'typing.Dict', 'typing.Any', 'datetime.datetime', '.utils.Timer', '.utils.safe_print'],
        'src/risk_pipeline/core/feature_engineer.py': ['.utils.jeffreys_counts', '.utils.ks_statistic', '.utils.safe_print', 'typing.Tuple', 'typing.Optional', 'sklearn.feature_selection.SelectKBest', 'sklearn.feature_selection.f_classif'],
        'src/risk_pipeline/core/model_trainer.py': ['.utils.ks_table', 'typing.Tuple', 'typing.Optional'],
    }
    
    for filepath, unused_items in files_to_fix.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for item in unused_items:
                # Remove import statements
                content = re.sub(f'from .* import .*{re.escape(item)}.*\n', '', content)
                content = re.sub(f'import .*{re.escape(item)}.*\n', '', content)
            
            # Special case for utf8_fix.py - remove unused 'e' variable
            if 'utf8_fix.py' in filepath:
                content = content.replace('except Exception as e:', 'except Exception:')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")


def fix_monitoring_indentation():
    """Fix IndentationError in monitoring.py"""
    filepath = Path('src/risk_pipeline/monitoring.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 53 indentation issue
    fixed_lines = []
    for i, line in enumerate(lines):
        # Remove unexpected indents
        if i == 52 and line.startswith('    '):  # Line 53 (0-indexed)
            fixed_lines.append(line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    print("Fixed: monitoring.py indentation")


def fix_pipeline_indentation():
    """Fix IndentationError in pipeline.py"""
    filepath = Path('src/risk_pipeline/pipeline.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 2 indentation issue
    fixed_lines = []
    for i, line in enumerate(lines):
        if i == 1 and line.startswith('    '):  # Line 2 (0-indexed)
            fixed_lines.append(line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    print("Fixed: pipeline.py indentation")


def fix_w504_errors():
    """Fix W504 line break after binary operator"""
    filepath = Path('src/risk_pipeline/core/feature_engineer.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Move operators to end of previous line
    content = re.sub(r'\n\s*(\|\|)', r' \1\n', content)
    content = re.sub(r'\n\s*(\&\&)', r' \1\n', content)
    content = re.sub(r'\n\s*(\|)', r' \1\n', content)
    content = re.sub(r'\n\s*(\&)', r' \1\n', content)
    
    # Remove duplicate imports
    lines = content.split('\n')
    seen_imports = set()
    fixed_lines = []
    for line in lines:
        if 'from sklearn.feature_selection import' in line:
            if line not in seen_imports:
                fixed_lines.append(line)
                seen_imports.add(line)
        else:
            fixed_lines.append(line)
    
    # Remove unused variable vw
    content = '\n'.join(fixed_lines)
    content = re.sub(r'vw = .*\n', '', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed: feature_engineer.py W504 and duplicates")


def main():
    """Main function"""
    print("Fixing final errors...")
    
    fix_unused_imports()
    fix_monitoring_indentation()
    fix_pipeline_indentation()
    fix_w504_errors()
    
    print("\nAll fixes completed!")


if __name__ == '__main__':
    main()