#!/usr/bin/env python
"""Remove all unused imports and fix remaining issues"""

import re
from pathlib import Path


def remove_unused_imports(filepath, unused_list):
    """Remove specific unused imports from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            should_keep = True
            for unused in unused_list:
                if f'import {unused}' in line or f'import {unused},' in line:
                    should_keep = False
                    break
                if f', {unused}' in line and 'import' in line:
                    # Remove just this import from the line
                    line = line.replace(f', {unused}', '')
                    line = line.replace(f'{unused}, ', '')
            
            if should_keep:
                fixed_lines.append(line)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        return True
    except Exception as e:
        print(f"Error in {filepath}: {e}")
        return False


def fix_all_unused():
    """Fix all unused imports"""
    files_to_fix = {
        'src/risk_pipeline/config/schema.py': ['List'],
        'src/risk_pipeline/core/base.py': ['Optional'],
        'src/risk_pipeline/core/config.py': ['Optional'],
        'src/risk_pipeline/core/config_old.py': ['List'],
        'src/risk_pipeline/core/data_processor.py': ['List', 'Dict', 'Any', 'datetime', 'Timer', 'safe_print'],
        'src/risk_pipeline/core/model_trainer.py': ['ks_table', 'Tuple', 'Optional'],
        'src/risk_pipeline/core/utils.py': ['os', 'sys', 'field', 'json'],
        'src/risk_pipeline/model/train.py': ['Tuple', 'np'],
        'src/risk_pipeline/model/versioning.py': ['np', 'precision_score', 'recall_score', 'f1_score'],
        'src/risk_pipeline/reporting/shap_utils.py': ['pd'],
    }
    
    for filepath, unused_list in files_to_fix.items():
        if remove_unused_imports(Path(filepath), unused_list):
            print(f"Fixed: {filepath}")


def fix_indentation_errors():
    """Fix specific indentation errors"""
    # Fix monitoring.py line 56
    filepath = Path('src/risk_pipeline/monitoring.py')
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Fix by ensuring consistent indentation
        lines = content.split('\n')
        for i in range(len(lines)):
            if i == 55:  # Line 56 (0-indexed)
                # Ensure proper indentation
                lines[i] = lines[i].lstrip()
                if lines[i]:
                    lines[i] = '    ' + lines[i]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print("Fixed monitoring.py indentation")
    
    # Fix pipeline.py line 3
    filepath = Path('src/risk_pipeline/pipeline.py')
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) > 2:
            lines[2] = lines[2].lstrip() + '\n' if lines[2].strip() else '\n'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Fixed pipeline.py indentation")
    
    # Fix feature_engineer.py line 421
    filepath = Path('src/risk_pipeline/core/feature_engineer.py')
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) > 420:
            lines[420] = '                ' + lines[420].lstrip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Fixed feature_engineer.py indentation")


def fix_long_lines():
    """Fix lines that are too long"""
    filepath = Path('src/risk_pipeline/core/utils.py')
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i in range(len(lines)):
            if len(lines[i]) > 120:
                # Break long lines at commas or operators
                if ',' in lines[i]:
                    parts = lines[i].split(',', 1)
                    if len(parts) == 2:
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        lines[i] = parts[0] + ',\n' + ' ' * (indent + 4) + parts[1].lstrip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("Fixed long lines in utils.py")


def main():
    """Main function"""
    print("Removing all unused imports and fixing errors...")
    
    fix_all_unused()
    fix_indentation_errors()
    fix_long_lines()
    
    print("\nAll fixes completed!")


if __name__ == '__main__':
    main()