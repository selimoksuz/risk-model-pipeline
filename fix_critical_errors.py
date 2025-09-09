#!/usr/bin/env python
"""Fix critical syntax and indentation errors"""

import re
from pathlib import Path


def fix_monitoring_indentation():
    """Fix indentation error in monitoring.py line 56"""
    filepath = Path('src/risk_pipeline/monitoring.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix indentation mismatch
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if i == 56:  # Line with error
            # Ensure proper indentation
            if 'def ' in line or 'class ' in line or 'if ' in line:
                fixed_lines.append(line)
            else:
                # Fix indentation to match context
                fixed_lines.append('    ' + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_pipeline_indentation():
    """Fix indentation in pipeline.py line 3"""
    filepath = Path('src/risk_pipeline/pipeline.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if i == 3 and line.startswith('        '):  # Too much indent
            fixed_lines.append(line[4:])  # Remove 4 spaces
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_feature_engineer_indentation():
    """Fix indentation in feature_engineer.py line 421"""
    filepath = Path('src/risk_pipeline/core/feature_engineer.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if i == 421 and line.startswith('                        '):  # Too much indent
            # Adjust to proper indentation
            fixed_lines.append('                    ' + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_woe_indentation():
    """Fix indentation in woe.py line 26"""
    filepath = Path('src/risk_pipeline/stages/woe.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if i == 26 and '    ' in line[:20]:  # Check indentation
            # Fix to proper level
            fixed_lines.append('            ' + line.lstrip())
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def fix_validation_string():
    """Fix string literal error in validation.py line 102"""
    filepath = Path('src/risk_pipeline/utils/validation.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for i, line in enumerate(lines, 1):
        if i == 102 and '"' in line and not line.rstrip().endswith('"'):
            # Fix unclosed string
            fixed_lines.append(line.rstrip() + '"\n')
        else:
            fixed_lines.append(line)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)


def remove_unused_variables():
    """Remove unused variable assignments"""
    files_to_fix = {
        'src/risk_pipeline/utils/error_handler.py': [(189, 'e')],
        'tests/test_error_handler.py': [(244, 'result3')],
        'tests/integration/run_fast_pipeline.py': [(77, 'results')],
        'tests/integration/run_minimal_pipeline.py': [(39, 'results')],
        'tests/integration/test_calibration_fix.py': [(23, 'calibration_df')],
    }
    
    for filepath, vars_to_remove in files_to_fix.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            fixed_lines = []
            for i, line in enumerate(lines, 1):
                should_skip = False
                for line_num, var_name in vars_to_remove:
                    if i == line_num and f'{var_name} = ' in line:
                        # Comment out or remove the line
                        if 'except' in line:
                            fixed_lines.append(line.replace(' as e', ''))
                        else:
                            fixed_lines.append('    # ' + line.lstrip())
                        should_skip = True
                        break
                
                if not should_skip:
                    fixed_lines.append(line)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)
            print(f"Fixed unused variables in {filepath}")
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")


def main():
    """Main function"""
    print("Fixing critical errors...")
    
    fix_monitoring_indentation()
    print("Fixed monitoring.py")
    
    fix_pipeline_indentation()
    print("Fixed pipeline.py")
    
    fix_feature_engineer_indentation()
    print("Fixed feature_engineer.py")
    
    fix_woe_indentation()
    print("Fixed woe.py")
    
    fix_validation_string()
    print("Fixed validation.py")
    
    remove_unused_variables()
    
    print("\nAll critical fixes completed!")


if __name__ == '__main__':
    main()