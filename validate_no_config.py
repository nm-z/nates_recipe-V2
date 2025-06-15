#!/usr/bin/env python3
"""
Configuration Violation Detector
=================================
Run this before committing to ensure you haven't violated the no-configuration principle.

Usage: python validate_no_config.py
"""

import sys
from pathlib import Path
import re

def check_forbidden_imports():
    """Check for forbidden imports that enable user configuration"""
    violations = []
    python_files = list(Path.cwd().glob("*.py"))
    forbidden_imports = ["argparse", "click", "fire", "typer", "configparser"]
    
    for py_file in python_files:
        if py_file.name.startswith("test_") or py_file.name == "validate_no_config.py":
            continue
            
        try:
            content = py_file.read_text()
            for forbidden in forbidden_imports:
                if f"import {forbidden}" in content or f"from {forbidden}" in content:
                    violations.append(f"❌ {py_file.name}: imports {forbidden} (enables user configuration)")
        except Exception as e:
            print(f"⚠️  Could not read {py_file.name}: {e}")
    
    return violations

def check_config_files():
    """Check for configuration files"""
    violations = []
    current_dir = Path.cwd()
    
    forbidden_patterns = [
        "*.config", "*.cfg", "*.yaml", "*.yml", "*.json", "*.ini", "*.toml",
        "config.*", "settings.*", "preferences.*", ".env*"
    ]
    
    legitimate_files = {
        "requirements.txt", ".gitignore", "package.json", "pyproject.toml",
        ".github/workflows/ci.yml"  # CI configuration is allowed
    }
    
    for pattern in forbidden_patterns:
        for config_file in current_dir.glob(pattern):
            if config_file.name not in legitimate_files and not config_file.name.startswith(".git"):
                violations.append(f"❌ Found config file: {config_file.name}")
    
    return violations

def check_user_interaction():
    """Check for user interaction code"""
    violations = []
    python_files = list(Path.cwd().glob("*.py"))
    
    for py_file in python_files:
        if py_file.name.startswith("test_") or py_file.name == "validate_no_config.py":
            continue
            
        try:
            content = py_file.read_text()
            
            # Check for interactive input statements
            input_pattern = "input" + "("
            if input_pattern in content:
                violations.append(f"❌ {py_file.name}: contains call to input function")
            
            # Check for environment variable configuration
            env_patterns = [
                r'os\.getenv\(["\'](?:MODEL|DATASET|TARGET|CONFIG)',
                r'os\.environ\.get\(["\'](?:MODEL|DATASET|TARGET|CONFIG)'
            ]
            
            for pattern in env_patterns:
                if re.search(pattern, content):
                    violations.append(f"❌ {py_file.name}: uses environment variables for configuration")
                    
        except Exception as e:
            print(f"⚠️  Could not read {py_file.name}: {e}")
    
    return violations

def check_command_line_args():
    """Check for command-line argument parsing"""
    violations = []
    python_files = list(Path.cwd().glob("*.py"))
    
    for py_file in python_files:
        if py_file.name.startswith("test_") or py_file.name == "validate_no_config.py":
            continue
            
        try:
            content = py_file.read_text()
            
            # Check for sys.argv usage (beyond just script name)
            if "sys.argv[1" in content or "len(sys.argv)" in content:
                violations.append(f"❌ {py_file.name}: processes command-line arguments")
                
        except Exception as e:
            print(f"⚠️  Could not read {py_file.name}: {e}")
    
    return violations

def check_hardcoded_dataset():
    """Verify dataset numbers are hardcoded"""
    violations = []
    
    try:
        import battle_tested_optuna_playbook
        if not hasattr(battle_tested_optuna_playbook, 'DATASET'):
            violations.append("❌ battle_tested_optuna_playbook.py: missing hardcoded DATASET")
        elif not isinstance(battle_tested_optuna_playbook.DATASET, int):
            violations.append("❌ battle_tested_optuna_playbook.py: DATASET is not an integer")
        elif battle_tested_optuna_playbook.DATASET not in [1, 2, 3]:
            violations.append(f"❌ battle_tested_optuna_playbook.py: DATASET={battle_tested_optuna_playbook.DATASET} not in [1,2,3]")
    except ImportError as e:
        violations.append(f"❌ Could not import battle_tested_optuna_playbook: {e}")
    
    return violations

def main():
    """Run all configuration violation checks"""
    print("🔍 Checking for configuration violations...")
    print("=" * 60)
    
    all_violations = []
    
    # Run all checks
    checks = [
        ("Forbidden imports", check_forbidden_imports),
        ("Configuration files", check_config_files),
        ("User interaction", check_user_interaction),
        ("Command-line arguments", check_command_line_args),
        ("Hardcoded dataset", check_hardcoded_dataset)
    ]
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}...")
        violations = check_func()
        if violations:
            all_violations.extend(violations)
            for violation in violations:
                print(f"  {violation}")
        else:
            print(f"  ✅ No violations found")
    
    print("\n" + "=" * 60)
    
    if all_violations:
        print(f"💥 CONFIGURATION VIOLATIONS DETECTED: {len(all_violations)}")
        print("\n🚨 YOUR CHANGES VIOLATE THE NO-CONFIGURATION PRINCIPLE!")
        print("\nRead AGENTS.md and fix these issues before committing:")
        for violation in all_violations:
            print(f"  {violation}")
        print("\n⚠️  Remember: 2 CSVs in → Model out. NO CONFIGURATION ALLOWED.")
        return 1
    else:
        print("🎉 NO CONFIGURATION VIOLATIONS DETECTED!")
        print("✅ Your changes respect the no-configuration principle.")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 