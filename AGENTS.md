# Team Development Guidelines for Nate's Recipe Optimization Pipeline

## 🎯 Core Principle: NO USER CONFIGURATION

**CRITICAL**: This pipeline follows a strict "2 CSVs in → Model out" philosophy. 

### ❌ ABSOLUTELY FORBIDDEN:
- **Configuration files** (.config, .cfg, .yaml, .json, .ini, .toml)
- **User prompts** or interactive input of any kind
- **Command-line arguments** for data paths, model selection, or parameters
- **Settings files** or user preferences
- **Environment variables** for configuration
- **Flags or toggles** that users must set
- **Optional parameters** that change behavior
- **"Choose your model" options**
- **"Set your target R²" inputs**
- **"Configure preprocessing" options**

### ✅ REQUIRED APPROACH:
- **Hardcoded dataset numbers** (1, 2, or 3)
- **Automatic operation** - no human decisions required
- **Sensible defaults** built into the code
- **Zero-configuration execution**
- **Deterministic behavior** - same inputs = same outputs

### 🚨 ENFORCEMENT:
**ANY PR that introduces user configuration will be IMMEDIATELY REJECTED.**

**Why?** This eliminates configuration complexity and ensures reproducible, automated operation. The moment you add "user choice," you've broken the core principle.

## 🏗️ Architecture Overview

### Core Components
- **`battle_tested_optuna_playbook.py`** - Main production pipeline
- **`auto_optuna-V1.x.py`** - Evolution of optimization approaches
- **`test_best_model.py`** - Model validation and testing
- **Custom Transformers** - KMeans, IsolationForest, LOF outlier detection

### Data Flow
```
CSV Files → Preprocessing → Feature Selection → Model Training → Optuna Optimization → Best Model
```

### Key Features
- **Rich Tree Logging** - Hierarchical, structured output
- **12-thread Cross-Validation** - With single-thread models to prevent crashes
- **No Fixed R² Targets** - Dynamic optimization approaching noise ceiling
- **Comprehensive Artifact Saving** - All models, logs, metrics preserved

## 🧪 Testing Strategy

### Automated Tests (pytest)
```bash
# Run all tests
pytest test_pipeline.py -v

# Run only fast tests
pytest test_pipeline.py -v -m "not slow"

# Run with coverage
pytest test_pipeline.py --cov=. --cov-report=html
```

### Test Categories

#### 1. Unit Tests ⚡ (Fast - Always Run)
- **Outlier Transformers** - KMeans, IsolationForest, LOF functionality
- **Data Validation** - CSV loading, type checking, shape validation
- **Configuration Tests** - Verify no config files required
- **Error Handling** - Invalid inputs, edge cases

#### 2. Integration Tests 🔄 (Slow - CI Only)
- **End-to-End Pipeline** - Complete training run with minimal trials
- **Model Persistence** - Save/load functionality
- **Performance Tests** - Memory usage, prediction speed

#### 3. Manual Testing 👨‍💻 (Human Required)
- **Full Training Runs** - Complete optimization with real data
- **Model Quality Assessment** - R² scores, residual analysis
- **Resource Usage** - Long-term memory leaks, CPU utilization
- **Cross-Platform Testing** - Windows/Mac/Linux compatibility

### CI/CD Pipeline
- **GitHub Actions** - Automated testing on push/PR
- **Multi-Python Support** - 3.8, 3.9, 3.10, 3.11
- **Cross-Platform** - Ubuntu, Windows, macOS
- **Code Quality** - Black, isort, flake8, mypy
- **Security Scanning** - Safety, bandit

## 📝 Pull Request Guidelines

### Before Creating a PR

1. **Run Local Tests**
   ```bash
   # Check for configuration violations FIRST
   python validate_no_config.py
   
   # Then run tests
   pytest test_pipeline.py -v
   python -m py_compile *.py  # Check syntax
   ```

2. **Verify Core Principle**
   - Ensure NO user configuration is introduced
   - Confirm hardcoded dataset numbers remain
   - Test that pipeline runs without user input

3. **Check Dependencies**
   ```bash
   pip install -r requirements.txt
   python -c "import battle_tested_optuna_playbook; print('✅ Imports work')"
   ```

### PR Requirements

#### ✅ Must Have
- **Clear Description** - What does this change do?
- **Testing Evidence** - Show that tests pass
- **No Breaking Changes** - Existing functionality preserved
- **Rich Tree Logging** - Maintain structured output format
- **Thread Safety** - 12-thread CV, 1-thread models

#### ❌ Will Be IMMEDIATELY REJECTED
- **User Configuration** - ANY form of config files, user prompts, flags, or choices
- **Interactive Features** - Anything requiring user input during execution
- **Optional Parameters** - Features that can be "turned on/off" by users
- **Model Selection Menus** - "Choose your algorithm" type features
- **Preprocessing Options** - "Configure your pipeline" type features
- **Breaking Changes** - Without explicit approval
- **Reduced Functionality** - Removing existing features
- **Poor Performance** - Significant speed/memory regressions
- **No Tests** - Changes without corresponding tests

#### 🚨 COMMON VIOLATIONS TO WATCH FOR:
- Adding `argparse` or `click` for command-line options
- Creating config.yaml, settings.json, or similar files
- Adding `input()` statements or user prompts
- Environment variable checks like `os.getenv("MODEL_TYPE")`
- Conditional logic based on user preferences
- "If you want X, set flag Y" type features

### PR Template
```markdown
## Summary
Brief description of changes

## Testing
- [ ] Unit tests pass: `pytest test_pipeline.py -m "not slow"`
- [ ] Integration tests pass: `pytest test_pipeline.py -m "slow"`
- [ ] Manual testing completed
- [ ] **ZERO user configuration introduced** (no config files, flags, prompts, or choices)
- [ ] Pipeline runs with ZERO user input required
- [ ] No argparse, click, or command-line argument parsing added
- [ ] No environment variable dependencies added

## Performance Impact
- Memory usage: [No change/Improved/Degraded by X%]
- Speed: [No change/Improved/Degraded by X%]

## Breaking Changes
- [ ] None
- [ ] Yes (requires approval)
```

## 🔧 Development Setup

### Environment Setup
```bash
# Clone repository
git clone <repo-url>
cd nates_recipe-V2

# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
# ml_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python -c "import battle_tested_optuna_playbook; print('✅ Setup complete')"
```

### Code Style
- **Line Length** - 127 characters max
- **Imports** - Use isort for organization
- **Formatting** - Black for consistent style
- **Type Hints** - Encouraged but not required
- **Docstrings** - Required for public functions

### Logging Standards
```python
# ✅ Good - Rich Tree Logging
from rich.tree import Tree
from rich.console import Console

console = Console()
tree = Tree("🔥 Phase 1: Data Loading")
tree.add("✅ Loaded predictors: 1000 samples")
tree.add("✅ Loaded targets: 1000 samples")
console.print(tree)

# ❌ Bad - Plain print statements
print("Loading data...")
print("Loaded 1000 samples")
```

## 🚀 Performance Guidelines

### Threading Configuration
```python
# ✅ Correct - Prevents system crashes
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, n_jobs=12)  # 12 threads for CV

# Individual models should use n_jobs=1
model = GradientBoostingRegressor(n_estimators=100, n_jobs=1)  # Single thread
```

### Memory Management
- Use `np.float32` for large datasets
- Clean up temporary variables
- Monitor memory usage in long-running processes
- Use generators for large data processing

### Optimization Principles
- No fixed R² targets - let optimization find natural ceiling
- Use Optuna's pruning for efficiency
- Save all artifacts for reproducibility
- Fail fast on errors

## 🐛 Debugging Guidelines

### Common Issues

1. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print(sys.path)"
   
   # Verify dependencies
   pip list | grep -E "(optuna|sklearn|pandas|numpy)"
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
   ```

3. **Threading Problems**
   ```bash
   # Check CPU count
   python -c "import os; print(f'CPU count: {os.cpu_count()}')"
   ```

### Debugging Tools
- **Rich Console** - For structured output
- **pytest -v** - Verbose test output
- **pytest --pdb** - Drop into debugger on failure
- **Memory Profiler** - For memory leak detection

## 📊 Monitoring & Metrics

### Key Metrics to Track
- **R² Scores** - Model performance
- **Training Time** - Optimization efficiency  
- **Memory Usage** - Resource consumption
- **Feature Count** - Before/after preprocessing
- **Trial Success Rate** - Optuna optimization health

### Logging Requirements
- All phases must use Rich Tree logging
- Save detailed logs to files
- Include timestamps and performance metrics
- Log hyperparameters and model choices

## 🔒 Security & Best Practices

### Data Handling
- Never commit CSV data files to git
- Use `.gitignore` for data directories
- Validate input data types and ranges
- Handle missing/invalid data gracefully

### Code Security
- No hardcoded secrets or API keys
- Validate all external inputs
- Use safe file operations
- Follow principle of least privilege

## 📚 Resources

### Documentation
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Rich Documentation](https://rich.readthedocs.io/)

### Internal References
- `results.md` - Performance benchmarks
- `test_best_model.py` - Model validation examples
- `battle_tested_optuna_playbook.py` - Production pipeline reference

---

## 🛑 FOR PERSISTENT CONFIGURATION VIOLATORS

### If Team Members Keep Adding User Configuration:

**STOP. READ THIS.**

You are violating the core principle of this project. Here's what happens next:

1. **Your PR will be REJECTED immediately**
2. **You will be asked to read this document again**
3. **You will be required to explain why configuration is forbidden**
4. **Repeated violations may result in restricted repository access**

### Common Excuses and Why They're Wrong:

❌ **"But users might want to choose different models"**
→ NO. The pipeline automatically finds the best model.

❌ **"But users might want different preprocessing"**
→ NO. The pipeline automatically optimizes preprocessing.

❌ **"But users might want to set their own R² target"**
→ NO. The pipeline automatically approaches the noise ceiling.

❌ **"But this would make it more flexible"**
→ NO. Flexibility = Configuration = FORBIDDEN.

❌ **"But other ML libraries do this"**
→ NO. This is NOT other libraries. This is a zero-config pipeline.

❌ **"But it's just a small flag"**
→ NO. There are no "small" configuration violations.

### The ONLY Acceptable Changes:

✅ **Bug fixes** that don't add configuration
✅ **Performance improvements** that don't add configuration  
✅ **New algorithms** that are automatically selected
✅ **Better preprocessing** that happens automatically
✅ **Documentation** improvements
✅ **Test** additions

### If You Don't Understand Why:

**Re-read the entire AGENTS.md file until you understand that this pipeline is designed to be ZERO-CONFIGURATION by design. This is not a limitation - it's the core feature.**

### 🛠️ Configuration Violation Detector

**BEFORE EVERY COMMIT, RUN:**
```bash
python validate_no_config.py
```

This script will automatically detect:
- ❌ Forbidden imports (argparse, click, etc.)
- ❌ Configuration files (.yaml, .json, etc.)
- ❌ User interaction code (input() statements)
- ❌ Environment variable configuration
- ❌ Command-line argument processing

**If this script finds violations, FIX THEM before committing.**

---

## ⚠️ Remember: 2 CSVs in → Model out. Nothing more, nothing less.

Any deviation from this principle will result in PR rejection. Keep it simple, keep it automated, keep it working.

**If you add configuration, you have fundamentally misunderstood the project.**

### Code Quality Requirements
- **Pylint Score**: Must achieve > 9.0/10 before PR approval
- **Pylint Configuration**: Use project .pylintrc file
- **CI Enforcement**: PRs automatically rejected if pylint score < 9.0
