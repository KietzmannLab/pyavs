# PyAVS Plotting Style Guide

## Core Principles

This document defines the plotting style conventions for the pyAVS codebase.

## Style Rules

### 1. Figure Layout
- **Single plots only** - No subplots or multi-panel figures
- Each plot should be saved as a separate file
- Use `plt.figure()` instead of `fig, ax = plt.subplots()`

### 2. Text Annotations
- **No plt.text() calls** - Never add text annotations to plots
- Let the axes labels and legends speak for themselves
- Statistics should go in figure captions, not on the plot

### 3. Units
- **Always use square brackets** - `[°]`, `[ms]`, `[count]`, `[a.u.]`
- Never use parentheses for units - `(°)` is incorrect
- Example: `'Response Time [ms]'` not `'Response Time (ms)'`

### 4. Font Sizes
- **Never define font sizes** - Always use seaborn context defaults
- No `fontsize=`, `fontweight=`, or font property specifications
- Use `sns.set_context("poster")` for publication figures
- Trust the seaborn defaults for consistency

### 5. Style Setup
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Standard setup for all plots
sns.set_context("poster")  # Large fonts for publications
sns.set_style("white")      # Clean background
```

### 6. Minimal Design
- Clean, uncluttered plots
- Minimal colors (2-3 max)
- White background
- Grid lines only where helpful (`alpha=0.3`)

## Example

```python
# CORRECT
sns.set_context("poster")
sns.set_style("white")

plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='condition', y='value')
plt.ylabel('Response Time [ms]')
plt.xlabel('Condition')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.close()
```

```python
# INCORRECT - DO NOT DO THIS
fig, (ax1, ax2) = plt.subplots(1, 2)  # No subplots!

ax1.text(0.5, 0.5, 'Statistics here')  # No text!
ax1.set_ylabel('Response Time (ms)', fontsize=14)  # Wrong units! No fontsize!
```

## Rationale

These conventions ensure:
- Consistency across all figures
- Publication-ready quality
- Easy reusability in papers (single figures)
- Clean, professional appearance
- Minimal maintenance (no hardcoded sizes)
