# Extract Phase Signal

This small utility reads a 1‑D signal image (e.g. a shroud-derived signal), applies optional preprocessing (moving average and unsharp masking), extracts the phase according to a chosen model, and writes the resulting phase as a plain-text file.


```bash
# Basic extraction (default model: LINEAR_BETWEEN_MINIMA)
rtkextractphasesignal -i signal.mha -o phase.txt

# Tune preprocessing parameters
rtkextractphasesignal -i signal.mha -o phase.txt --movavg 3 --unsharp 53

# Use LOCAL_PHASE model
rtkextractphasesignal -i signal.mha -o phase.txt --model LOCAL_PHASE
```
