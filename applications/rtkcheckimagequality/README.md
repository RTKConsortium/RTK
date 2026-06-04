# Quality Checker

Checks the MSE of reconstructed volume(s) against reference volume(s).

This utility compares one or more reconstructed images to reference image(s) and fails if the computed error exceeds the provided threshold(s).

```bash
# Single comparison
rtkcheckimagequality -i reference.mha -j reconstruction.mha -t 1e-3

# Multiple one-to-one comparisons (per-file thresholds)
rtkcheckimagequality -i ref1.mha ref2.mha -j rec1.mha rec2.mha -t 1e-3 2e-3

# Broadcast a single threshold to multiple comparisons
rtkcheckimagequality -i ref1.mha ref2.mha -j rec1.mha rec2.mha -t 2e-3
```
