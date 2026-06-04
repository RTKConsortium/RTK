# Lag correction

Fourth-order linear time invariant (LTI) lag correction. See [\[Hsieh, Proceedings of SPIE, 2000\]](https://doi.org/10.1117/12.384505).

```bash
# Replace the -a (rates) and -c (coefficients) values with calibrated detector parameters
rtklagcorrection -p . -r projections.mha -o corrected_projections.mha \
  -a 0.0,0.1,0.05,0.01 \
  -c 1.0,0.5,0.2,0.1
```
