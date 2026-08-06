# Extract shroud signal

Extracts the breathing signal from a shroud image.

This application reads a shroud image and extracts a respiratory signal using either a regularised 1D method (`Reg1D`) or dynamic programming (`DynamicProgramming`). It can also write an estimated Hilbert phase image.

```bash
# Basic extraction (default method: Reg1D)
rtkextractshroudsignal -i shroud.mha -o breathing.txt

# Use Dynamic Programming method (requires a maximum amplitude)
rtkextractshroudsignal -i shroud.mha -o breathing.txt -m DynamicProgramming -a 30

# Also write Hilbert phase text file and tune preprocessing
rtkextractshroudsignal -i shroud.mha -o breathing.txt -p phase.txt --movavg 5 --unsharp 65 --model LOCAL_PHASE
```

See the [rtkamsterdamshroud documentation](../rtkamsterdamshroud/README.md) for a shroud-extraction workflow
