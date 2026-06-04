#!/usr/bin/env bash
# Create Amsterdam shroud from projections
rtkamsterdamshroud --path projections --regexp '.*.his' --output shroud.mha --unsharp 650

# Extract phase signal
rtkextractshroudsignal --input shroud.mha --output signal.txt

# Overlay phase minima on the shroud and write an RGB image
rtkoverlayphaseandshroud -i shroud.mha -o overlay.png --signal signal.txt
