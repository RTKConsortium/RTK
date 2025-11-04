# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Reconstruct (SART)
rtksart -p . -r projections.mha -o sart.mha -g geometry.xml --spacing 2 --size 128 \
  -n 5 --nprojpersubset 1 -l 0.3
# You may add "--positivity" to enforce positivity or "--nodisplaced" to disable
# displaced detector weighting.
# You may add "--fp CudaRayCast --bp CudaRayCast" to run on GPU (if available)

# Create a reference volume for comparison
rtkdrawshepploganphantom --like sart.mha -o ref.mha
