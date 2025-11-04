# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Reconstruct (OSEM)
rtkosem -p . -r projections.mha -o osem.mha -g geometry.xml --spacing 2 --size 128 \
	-n 10 --nprojpersubset 30
# You may add "--betaregularization" to enable regularization
# You may add "--fp CudaRayCast --bp CudaRayCast" to run on GPU (if available)

# Create a reference volume for comparison
rtkdrawshepploganphantom --like osem.mha -o ref.mha
