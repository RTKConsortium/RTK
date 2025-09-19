# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml
# You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 256

# Perform least squares reconstruction
rtkconjugategradient -p . -r noisyLineIntegrals.mha -o LeastSquares.mha -g geom.xml -n 20

# Perform weighted least squares reconstruction
rtkconjugategradient -p . -r noisyLineIntegrals.mha -o WeightedLeastSquares.mha -g geom.xml -w weightsmap.mha -n 10

# Create a reference volume for comparison
 rtkdrawshepploganphantom --spacing 2 --size 256 -o ref.mha
