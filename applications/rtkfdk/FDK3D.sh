# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml
# You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Reconstruct
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --size 128

# Create a reference volume for comparison
rtkdrawshepploganphantom --like fdk.mha -o ref.mha
