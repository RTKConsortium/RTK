# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml
# You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 200

# Backproject the generated projections
rtkbackprojections -p . -r projections.mha -g geometry.xml -o backproj.mha --bp Joseph --spacing 2 --size 128
