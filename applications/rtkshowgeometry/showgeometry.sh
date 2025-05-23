# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --size 256

# Reconstruct
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --size 256

# Show geometry
rtkshowgeometry -p . -r projections.mha --geometry geometry.xml  --input fdk.mha --show_trajectory
