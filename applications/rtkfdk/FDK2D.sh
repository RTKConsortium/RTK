# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
# Note the sinogram being 3 pixels wide in the y direction to allow back-projection interpolation in a 2D image
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --dimension=512,3,512 --phantomscale=256,1,256

# Reconstruct
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256,1,256

# Create a reference volume for comparison
rtkdrawshepploganphantom --spacing 2 --dimension=256,1,256 -o ref.mha --phantomscale=256,1,256
