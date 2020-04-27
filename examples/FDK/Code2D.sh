# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
# Note the sinogram being 3 pixels wide in the y direction to allow back-projection interpolation in a 2D image
rtkprojectgeometricphantom -g geometry.xml -o projections.mha --spacing 2 --dimension=512,3,512 --phantomfile SheppLogan-2d.txt --phantomscale=256,1,256

# Reconstruct
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256,1,256

# Create a reference volume for comparison
rtkdrawgeometricphantom --spacing 2 --dimension=256,1,256 --phantomfile SheppLogan-2d.txt -o ref.mha --phantomscale=256,1,256
