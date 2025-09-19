# Generate geometry for fan-beam setup
rtksimulatedgeometry -n 720 -o geometry.xml --arc 360

# Create projections of the phantom file
# Note the sinogram being 3 pixels wide in the y direction to allow back-projection interpolation in a 2D image
rtkprojectgeometricphantom -g geometry.xml -o projections.mha --spacing 2 --size=512,3,512 --phantomfile SheppLogan-2d.txt --phantomscale=256,1,256

# Perform Conjugate Gradient reconstruction
rtkconjugategradient -p . -r projections.mha -o cg.mha -g geometry.xml --spacing 2 --size 256 1 256 -n 10

# Create a reference volume for comparison
rtkdrawgeometricphantom --spacing 2 --size=256,1,256 --phantomfile SheppLogan-2d.txt -o ref.mha --phantomscale=256,1,256
