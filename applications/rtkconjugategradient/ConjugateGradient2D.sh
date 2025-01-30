# Generate geometry for fan-beam setup
rtksimulatedgeometry -n 720 -o geometry.xml --arc 360

# Simulate projections using the Shepp-Logan phantom
rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 0.5 --dimension 1024,3,720

# Create an initial 3D volume for reconstruction
rtkdrawshepploganphantom --spacing 0.5 --dimension 512,1,512 -o initial_volume.mha --phantomscale=512,1,512

# Perform Conjugate Gradient reconstruction
rtkconjugategradient -p . -r projections.mha -o cg.mha -g geometry.xml --spacing 0.5 --dimension 512 3 512 -n 50
