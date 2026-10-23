# Create a simulated geometry
 rtksimulatedgeometry -n 180 -o geometry.xml

# simulate noisy projections
rtkprojectshepploganphantom -o projections.mha -g geometry.xml --spacing 2 \
  --size 200 --gaussian 0.01 --poisson 1e6,0.01879

# reconstruct
rtkfdk -p . -r projections.mha -o noisy-fdk.mha -g geometry.xml --spacing 2 --size 128

# post‑process with TV denoising
rtktotalvariationdenoising -i noisy-fdk.mha -o denoised.mha --gamma 0.5 --niter 10 --verbose
