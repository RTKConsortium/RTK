import itk
from itk import RTK as rtk

# Constant values of the script
I0 = 1e4  # Number of photons before attenuation
mu = 0.01879  # mm^-1 (attenuation coefficient of water at 75 keV)

# Simulate a Shepp Logan projection
ImageType = itk.Image[itk.F, 3]
geometry = rtk.ThreeDCircularProjectionGeometry.New()
geometry.AddProjection(1000.0, 0.0, 0)
projection = rtk.constant_image_source(
    size=[64, 64, 1], spacing=[2.0] * 3, ttype=[ImageType], origin=[-63.0, -63.0, 0]
)
projection = rtk.shepp_logan_phantom_filter(
    projection, geometry=geometry, phantom_scale=70
)
itk.imwrite(projection, "projection.mha")

# Use ITK to add pre-log Poisson noise
noisy_projection = itk.multiply_image_filter(projection, constant=-mu)
noisy_projection = itk.exp_image_filter(noisy_projection)
noisy_projection = itk.multiply_image_filter(noisy_projection, constant=I0)
noisy_projection = itk.shot_noise_image_filter(noisy_projection)
noisy_projection = itk.threshold_image_filter(
    noisy_projection, lower=1.0, outside_value=1.0
)
noisy_projection = itk.multiply_image_filter(noisy_projection, constant=1.0 / I0)
noisy_projection = itk.log_image_filter(noisy_projection)
noisy_projection = itk.multiply_image_filter(noisy_projection, constant=-1.0 / mu)
itk.imwrite(noisy_projection, "noisy_projection.mha")

# Alternative NumPy implementation
# projection = rtk.constant_image_source(size=[64,64,1],spacing=[2.]*3, ttype=[ImageType], origin=[-63.,-63.,0])
# projection = rtk.shepp_logan_phantom_filter(projection, geometry=geometry, phantom_scale=70)
# import numpy as np
# proj_array = itk.array_view_from_image(projection)
# proj_array = I0*np.exp(-1.*mu*proj_array)
# proj_array = np.maximum(np.random.poisson(proj_array), 1)
# proj_array = np.log(I0/proj_array)/mu
# projection_noisy = itk.image_view_from_array(proj_array)
# projection_noisy.CopyInformation(projection)
# itk.imwrite(projection_noisy, 'projection.mha')

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))
plt.subplot(131)
plt.title("Noiseless projection")
plt.imshow(
    itk.array_view_from_image(projection)[0, :, :],
    cmap=plt.cm.gray,
    origin="lower",
    clim=[0, 140],
)
plt.xlabel("u pixel index")
plt.ylabel("v pixel index")
plt.subplot(132)
plt.title("Noisy projection")
plt.imshow(
    itk.array_view_from_image(noisy_projection)[0, :, :],
    cmap=plt.cm.gray,
    origin="lower",
    clim=[0, 140],
)
plt.xlabel("u pixel index")
plt.subplot(133)
plt.title("Central vertical profile")
plt.plot(itk.array_view_from_image(projection)[0, :, 32], label="Noiseless")
plt.plot(itk.array_view_from_image(noisy_projection)[0, :, 32], label="Noisy")
plt.xlabel("v pixel index")
plt.ylabel("Pixel intensity")
plt.legend()
plt.savefig("AddNoise.png", bbox_inches="tight")
plt
