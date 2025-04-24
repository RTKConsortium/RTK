import itk
from itk import RTK as rtk

Dimension = 3
OutputPixelType = itk.F
OutputImageType = itk.Image[OutputPixelType, Dimension]

numberOfProjections = 180
angularArc = 360.0
sid = 600
sdd = 1200
scale = 2.0
config_file_name = "Thorax"
rotation = itk.matrix_from_array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Set up the geometry for the projections
geometry = rtk.ThreeDCircularProjectionGeometry.New()
for x in range(0, numberOfProjections):
    angle = x * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)
rtk.write_geometry(geometry, "geometry.xml")

# Project the geometric phantom image
projections_source = rtk.constant_image_source(
    ttype=[OutputImageType],
    origin=[-127.0, -127.0, 0.0],
    size=[128, 128, numberOfProjections],
    spacing=[2.0] * 3,
)
pgp = rtk.project_geometric_phantom_image_filter(
    projections_source,
    geometry=geometry,
    phantom_scale=scale,
    rotation_matrix=rotation,
    config_file=config_file_name,
    is_forbild_config_file=True,
)
itk.imwrite(pgp, "projections.mha")

# Draw the geometric phantom image
ref_source = rtk.constant_image_source(
    ttype=[OutputImageType], origin=[-63.5] * 3, size=[128] * 3
)
dgp = rtk.draw_geometric_phantom_image_filter(
    ref_source,
    phantom_scale=scale,
    rotation_matrix=rotation,
    config_file=config_file_name,
    is_forbild_config_file=True,
)
itk.imwrite(dgp, "ref.mha")

# Perform FDK reconstruction filtering
fdk_source = rtk.constant_image_source(
    ttype=[OutputImageType],
    origin=[-63.5] * 3,
    size=[128] * 3,
    spacing=[1.0] * 3,
    constant=0.0,
)
feldkamp = rtk.fdk_cone_beam_reconstruction_filter(fdk_source, pgp, geometry=geometry)
itk.imwrite(feldkamp, "fdk.mha")
