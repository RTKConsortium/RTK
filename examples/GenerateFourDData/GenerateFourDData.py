import itk
from itk import RTK as rtk

PixelType = itk.F
DVFVectorType = itk.CovariantVector[PixelType, 3]
VolumeSeriesType = itk.Image[PixelType, 4]
ProjectionStackType = itk.Image[PixelType, 3]
VolumeType = itk.Image[PixelType, 3]
DVFSequenceImageType = itk.Image[DVFVectorType, 4]

NumberOfProjectionImages = 64
NumberOfFrames = 8

# /* =========================
#  *  Constant volume source
#  * ========================= */
ConstantVolumeSourceType = rtk.ConstantImageSource[VolumeType]
volumeSource = ConstantVolumeSourceType.New()

origin = [-63.0, -31.0, -63.0]
spacing = [4.0, 4.0, 4.0]
size = [32, 16, 32]

volumeSource.SetOrigin(origin)
volumeSource.SetSpacing(spacing)
volumeSource.SetSize(size)
volumeSource.Update()

# /* =========================
#  *  Projection accumulation
#  * ========================= */
ConstantProjectionSourceType = rtk.ConstantImageSource[ProjectionStackType]
projectionsSource = ConstantProjectionSourceType.New()

projOrigin = [-254.0, -254.0, -254.0]
projSpacing = [8.0, 8.0, 1.0]
projSize = [64, 64, NumberOfProjectionImages]

projectionsSource.SetOrigin(projOrigin)
projectionsSource.SetSpacing(projSpacing)
projectionsSource.SetSize(projSize)
projectionsSource.Update()

oneProjectionSource = ConstantProjectionSourceType.New()
oneProjSize = [projSize[0], projSize[1], 1]
oneProjectionSource.SetOrigin(projOrigin)
oneProjectionSource.SetSpacing(projSpacing)
oneProjectionSource.SetSize(oneProjSize)

REIType = rtk.RayEllipsoidIntersectionImageFilter[VolumeType, ProjectionStackType]
PasteType = itk.PasteImageFilter[ProjectionStackType]

# /* Allocate explicit accumulator image */
accumulated = ProjectionStackType.New()
accumulated.SetOrigin(projectionsSource.GetOutput().GetOrigin())
accumulated.SetSpacing(projectionsSource.GetOutput().GetSpacing())
accumulated.SetDirection(projectionsSource.GetOutput().GetDirection())
accumulated.SetRegions(projectionsSource.GetOutput().GetLargestPossibleRegion())
accumulated.Allocate()
accumulated.FillBuffer(0.0)

paste = PasteType.New()
paste.SetDestinationImage(accumulated)

destIndex = [0, 0, 0]

# Create the signal file and the geometry, to be filled in the for loop
signalFileName = "four_d_signal.txt"
geometry = rtk.ThreeDCircularProjectionGeometry.New()

print("Generating projections, geometry, and signal...")
with open(signalFileName, "w", encoding="utf-8") as signalFile:
    for i in range(NumberOfProjectionImages):
        if i % 25 == 0:
            print(f"  projection {i + 1}/{NumberOfProjectionImages}")

        geometry.AddProjection(
            600.0,
            1200.0,
            i * 360.0 / NumberOfProjectionImages,
            0,
            0,
            0,
            0,
            20,
            15,
        )

        geom = rtk.ThreeDCircularProjectionGeometry.New()
        geom.AddProjection(
            600.0,
            1200.0,
            i * 360.0 / NumberOfProjectionImages,
            0,
            0,
            0,
            0,
            20,
            15,
        )

        e1 = REIType.New()
        e1.SetInput(oneProjectionSource.GetOutput())
        e1.SetGeometry(geom)
        e1.SetDensity(2.0)
        e1.SetAxis([60.0, 30.0, 60.0])
        e1.SetCenter([0.0, 0.0, 0.0])
        e1.InPlaceOff()
        e1.Update()

        e2 = REIType.New()
        e2.SetInput(e1.GetOutput())
        e2.SetGeometry(geom)
        e2.SetDensity(-1.0)
        e2.SetAxis([8.0, 8.0, 8.0])
        center = [4 * (abs((4 + i) % 8 - 4.0) - 2.0), 0.0, 0.0]
        e2.SetCenter(center)
        e2.InPlaceOff()
        e2.Update()

        paste.SetSourceImage(e2.GetOutput())
        paste.SetSourceRegion(e2.GetOutput().GetLargestPossibleRegion())
        paste.SetDestinationIndex(destIndex)
        paste.Update()

        accumulated = paste.GetOutput()
        accumulated.DisconnectPipeline()
        paste.SetDestinationImage(accumulated)

        destIndex[2] += 1

        signalFile.write(f"{(i % 8) / 8.0}\n")

itk.imwrite(accumulated, "four_d_projections.mha")
rtk.write_geometry(geometry, "four_d_geometry.xml")

# /* =========================
#  *  DVF & inverse DVF
#  * ========================= */
print("Generating DVF and inverse DVF...")

fourDOrigin = [-63.0, -31.0, -63.0, 0.0]
fourDSpacing = [4.0, 4.0, 4.0, 1.0]
fourDSize = [32, 16, 32, NumberOfFrames]

dvf = DVFSequenceImageType.New()
idvf = DVFSequenceImageType.New()

region = itk.ImageRegion[4]()
dvfSize = [fourDSize[0], fourDSize[1], fourDSize[2], 2]
region.SetSize(dvfSize)

dvf.SetRegions(region)
dvf.SetOrigin(fourDOrigin)
dvf.SetSpacing(fourDSpacing)
dvf.Allocate()

idvf.SetRegions(region)
idvf.SetOrigin(fourDOrigin)
idvf.SetSpacing(fourDSpacing)
idvf.Allocate()

centerIndex = [0, 0, 0, 0]
centerIndex[0] = dvfSize[0] // 2
centerIndex[1] = dvfSize[1] // 2
centerIndex[2] = dvfSize[2] // 2

for t in range(dvfSize[3]):
    for z in range(dvfSize[2]):
        for y in range(dvfSize[1]):
            for x in range(dvfSize[0]):
                v = DVFVectorType()
                v.Fill(0.0)

                d0 = x - centerIndex[0]
                d1 = y - centerIndex[1]
                d2 = z - centerIndex[2]
                if 0.3 * d0 * d0 + d1 * d1 + d2 * d2 < 40:
                    v[0] = -8.0 if t == 0 else 8.0

                iv = DVFVectorType()
                iv.Fill(0.0)
                iv[0] = -v[0]

                index = [x, y, z, t]
                dvf.SetPixel(index, v)
                idvf.SetPixel(index, iv)

itk.imwrite(dvf, "four_d_dvf.mha")
itk.imwrite(idvf, "four_d_idvf.mha")

# /* =========================
#  *  Ground truth
#  * ========================= */
print("Generating ground truth...")

join = itk.JoinSeriesImageFilter[VolumeType, VolumeSeriesType].New()

for t in range(fourDSize[3]):
    DEType = rtk.DrawEllipsoidImageFilter[VolumeType, VolumeType]

    de1 = DEType.New()
    de1.SetInput(volumeSource.GetOutput())
    de1.SetDensity(2.0)
    de1.SetAxis([60.0, 30.0, 60.0])
    de1.SetCenter([0.0, 0.0, 0.0])
    de1.InPlaceOff()
    de1.Update()

    de2 = DEType.New()
    de2.SetInput(de1.GetOutput())
    de2.SetDensity(-1.0)
    de2.SetAxis([8.0, 8.0, 8.0])
    de2.SetCenter([4 * (abs((4 + t) % 8 - 4.0) - 2.0), 0.0, 0.0])
    de2.InPlaceOff()
    de2.Update()

    duplicator = itk.ImageDuplicator[VolumeType].New()
    duplicator.SetInputImage(de2.GetOutput())
    duplicator.Update()

    join.SetInput(t, duplicator.GetOutput())

join.Update()
itk.imwrite(join.GetOutput(), "four_d_ground_truth.mha")

print("Done.")
