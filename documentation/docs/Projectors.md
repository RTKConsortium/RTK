# Forward and backprojectors

In cone-beam computed tomography (CBCT), forward and backprojectors are crucial components of the reconstruction algorithms. RTK implements multiple forward and backprojectors, each tailored to different purposes.

All projector classes in RTK are implemented as ITK filters. They inherit from either [`rtk::ForwardProjectionImageFilter`](https://www.openrtk.org/Doxygen/classrtk_1_1ForwardProjectionImageFilter.html) or [`rtk::BackProjectionImageFilter`](https://www.openrtk.org/Doxygen/classrtk_1_1BackProjectionImageFilter.html), both inheriting from [`itk::InPlaceImageFilter`](https://itk.org/Doxygen/html/classitk_1_1InPlaceImageFilter.html).

## Forward projectors

An intuitive design for a forward projector would be:

```{eval-rst}
.. graphviz::
  :align: center

  digraph IntuitiveDesign {
    node [shape=box];
    Input [ label="Input (Volume)" ];
    Input [shape=Mdiamond];
    ForwardProject [ label="rtk::ForwardProjectionImageFilter" ];
    Output [ label="Output (Projections)"];
    Output [shape=Mdiamond];
    Input -> ForwardProject;
    ForwardProject -> Output;
  }
```

but ITK's filter output has the same information (size, spacing, direction, ...) by default as the first input. This could be changed but it was deemed simpler to chose the following design:

```{eval-rst}
.. graphviz::
  :align: center

  digraph IntuitiveDesign {
    node [shape=box];
    Input0 [ label="Input (Projections)" ];
    Input0 [ shape=Mdiamond ];
    Input1 [ label="Input (Volume)" ];
    Input1 [ shape=Mdiamond ];
    ForwardProject [ label="rtk::ForwardProjectionImageFilter" href="https://www.openrtk.org/Doxygen/classrtk_1_1ForwardProjectionImageFilter.html" target="_top"];
    Output [ label="Output (Projections)"];
    Output [ shape=Mdiamond ];
    Input0 -> ForwardProject;
    Input1 -> ForwardProject;
    ForwardProject -> Output;
  }
```

Forward projections of the second input are then added to the first input. Because forward projectors inherit from [`itk::InplaceImageFilter`](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1InPlaceImageFilter.html), there is still a single memory buffer for the projections. Projections filled with a constant, 0 by default, can be generated with [`rtk::ConstantImageSource`](http://www.openrtk.org/Doxygen/classrtk_1_1ConstantImageSource.html) which only generates the requested part in the memory buffer.

RTK supports the following forward projector implementations :

- [Joseph](https://www.openrtk.org/Doxygen/classrtk_1_1JosephForwardProjectionImageFilter.html)
- [JosephAttenuated](https://www.openrtk.org/Doxygen/classrtk_1_1JosephForwardAttenuatedProjectionImageFilter.html)
- [Zeng](https://www.openrtk.org/Doxygen/classrtk_1_1ZengForwardProjectionImageFilter.html)
- [MIP](https://www.openrtk.org/Doxygen/classrtk_1_1MaximumIntensityProjectionImageFilter.html)

Cuda based projector:
- [CudaRayCast](https://www.openrtk.org/Doxygen/classrtk_1_1CudaForwardProjectionImageFilter.html)
- [CudaWarpRayCast](https://www.openrtk.org/Doxygen/classrtk_1_1CudaWarpForwardProjectionImageFilter.html)

## Back projectors

Similarly, the design chosen for RTK back projectors is:

```{eval-rst}
.. graphviz::
  :align: center

  digraph IntuitiveDesign {
    node [shape=box];
    Input0 [ label="Input (Volume)" ];
    Input0 [ shape=Mdiamond ];
    Input1 [ label="Input (Projections)" ];
    Input1 [ shape=Mdiamond ];
    BackProject [ label="rtk::BackProjectionImageFilter" href="https://www.openrtk.org/Doxygen/classrtk_1_1BackProjectionImageFilter.html" target="_top"];
    Output [ label="Output (Volume)"];
    Output [ shape=Mdiamond ];
    Input0 -> BackProject;
    Input1 -> BackProject;
    BackProject -> Output;
  }
```

RTK supports the following back projector implementations :

- [VoxelBasedBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1BackProjectionImageFilter.html)
- [FDKBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1FDKBackProjectionImageFilter.html)
- [FDKWarpBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1FDKWarpBackProjectionImageFilter.html)
- [Joseph](https://www.openrtk.org/Doxygen/classrtk_1_1JosephBackProjectionImageFilter.html)
- [JosephAttenuated](https://www.openrtk.org/Doxygen/classrtk_1_1JosephBackAttenuatedProjectionImageFilter.html)
- [Zeng](https://www.openrtk.org/Doxygen/classrtk_1_1ZengBackProjectionImageFilter.html)

Cuda based projector:
- [CudaBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1CudaBackProjectionImageFilter.html)
- [CudaFDKBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1CudaFDKBackProjectionImageFilter.html)
- [CudaRayCast](https://www.openrtk.org/Doxygen/classrtk_1_1CudaRayCastBackProjectionImageFilter.html)
- [CudaWarpBackProjection](https://www.openrtk.org/Doxygen/classrtk_1_1CudaWarpBackProjectionImageFilter.html)
