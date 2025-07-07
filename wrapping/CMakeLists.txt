itk_wrap_module(RTK)

set(WRAPPER_SUBMODULE_ORDER
  itkFixedArrayRTK
  itkVectorRTK
  itkCovariantVectorRTK
  itkPointRTK
  itkContinuousIndexRTK
  itkMatrixRTK
  itkVariableLengthVectorRTK
  itkImageBaseRTK
  itkImageRTK
  itkCudaImageRTK
  itkVectorImageRTK
  rtkFFTRampImageFilter
  rtkParkerShortScanImageFilter
  rtkScatterGlareCorrectionImageFilter
  itkImageSourceRTK
  itkImageToImageFilterRTK
  itkCudaImageToImageFilterRTK
  itkInPlaceImageFilterRTK
  itkCudaInPlaceImageFilterRTK
  rtkForwardProjectionImageFilter
  itkPyBufferRTK
)

itk_auto_load_submodules()
itk_end_wrap_module()

configure_file (${RTK_SOURCE_DIR}/applications/rtkversion.py.in ${RTK_BINARY_DIR}/applications/rtkversion.py)

wrap_itk_python_bindings_install(/itk "RTK"
    __init_rtk__.py
    ${RTK_BINARY_DIR}/applications/rtkversion.py
    ${RTK_SOURCE_DIR}/applications/rtkargumentparser.py
    ${RTK_SOURCE_DIR}/applications/rtk3Doutputimage_group.py
    ${RTK_SOURCE_DIR}/applications/rtkinputprojections_group.py
    ${RTK_SOURCE_DIR}/applications/rtkiterations_group.py
    ${RTK_SOURCE_DIR}/applications/rtkprojectors_group.py
    ${RTK_SOURCE_DIR}/applications/rtkadmmtotalvariation/rtkadmmtotalvariation.py
    ${RTK_SOURCE_DIR}/applications/rtkamsterdamshroud/rtkamsterdamshroud.py
    ${RTK_SOURCE_DIR}/applications/rtkbackprojections/rtkbackprojections.py
    ${RTK_SOURCE_DIR}/applications/rtkbioscangeometry/rtkbioscangeometry.py
    ${RTK_SOURCE_DIR}/applications/rtkcheckimagequality/rtkcheckimagequality.py
    ${RTK_SOURCE_DIR}/applications/rtkconjugategradient/rtkconjugategradient.py
    ${RTK_SOURCE_DIR}/applications/rtkdigisensgeometry/rtkdigisensgeometry.py
    ${RTK_SOURCE_DIR}/applications/rtkdrawgeometricphantom/rtkdrawgeometricphantom.py
    ${RTK_SOURCE_DIR}/applications/rtkdrawshepploganphantom/rtkdrawshepploganphantom.py
    ${RTK_SOURCE_DIR}/applications/rtkelektasynergygeometry/rtkelektasynergygeometry.py
    ${RTK_SOURCE_DIR}/applications/rtkfdk/rtkfdk.py
    ${RTK_SOURCE_DIR}/applications/rtkforwardprojections/rtkforwardprojections.py
    ${RTK_SOURCE_DIR}/applications/rtkorageometry/rtkorageometry.py
    ${RTK_SOURCE_DIR}/applications/rtkprojectgeometricphantom/rtkprojectgeometricphantom.py
    ${RTK_SOURCE_DIR}/applications/rtkprojections/rtkprojections.py
    ${RTK_SOURCE_DIR}/applications/rtkprojectshepploganphantom/rtkprojectshepploganphantom.py
    ${RTK_SOURCE_DIR}/applications/rtkshowgeometry/rtkshowgeometry.py
    ${RTK_SOURCE_DIR}/applications/rtksimulatedgeometry/rtksimulatedgeometry.py
    ${RTK_SOURCE_DIR}/applications/rtkvarianobigeometry/rtkvarianobigeometry.py
    ${RTK_SOURCE_DIR}/wrapping/rtkExtras.py
)
