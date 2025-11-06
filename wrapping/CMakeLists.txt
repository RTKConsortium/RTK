itk_wrap_module(RTK)

set(
  WRAPPER_SUBMODULE_ORDER
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
  rtkPolynomialGainCorrectionImageFilter
  itkImageSourceRTK
  itkImageToImageFilterRTK
  rtkTotalVariationDenoisingBPDQImageFilter
  itkCudaImageToImageFilterRTK
  itkInPlaceImageFilterRTK
  itkCudaInPlaceImageFilterRTK
  rtkForwardProjectionImageFilter
  itkPyBufferRTK
)

itk_auto_load_submodules()
itk_end_wrap_module()

set(RTK_VERSION_SCRIPT "${RTK_BINARY_DIR}/applications/rtkversion.py")
configure_file(
  ${RTK_SOURCE_DIR}/applications/rtkversion.py.in
  ${RTK_VERSION_SCRIPT}
)

file(
  GLOB RTK_GROUP_SCRIPTS
  CONFIGURE_DEPENDS
  "${RTK_SOURCE_DIR}/applications/rtk*_group.py"
)
file(
  GLOB RTK_APP_SCRIPTS
  CONFIGURE_DEPENDS
  "${RTK_SOURCE_DIR}/applications/rtk*/rtk*.py"
)

wrap_itk_python_bindings_install(/itk "RTK"
  __init_rtk__.py
  ${RTK_GROUP_SCRIPTS}
  ${RTK_APP_SCRIPTS}
  ${RTK_VERSION_SCRIPT}
  ${RTK_SOURCE_DIR}/wrapping/rtkExtras.py
  ${RTK_SOURCE_DIR}/applications/rtkargumentparser.py
)
