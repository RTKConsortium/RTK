cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(FirstReconstruction)

# Find ITK with RTK
find_package(
  ITK
  REQUIRED
  COMPONENTS
    RTK
    ITKImageIO
)
if("${ITK_VERSION_MAJOR}" VERSION_LESS "6")
  include(${ITK_USE_FILE})
  set(RTK_EXAMPLE_TARGETS ${ITK_LIBRARIES})
else()
  itk_generate_factory_registration()
  set(
    RTK_EXAMPLE_TARGETS
    ${ITK_INTERFACE_LIBRARIES}
    ITK::ITKImageIO
    ITK::ITKFFTImageFilterInit
  )
  message(ITK_INTERFACE_LIBRARIES=${ITK_INTERFACE_LIBRARIES})
endif()

# Executable(s)
add_executable(FirstReconstruction FirstReconstruction.cxx)
target_link_libraries(FirstReconstruction ${RTK_EXAMPLE_TARGETS})

if(${RTK_USE_CUDA})
  add_executable(FirstCudaReconstruction FirstCudaReconstruction.cxx)
  target_link_libraries(FirstCudaReconstruction ${RTK_EXAMPLE_TARGETS})
endif()
