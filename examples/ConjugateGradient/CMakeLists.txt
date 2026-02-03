cmake_minimum_required(VERSION 3.9.5 FATAL_ERROR)

# This project is designed to be built outside the RTK source tree.
project(ConjugateGradient)

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
  )
endif()

# Executable(s)
add_executable(ConjugateGradient ConjugateGradient.cxx)
target_link_libraries(ConjugateGradient ${RTK_EXAMPLE_TARGETS})
