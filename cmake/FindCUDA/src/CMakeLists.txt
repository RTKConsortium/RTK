
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:

# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

# The include_directories adds paths to both C/C++ compilation in the native
# compiler and cuda compilation in NVCC.  Note that CUDA_INCLUDE_DIRS is added
# automatically by CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.

# INCLUDE_DIRECTORIES(
#   ${CUDA_INCLUDE_DIRS}
#   )

##############################################################################
# There are four ways to compile source files with NVCC. 

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

# Use one executable only.
CUDA_ADD_EXECUTABLE(test
  test_bin.cu
  main.cc
  external_dependency.h
  )

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
add_definitions(-DMULTIPLIER=2)

set(BUILD_SHARED_LIBS ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
CUDA_ADD_LIBRARY(test_lib
  test_lib.cu
  external_dependency.h
  OPTIONS "-DSTUFF=blah blah"
  RELEASE --use_fast_math -DNDEBUG
  DEBUG -g -DDEBUG
  )

# Then link the shared library to another executable.
ADD_EXECUTABLE(lib_example
  main_for_lib.cc
  )

# Specify the dependency.
TARGET_LINK_LIBRARIES(lib_example
  test_lib
  )

# Using the CUDA_COMPILE macro
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(source_files test_bin.cu)
CUDA_COMPILE(CUDA_FILES test_bin.cu)

ADD_EXECUTABLE(cuda_compile_example
  ${CUDA_FILES}
  ${source_files}
  main.cc
  external_dependency.h
  )
TARGET_LINK_LIBRARIES(cuda_compile_example
 ${CUDA_LIBRARIES}
 )

# Generating PTX files. 
# CUDA_COMPILE_PTX(PTX_FILES CUDA_FILES test_bin.cu)

# Add a special target to clean nvcc generated files.
CUDA_BUILD_CLEAN_TARGET()

