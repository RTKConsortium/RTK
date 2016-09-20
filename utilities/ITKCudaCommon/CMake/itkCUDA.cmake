if(NOT _BUILD_CUDA_KERNEL)
  find_package(CUDA REQUIRED)
  if(NOT ${CUDA_FOUND})
     message(FATAL "Could not find CUDA")
  endif()

  macro(generate_cuda_ptx_wrappers KERNELSPTX KERNELSCXX)

    set(PTX_FILES "")
    set(CU_FILES "")

    foreach(file ${ARGN})
      if(${file} MATCHES ".*\\.cu$")
        list(APPEND CU_FILES ${file})
      endif()
    endforeach()

    CUDA_COMPILE_PTX(KernelsPTX ${CU_FILES})

    foreach(file ${KernelsPTX})
      if(${file} MATCHES ".*\\.ptx$")
        get_filename_component(FilterName ${file} NAME_WE)
        string(REGEX REPLACE ".*generated_" "" KernelName ${FilterName})
        list(APPEND PTX_FILES ${file})
        list(APPEND CUDAKERNELS
          "${CMAKE_CURRENT_BINARY_DIR}/${KernelName}CudaKernel.cxx")
      endif()
    endforeach()

    add_custom_command(
      OUTPUT ${CUDAKERNELS}
      DEPENDS ${PTX_FILES}
      COMMAND ${CMAKE_COMMAND} ARGS
        -D "_BUILD_CUDA_KERNEL=TRUE"
        -D "PTXKernels:STRING=\"${PTX_FILES}\""
	-P "${ITKCudaCommon_SOURCE_DIR}/CMake/itkCUDA.cmake" # call to itself!
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      COMMENT "Building CUDA PTX wrappers"
    )
    set_source_files_properties(${CUDAKERNELS} PROPERTIES GENERATED TRUE)
    set(${KERNELSPTX} ${KernelsPTX})
    set(${KERNELSCXX} ${CUDAKERNELS})
  endmacro()
endif()

# Script called to generate a CUDA PTX Source file
if(_BUILD_CUDA_KERNEL)

macro(sourcefile_to_string SOURCE_FILE RESULT_CMAKE_VAR)
   file(STRINGS ${SOURCE_FILE} FileStrings)
   foreach(SourceLine ${FileStrings})
     # replace all \ with \\ to make the c string constant work
     string(REGEX REPLACE "\\\\" "\\\\\\\\" TempSourceLine "${SourceLine}")
     # replace all " with \" to make the c string constant work
     string(REGEX REPLACE "\"" "\\\\\"" EscapedSourceLine "${TempSourceLine}")
     set(${RESULT_CMAKE_VAR} "${${RESULT_CMAKE_VAR}}\n s << \"${EscapedSourceLine}\\n\";")
   endforeach()
endmacro()

macro(write_cuda_ptx_kernel_to_file NAMESPACE PTX_FILE GPUFILTER_NAME GPUFILTER_KERNELNAME
   OUTPUT_FILE)
  sourcefile_to_string(${PTX_FILE} ${GPUFILTER_KERNELNAME}_SourceString)
  set(${GPUFILTER_KERNELNAME}_KernelString
    "#include \"${NAMESPACE}${GPUFILTER_NAME}.h\"
    namespace ${NAMESPACE}
    {
    std::string ${GPUFILTER_KERNELNAME}::GetCudaPTXSource()
    {
      std::stringstream s;
      ${${GPUFILTER_KERNELNAME}_SourceString};
      return s.str();
    }
    }"
  )

  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
       "${${GPUFILTER_KERNELNAME}_KernelString}")
endmacro()

macro(write_cuda_ptx_kernels GPUKernels)
  foreach(GPUKernel ${GPUKernels})
    get_filename_component(FilterName ${GPUKernel} NAME_WE)
    string(REGEX REPLACE ".*generated_" "" KernelName ${FilterName})
    message(STATUS "Writing CUDA PTX kernel wrapper ${KernelName}CudaKernel.cxx")
    write_cuda_ptx_kernel_to_file(itk ${GPUKernel} ${KernelName} ${KernelName}Kernel "${KernelName}CudaKernel.cxx")
  endforeach()
endmacro()

write_cuda_ptx_kernels("${PTXKernels}")
endif()
