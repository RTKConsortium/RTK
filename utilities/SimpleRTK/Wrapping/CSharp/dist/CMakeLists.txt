
#
# CSharp Packaging
#
if( Java_JAR_EXECUTABLE )

  set(_files "")
  list( APPEND _files
    ${SimpleRTK_DOC_FILES}
    "${CSHARP_BINARY_DIRECTORY}/SimpleRTKCSharpManaged.dll"
    )

  if(NOT DEFINED SIMPLERTK_CSHARP_ARCH)
    if(MSVC)
      if (CMAKE_CL_64)
        set(SIMPLERTK_CSHARP_ARCH "win64")
      else()
        set(SIMPLERTK_CSHARP_ARCH "win32")
      endif()
    else()
      set(SIMPLERTK_CSHARP_ARCH "unknown")
    endif()

    if(CSHARP_PLATFORM)
      set(SIMPLERTK_CSHARP_ARCH "${SIMPLERTK_CSHARP_ARCH}-${CSHARP_PLATFORM}")
    endif()
  endif()

  set( CSHARP_PACKAGE_STAGE_DIR "SimpleRTK-${SimpleRTK_VERSION}-CSharp-${SIMPLERTK_CSHARP_ARCH}")

  add_custom_target( dist.CSharp
    COMMENT "Creating CSharp package ${CSHARP_PACKAGE_STAGE_DIR}.zip"
    DEPENDS SimpleRTKCSharpManaged
    )

  add_custom_command( TARGET dist.CSharp
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CSHARP_PACKAGE_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CSHARP_PACKAGE_STAGE_DIR}"
    COMMENT "Creating CSharp staging directory..."
    )

  foreach(_f ${_files})
    get_filename_component(_f_name ${_f} NAME )
    add_custom_command( TARGET dist.CSharp
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "${CSHARP_PACKAGE_STAGE_DIR}/${_f_name}"
      COMMENT "Copying ${_f_name} to CSharp stage..."
    )
  endforeach()

  add_custom_command( TARGET dist.CSharp
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${SWIG_MODULE_SimpleRTKCSharpNative_TARGET_NAME}>" "${CSHARP_PACKAGE_STAGE_DIR}"
    COMMENT "Copying $<TARGET_FILE:${SWIG_MODULE_SimpleRTKCSharpNative_TARGET_NAME}> to CSharp stage..."
#    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:SimpleRTKCSharpManaged>" "${CSHARP_PACKAGE_STAGE_DIR}"
#    COMMENT "Copying $<TARGET_FILE:${SimpleRTKCSharpManaged}> to CSharp stage..."

    SimpleRTKCSharpManaged
    COMMAND ${Java_JAR_EXECUTABLE} cfM "${CSHARP_PACKAGE_STAGE_DIR}.zip" "${CSHARP_PACKAGE_STAGE_DIR}"
    COMMENT "Packaging CSHARP distribution..."
    )

  add_dependencies( dist dist.CSharp )

endif()
