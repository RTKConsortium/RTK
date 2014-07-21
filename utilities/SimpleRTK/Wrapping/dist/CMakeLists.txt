#
# Packaging
#

# A general packaging target, not built by default, to build packages for each
# language.  This should depend on all language specific targets.
add_custom_target( dist cmake -E echo "Finished generating wrapped packages for distribution..." )

if( WRAP_PYTHON )

  set(bdist_commands "bdist_egg")
  if( SimpleRTK_PYTHON_WHEEL )
    set(bdist_commands ${bdist_commands} bdist_wheel)
  endif()

  add_custom_target( dist.Python
    ${VIRTUAL_PYTHON_EXECUTABLE} ${SimpleRTK_BINARY_DIR}/Wrapping/PythonPackage/setup.py ${bdist_commands}
    WORKING_DIRECTORY ${SimpleRTK_BINARY_DIR}/Wrapping
    DEPENDS ${SWIG_MODULE_SimpleRTKPython_TARGET_NAME}
    COMMENT "Creating Python binary distribution" )

  add_dependencies( dist.Python PythonVirtualEnv)
  add_dependencies( dist dist.Python )

endif()

#
# JAVA Packaging
#
if( WRAP_JAVA )

  set(_files "")
  list( APPEND _files
    ${SimpleRTK_DOC_FILES}
    ${SimpleRTK_BINARY_DIR}/Wrapping/${JAR_FILE}
    ${SimpleRTK_BINARY_DIR}/Wrapping/${JAVA_SOURCE_FILE}
    ${SimpleRTK_BINARY_DIR}/Wrapping/${JAVADOC_FILE}
    )

  if(NOT DEFINED SIMPLERTK_JAVA_ARCH)
    if(MSVC)
      if (CMAKE_CL_64)
        set(SIMPLERTK_JAVA_ARCH "win64")
      else()
        set(SIMPLERTK_JAVA_ARCH "win32")
      endif()
    elseif(APPLE)
      # assemble macosx-OSX_DEPLOYMENT_TARGET-OSX_ARCHITECTURE
      set(SIMPLERTK_JAVA_ARCH "macosx")
      if(CMAKE_OSX_DEPLOYMENT_TARGET)
        set(SIMPLERTK_JAVA_ARCH "${SIMPLERTK_JAVA_ARCH}-${CMAKE_OSX_DEPLOYMENT_TARGET}")
      endif()
      if(CMAKE_OSX_ARCHITECTURES MATCHES "i386" AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        if (CMAKE_OSX_ARCHITECTURES MATCHES "ppc")
          set(SIMPLERTK_JAVA_ARCH "${SIMPLERTK_JAVA_ARCH}-universal")
        else()
          set(SIMPLERTK_JAVA_ARCH "${SIMPLERTK_JAVA_ARCH}-intel")
          set(_OSX_ARCHITECTURES "intel")
        endif()
      elseif(CMAKE_OSX_ARCHITECTURES)
        string(REPLACE ";" "_" _OSX_ARCHITECTURES "${CMAKE_OSX_ARCHITECTURES}")
        set(SIMPLERTK_JAVA_ARCH "${SIMPLERTK_JAVA_ARCH}-${_OSX_ARCHITECTURES}")
      else()
        set(SIMPLERTK_JAVA_ARCH "${SIMPLERTK_JAVA_ARCH}-unknown")
      endif()
    else()
      set(SIMPLERTK_JAVA_ARCH "unknown")
    endif()
  endif()

  set( JAVA_PACKAGE_STAGE_DIR "SimpleRTK-${SimpleRTK_VERSION}-Java-${SIMPLERTK_JAVA_ARCH}")

  add_custom_target( dist.Java
    COMMENT "Creating Java package ${JAVA_PACKAGE_STAGE_DIR}.zip"
    DEPENDS org_rtk_simple_jar
    )

  add_custom_command( TARGET dist.Java
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${JAVA_PACKAGE_STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${JAVA_PACKAGE_STAGE_DIR}"
    COMMENT "Creating Java staging directory..."
    )

  foreach(_f ${_files})
    get_filename_component(_f_name ${_f} NAME )
    add_custom_command( TARGET dist.Java
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_f}" "${JAVA_PACKAGE_STAGE_DIR}/${_f_name}"
      COMMENT "Copying ${_f_name} to Java stage..."
    )
  endforeach()

  add_custom_command( TARGET dist.Java
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${SWIG_MODULE_SimpleRTKJava_TARGET_NAME}>" "${JAVA_PACKAGE_STAGE_DIR}"
    COMMENT "Copying $<TARGET_FILE:${SWIG_MODULE_SimpleRTKJava_TARGET_NAME}> to Java stage..."
    COMMAND ${Java_JAR_EXECUTABLE} cfM "${JAVA_PACKAGE_STAGE_DIR}.zip" "${JAVA_PACKAGE_STAGE_DIR}"
    COMMENT "Packaging JAVA distribution..."
    )

  add_dependencies( dist dist.Java )

endif()

#
# CSharp Packaging
#
if( WRAP_CSHARP AND Java_JAR_EXECUTABLE )

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
