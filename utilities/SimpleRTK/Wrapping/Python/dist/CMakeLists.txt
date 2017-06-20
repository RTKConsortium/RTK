#
# Packaging
#
if( SimpleRTK_PYTHON_EGG OR SimpleRTK_PYTHON_WHEEL )
  if( NOT SRTK_PYTHON_USE_VIRTUALENV )
    message( STATUS "Not using SimpleRTK's virtualenv for distribution!\n
Using unknown versions of pip, setuptools and/or wheel packages/" )
  endif()

  set(bdist_setup "${SimpleRTK_Python_BINARY_DIR}/Packaging/setupegg.py")
  set(bdist_commands "")

  if( SimpleRTK_PYTHON_EGG )
    set(bdist_commands "bdist_egg")
  endif()

  if( SimpleRTK_PYTHON_WHEEL )
    set(bdist_commands ${bdist_commands} "bdist_wheel")
  endif()

  add_custom_target( dist.Python
    ${VIRTUAL_PYTHON_EXECUTABLE} ${bdist_setup} ${bdist_commands}
    WORKING_DIRECTORY ${SimpleRTK_Python_BINARY_DIR}
    DEPENDS ${SWIG_MODULE_SimpleRTKPython_TARGET_NAME}
    COMMENT "Creating Python binary distribution" )

  if( SRTK_PYTHON_USE_VIRTUALENV )
    add_dependencies( dist.Python PythonVirtualEnv)
  endif()
  add_dependencies( dist dist.Python )
elseif()
  message( STATUS "Not creating dist.Python target since SRTK_FORBID_DOWNLOADS is enabled" )
endif()
