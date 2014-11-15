 set ( SimpleRTKCommonSource
  srtkImage.cxx
  srtkImageExplicit.cxx
  srtkProcessObject.cxx
  srtkTransform.cxx
  srtkThreeDCircularProjectionGeometry.cxx
  srtkCommand.cxx
  srtkFunctionCommand.cxx
  srtkPixelIDValues.cxx
  srtkExceptionObject.cxx
  srtkEvent.cxx
  srtkInterpolator.cxx
  srtkVersion.cxx
  ../include/Ancillary/hl_md5.cxx
  ../include/Ancillary/hl_sha1.cxx
  )

set ( ITK_NO_IO_FACTORY_REGISTER_MANAGER 1 )
include(${ITK_USE_FILE})
include(${RTK_USE_FILE})

add_library ( SimpleRTKCommon ${SimpleRTKCommonSource} ${SimpleRTKAncillarySource} )
target_link_libraries ( SimpleRTKCommon ${ITK_LIBRARIES} ${RTK_LIBRARIES})
set_target_properties( SimpleRTKCommon PROPERTIES SKIP_BUILD_RPATH TRUE )

srtk_install_exported_target( SimpleRTKCommon )

# Add custom command that will delete java files which need to be rebuilt when changes
# are made to Common
if( WRAP_JAVA )
  add_custom_command(
    TARGET SimpleRTKCommon
    POST_BUILD
    COMMENT "Cleaning java build..."
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SimpleRTK_BINARY_DIR}/Wrapping/org/rtk/simple/*.java
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SimpleRTK_BINARY_DIR}/Wrapping/build/org/rtk/simple/*.class
  )
endif()
