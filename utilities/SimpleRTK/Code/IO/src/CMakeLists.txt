
set( SimpleRTKIOSource
  srtkImageFileReader.cxx
  srtkImageFileWriter.cxx
  srtkImageReaderBase.cxx
  srtkImageSeriesReader.cxx
  srtkImageSeriesWriter.cxx
  srtkImportImageFilter.cxx
  srtkShow.cxx
  srtkThreeDCircularProjectionGeometryXMLFileReader.cxx
  srtkThreeDCircularProjectionGeometryXMLFileWriter.cxx
  srtkProjectionsReader.cxx
  )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

include(${ITK_USE_FILE})
include(${RTK_USE_FILE})

add_library ( SimpleRTKIO ${SimpleRTKIOSource} )
target_link_libraries ( SimpleRTKIO ${ITK_LIBRARIES} SimpleRTKCommon )
set_target_properties( SimpleRTKIO PROPERTIES SKIP_BUILD_RPATH TRUE )

srtk_install_exported_target( SimpleRTKIO )

# Add custom command that will delete java files which need to be rebuilt when changes
# are made to IO
if( WRAP_JAVA )
  add_custom_command(
    TARGET SimpleRTKIO
    POST_BUILD
    COMMENT "Cleaning java build..."
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SimpleRTK_BINARY_DIR}/Wrapping/org/rtk/simple/*.java
    COMMAND ${CMAKE_COMMAND} -E remove -f ${SimpleRTK_BINARY_DIR}/Wrapping/build/org/rtk/simple/*.class
  )
endif()
