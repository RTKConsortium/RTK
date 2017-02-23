# Include the hand-written and auto-generated code
include_directories (
 ${CMAKE_CURRENT_SOURCE_DIR}
 )

set_source_files_properties ( ${SimpleRTKBasicFiltersGeneratedSource} PROPERTIES GENERATED 1 )

# these files are the instantiated ITK filter, but they do not do IO,
# so we don't need to register the IO factoried here.
set ( ITK_NO_IO_FACTORY_REGISTER_MANAGER 1 )
include ( ${ITK_USE_FILE} )
include ( ${RTK_USE_FILE} )

# common source which all basic filter libraries need to be linked against
set ( SimpleRTKBasicFilters0Source
  srtkImageFilter.cxx
)

add_library ( SimpleRTKBasicFilters0 ${SimpleRTKBasicFilters0Source} )
target_link_libraries ( SimpleRTKBasicFilters0 SimpleRTKCommon ${ITK_LIBRARIES} )
set_target_properties( SimpleRTKBasicFilters0  PROPERTIES SKIP_BUILD_RPATH TRUE )
srtk_install_exported_target( SimpleRTKBasicFilters0 )

# append this new library to the globally cached list
list ( APPEND SimpleRTK_LIBRARIES SimpleRTKBasicFilters0 )
set ( SimpleRTK_LIBRARIES ${SimpleRTK_LIBRARIES} CACHE INTERNAL "" )


set ( SimpleRTKBasicFiltersSource
  srtkCastImageFilter-2.cxx
  srtkCastImageFilter-2l.cxx
  srtkCastImageFilter-2v.cxx
  srtkCastImageFilter-3.cxx
  srtkCastImageFilter-3l.cxx
  srtkCastImageFilter-3v.cxx
  srtkCastImageFilter.cxx
  srtkHashImageFilter.cxx
  srtkCreateInterpolator.hxx
  srtkMeasurementMap.cxx
  srtkStatisticsImageFilter.cxx
)
# sort the list to en-sure the same list across platforms
list ( SORT SimpleRTKBasicFiltersGeneratedSource )

# add additional files which may depend on Filters0
list ( APPEND SimpleRTKBasicFiltersSource ${SimpleRTKBasicFiltersGeneratedSource} )
list ( APPEND SimpleRTKBasicFiltersSource srtkAdditionalProcedures.cxx )

list ( LENGTH SimpleRTKBasicFiltersSource _length )
math( EXPR _end_range "${_length} - 1 " )

# The other filters an automatically split into files. They all
# use the following for export their interface. There must not be and
# inter-filter include.
if(BUILD_SHARED_LIBS)
  add_definitions(-DSimpleRTKBasicFilters_EXPORTS)
endif()

# Break all these source files into a bunch of libraries base on every
# n files in the list.
set ( _stride 25 )
set ( _library_i 0 )
foreach ( _start RANGE 0 ${_end_range} ${_stride} )

  math( EXPR _library_i "${_library_i} + 1 " )

  # compute the end of the range ensuring that the inclusive range
  # [_start, _end] is all valid indexes
  math( EXPR _end "${_start} + ${_stride} - 1" )
  if ( ${_end} VERSION_GREATER ${_end_range} OR
      ${_end} VERSION_EQUAL ${_end_range} )
    math( EXPR _end "${_end_range}" )
    set(_last 1)
  endif()

  # extract out the range into SRC
  set ( SRC "" )
  foreach( _j RANGE ${_start} ${_end} )
    list ( GET SimpleRTKBasicFiltersSource ${_j} _v )
    list ( APPEND SRC ${_v} )
  endforeach()

  add_library ( SimpleRTKBasicFilters${_library_i} ${SRC} )
  target_link_libraries ( SimpleRTKBasicFilters${_library_i} SimpleRTKCommon SimpleRTKBasicFilters0 ${ITK_LIBRARIES} )
  if(_last)
    # the last library include additional cxx files which may depend on
    # other filter libraries
    target_link_libraries ( SimpleRTKBasicFilters${_library_i} ${SimpleRTK_LIBRARIES} )
  endif()
  set_target_properties( SimpleRTKBasicFilters${_library_i}  PROPERTIES SKIP_BUILD_RPATH TRUE )
  add_dependencies ( SimpleRTKBasicFilters${_library_i} BasicFiltersSourceCode )

  # append this new library to the globaly cached list
  list ( APPEND SimpleRTK_LIBRARIES SimpleRTKBasicFilters${_library_i} )
  set ( SimpleRTK_LIBRARIES ${SimpleRTK_LIBRARIES} CACHE INTERNAL "" )

  srtk_install_exported_target( SimpleRTKBasicFilters${_library_i} )

endforeach ()


install( FILES ${CMAKE_CURRENT_BINARY_DIR}/../include/SimpleRTKBasicFiltersGeneratedHeaders.h
  DESTINATION ${SimpleRTK_INSTALL_INCLUDE_DIR}
  COMPONENT Development )
