WRAP_GGO(rtksimulatedgeometry_GGO_C rtksimulatedgeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtksimulatedgeometry rtksimulatedgeometry.cxx ${rtksimulatedgeometry_GGO_C})
target_link_libraries(rtksimulatedgeometry RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtksimulatedgeometry) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

