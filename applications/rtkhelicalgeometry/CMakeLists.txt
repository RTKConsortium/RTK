WRAP_GGO(rtkhelicalgeometry_GGO_C rtkhelicalgeometry.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkhelicalgeometry rtkhelicalgeometry.cxx ${rtkhelicalgeometry_GGO_C})
target_link_libraries(rtkhelicalgeometry RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkhelicalgeometry)
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach()

  # Install Python application
  install(FILES rtkhelicalgeometry.py
    DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT PythonWheelRuntimeLibraries)
endif()

