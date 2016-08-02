WRAP_GGO(rtkiterativefdk_GGO_C rtkiterativefdk.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkiterativefdk rtkiterativefdk.cxx ${rtkiterativefdk_GGO_C})
target_link_libraries(rtkiterativefdk RTK)

if(RTK_USE_CUDA)
  target_link_libraries(rtkiterativefdk rtkcuda)
endif()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkiterativefdk) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

