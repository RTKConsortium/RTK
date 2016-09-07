WRAP_GGO(rtkwarpedbackprojectsequence_GGO_C rtkwarpedbackprojectsequence.ggo ../rtkinputprojections_section.ggo ../rtk4Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkwarpedbackprojectsequence rtkwarpedbackprojectsequence.cxx ${rtkwarpedbackprojectsequence_GGO_C})
target_link_libraries(rtkwarpedbackprojectsequence RTK)

if (RTK_USE_CUDA)
  target_link_libraries(rtkwarpedbackprojectsequence rtkcuda)
endif ()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkwarpedbackprojectsequence)
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

