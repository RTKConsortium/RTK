WRAP_GGO(rtkadmmtotalvariation_GGO_C rtkadmmtotalvariation.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkadmmtotalvariation rtkadmmtotalvariation.cxx ${rtkadmmtotalvariation_GGO_C})
target_link_libraries(rtkadmmtotalvariation RTK)

if (RTK_USE_CUDA)
  target_link_libraries(rtkadmmtotalvariation rtkcuda)
endif()

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkadmmtotalvariation) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

