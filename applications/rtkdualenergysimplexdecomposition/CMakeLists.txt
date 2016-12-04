WRAP_GGO(rtkdualenergysimplexdecomposition_GGO_C rtkdualenergysimplexdecomposition.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkdualenergysimplexdecomposition rtkdualenergysimplexdecomposition.cxx ${rtkdualenergysimplexdecomposition_GGO_C})
target_link_libraries(rtkdualenergysimplexdecomposition RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkdualenergysimplexdecomposition) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()
