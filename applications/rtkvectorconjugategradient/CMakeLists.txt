WRAP_GGO(rtkvectorconjugategradient_GGO_C rtkvectorconjugategradient.ggo ../rtk3Doutputimage_section.ggo ../rtkprojectors_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkvectorconjugategradient rtkvectorconjugategradient.cxx ${rtkvectorconjugategradient_GGO_C})
target_link_libraries(rtkvectorconjugategradient RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkvectorconjugategradient) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

