WRAP_GGO(rtkextractshroudsignal_GGO_C rtkextractshroudsignal.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
add_executable(rtkextractshroudsignal rtkextractshroudsignal.cxx ${rtkextractshroudsignal_GGO_C})
target_link_libraries(rtkextractshroudsignal RTK)

# Installation code
if(NOT RTK_INSTALL_NO_EXECUTABLES)
  foreach(EXE_NAME rtkextractshroudsignal) 
    install(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  endforeach() 
endif()

