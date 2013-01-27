WRAP_GGO(rtkextractshroudsignal_GGO_C rtkextractshroudsignal.ggo)
ADD_EXECUTABLE(rtkextractshroudsignal rtkextractshroudsignal.cxx ${rtkextractshroudsignal_GGO_C})
TARGET_LINK_LIBRARIES(rtkextractshroudsignal RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkextractshroudsignal) 
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME) 
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

