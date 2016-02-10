WRAP_GGO(rtkoverlayphaseandshroud_GGO_C rtkoverlayphaseandshroud.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkoverlayphaseandshroud rtkoverlayphaseandshroud.cxx ${rtkoverlayphaseandshroud_GGO_C})
TARGET_LINK_LIBRARIES(rtkoverlayphaseandshroud RTK)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkoverlayphaseandshroud)
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME)
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

