WRAP_GGO(rtkregularizedconjugategradient_GGO_C rtkregularizedconjugategradient.ggo ../rtkinputprojections_section.ggo ../rtk3Doutputimage_section.ggo ${RTK_BINARY_DIR}/rtkVersion.ggo)
ADD_EXECUTABLE(rtkregularizedconjugategradient rtkregularizedconjugategradient.cxx ${rtkregularizedconjugategradient_GGO_C})
TARGET_LINK_LIBRARIES(rtkregularizedconjugategradient RTK)

IF (RTK_USE_CUDA)
  TARGET_LINK_LIBRARIES(rtkregularizedconjugategradient rtkcuda)
ENDIF (RTK_USE_CUDA)

# Installation code
IF(NOT RTK_INSTALL_NO_EXECUTABLES)
  FOREACH(EXE_NAME rtkregularizedconjugategradient)
    INSTALL(TARGETS ${EXE_NAME}
      RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
      LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
      ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development)
  ENDFOREACH(EXE_NAME)
ENDIF(NOT RTK_INSTALL_NO_EXECUTABLES)

