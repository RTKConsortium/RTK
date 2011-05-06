FIND_PROGRAM(GENGETOPT gengetopt)
IF (GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
  ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/cmake/gengetopt)
  GET_TARGET_PROPERTY(GENGETOPT gengetopt OUTPUT_NAME)
ELSE(GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
  ADD_CUSTOM_TARGET(gengetopt DEPENDS ${GENGETOPT})
ENDIF(GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")

MACRO (WRAP_GGO GGO_SRCS)
  FOREACH(GGO_FILE ${ARGN})
    GET_FILENAME_COMPONENT(GGO_BASEFILENAME ${GGO_FILE} NAME_WE)
    GET_FILENAME_COMPONENT(GGO_FILE_ABS ${GGO_FILE} ABSOLUTE)
    SET(GGO_H ${GGO_BASEFILENAME}_ggo.h)
    SET(GGO_C ${GGO_BASEFILENAME}_ggo.c)
    SET(GGO_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${GGO_H} ${CMAKE_CURRENT_BINARY_DIR}/${GGO_C})
    ADD_CUSTOM_COMMAND(OUTPUT ${GGO_OUTPUT}
                       COMMAND ${GENGETOPT}
                       ARGS < ${GGO_FILE_ABS}
                              --output-dir=${CMAKE_CURRENT_BINARY_DIR}
                              --arg-struct-name=args_info_${GGO_BASEFILENAME}
                              --func-name=cmdline_parser_${GGO_BASEFILENAME}
                              --file-name=${GGO_BASEFILENAME}_ggo
                              --unamed-opts
                              --conf-parser
                              --include-getopt
                       DEPENDS ${GGO_FILE} ${GENGETOPT}
                      )
    SET(${GGO_SRCS} ${${GGO_SRCS}} ${GGO_OUTPUT})
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
  ENDFOREACH(GGO_FILE)
  SET_SOURCE_FILES_PROPERTIES(${${GGO_SRCS}} PROPERTIES GENERATED TRUE)
ENDMACRO (WRAP_GGO)
