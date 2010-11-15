# Find gengetopt
FIND_PROGRAM(GENGETOPT gengetopt)
IF (GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
  MESSAGE("gengetopt not found, you will not be able to add/modify .ggo files")
ENDIF (GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")

# If set, process ggo files (http://www.gnu.org/software/gengetopt/)
MACRO (WRAP_GGO GGO_SRCS)
  FOREACH(GGO_FILE ${ARGN})
    GET_FILENAME_COMPONENT(GGO_BASEFILENAME ${GGO_FILE} NAME_WE)
    GET_FILENAME_COMPONENT(GGO_FILE_ABS ${GGO_FILE} ABSOLUTE)
    SET(GGO_H ${GGO_BASEFILENAME}_ggo.h)
    SET(GGO_C ${GGO_BASEFILENAME}_ggo.c)
    SET(GGO_OUTPUT ${PROJECT_SOURCE_DIR}/${GGO_H} ${PROJECT_SOURCE_DIR}/${GGO_C})
    IF (NOT GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
      ADD_CUSTOM_COMMAND(OUTPUT ${GGO_OUTPUT}
                         COMMAND ${GENGETOPT}
                         ARGS < ${GGO_FILE_ABS}
                                --output-dir=${PROJECT_SOURCE_DIR}
                                --arg-struct-name=args_info_${GGO_BASEFILENAME}
                                --func-name=cmdline_parser_${GGO_BASEFILENAME}
                                --file-name=${GGO_BASEFILENAME}_ggo
                                --unamed-opts
                                --conf-parser
                                --include-getopt
                         DEPENDS ${GGO_FILE}
                         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        )
    ENDIF (NOT GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
    SET(${GGO_SRCS} ${${GGO_SRCS}} ${GGO_OUTPUT})
  ENDFOREACH(GGO_FILE)
  SET_SOURCE_FILES_PROPERTIES(${${GGO_SRCS}} PROPERTIES GENERATED TRUE)
ENDMACRO (WRAP_GGO)
