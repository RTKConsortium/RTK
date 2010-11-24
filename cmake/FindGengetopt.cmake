# Find gengetopt
FIND_PROGRAM(GENGETOPT gengetopt)
IF (GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
  MESSAGE("gengetopt not found, you will not be able to add/modify .ggo files")
ENDIF (GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")

# Detect gengetopt version.  It should be 2.22.4, or else 
# it is not acceptable.
IF (NOT GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")
  EXECUTE_PROCESS (COMMAND
    "${GENGETOPT}" --version
    TIMEOUT 10
    RESULT_VARIABLE GENGETOPT_RESULT
    OUTPUT_VARIABLE GENGETOPT_STDOUT
    ERROR_VARIABLE GENGETOPT_STDERR
    )
  IF (GENGETOPT_RESULT EQUAL 0)
    STRING (REGEX MATCH "GNU *gengetopt *([^\n]*)" JUNK ${GENGETOPT_STDOUT})
    SET (GENGETOPT_VERSION "${CMAKE_MATCH_1}")
    IF (GENGETOPT_VERSION STREQUAL "2.22.4")
      MESSAGE (STATUS 
	"Gengetopt version \"${GENGETOPT_VERSION}\" is acceptable.")
    ELSE (GENGETOPT_VERSION STREQUAL "2.22.4")
      MESSAGE (STATUS 
	"Gengetopt version \"${GENGETOPT_VERSION}\" is not acceptable.")
      SET (GENGETOPT "GENGETOPT-NOTFOUND")
    ENDIF (GENGETOPT_VERSION STREQUAL "2.22.4")
  ELSE (GENGETOPT_RESULT EQUAL 0)
    MESSAGE (STATUS "Could not run gengetopt --version.")
    SET (GENGETOPT "GENGETOPT-NOTFOUND")
  ENDIF (GENGETOPT_RESULT EQUAL 0)
ENDIF (NOT GENGETOPT STREQUAL "GENGETOPT-NOTFOUND")

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
