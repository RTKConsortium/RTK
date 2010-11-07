######################################################
##  OpenMP
######################################################
INCLUDE(CheckFunctionExists)
MESSAGE(STATUS "Check for compiler OpenMP support...")
SET(OPENMP_FLAGS)
SET(OPENMP_LIBRARIES)
SET(OPENMP_FOUND FALSE)

# Key: CFLAGS##LDFLAGS#LIBRARIES
# Neither CFLAGS nor LDFLAGS can be empty.  Use NONE instead.
SET(
  OPENMP_FLAGS_AND_LIBRARIES
  # gcc
  "-fopenmp##-fopenmp#"
  "-fopenmp##-fopenmp#gomp"
  "-fopenmp##-fopenmp#gomp pthread"
  # icc
  "-openmp##-openmp#"
  "-openmp -parallel##-openmp -parallel#"
  # SGI & PGI
  "-mp##-mp#"
  # Sun
  "-xopenmp##-xopenmp#"
  # Tru64
  "-omp##-omp#"
  # AIX
  "-qsmp=omp##-qsmp=omp#"
  # MSVC
  "/openmp##NONE#"
)

# Massive hack to workaround CMake limitations
LIST(LENGTH OPENMP_FLAGS_AND_LIBRARIES NUM_FLAGS)
MATH(EXPR NUM_FLAGS "${NUM_FLAGS} - 1")
FOREACH(I RANGE 0 ${NUM_FLAGS})
  IF(NOT OPENMP_FOUND)
    LIST(GET OPENMP_FLAGS_AND_LIBRARIES ${I} TMP)
    STRING(REGEX MATCH "([^#]*)" OPENMP_FLAGS ${TMP})
    STRING(REGEX REPLACE "[^#]*##" "" TMP ${TMP})
    STRING(REGEX MATCH "([^#]*)" OPENMP_LDFLAGS ${TMP})
    STRING(REGEX REPLACE "[^#]*#" "" OPENMP_LIBRARIES ${TMP})
    #MESSAGE(STATUS "OPENMP_FLAGS=${OPENMP_FLAGS}")
    #MESSAGE(STATUS "OPENMP_LDFLAGS = ${OPENMP_LDFLAGS}")
    #MESSAGE(STATUS "OPENMP_LIBRARIES = ${OPENMP_LIBRARIES}")
    #MESSAGE(STATUS "-------")

    IF(OPENMP_LDFLAGS MATCHES "NONE")
      SET(OPENMP_LDFLAGS "")
    ENDIF(OPENMP_LDFLAGS MATCHES "NONE")
    IF(OPENMP_LIBRARIES MATCHES " ")
      STRING(REPLACE " " ";" OPENMP_LIBRARIES ${OPENMP_LIBRARIES})
    ENDIF(OPENMP_LIBRARIES MATCHES " ")

    ## I think I need to do a try-compile
    SET(CMAKE_REQUIRED_FLAGS ${OPENMP_FLAGS})
    SET(CMAKE_REQUIRED_LIBRARIES ${OPENMP_LIBRARIES})
    CHECK_FUNCTION_EXISTS(omp_get_thread_num OPENMP_FOUND${I})

    IF(OPENMP_FOUND${I})
      SET(OPENMP_FOUND TRUE)
    ENDIF(OPENMP_FOUND${I})
  ENDIF(NOT OPENMP_FOUND)
ENDFOREACH(I RANGE 0 ${NUM_FLAGS})

IF(OPENMP_FOUND)
  MESSAGE(STATUS "OpenMP flags \"${OPENMP_FLAGS}\", OpenMP libraries \"${OPENMP_LIBRARIES}\"")
ELSE(OPENMP_FOUND)
  MESSAGE(STATUS "Given compiler does not support OpenMP.")
ENDIF(OPENMP_FOUND)
