#=========================================================
# Help function to debug CMake
MACRO (DD in)
    MESSAGE(${in}=${${in}})
ENDMACRO(DD)
#=========================================================

FILE(GLOB ALL_CPP_FILES *.cpp)
ADD_LIBRARY(rtkXbase STATIC ${ALL_CPP_FILES})

