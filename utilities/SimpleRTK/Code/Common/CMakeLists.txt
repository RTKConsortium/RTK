add_subdirectory( src )

configure_file( "src/srtkConfigure.h.in" "include/srtkConfigure.h" )
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/srtkConfigure.h"
  DESTINATION "${SimpleRTK_INSTALL_INCLUDE_DIR}"
  COMPONENT Development)

configure_file("src/srtkVersionConfig.h.in" "include/srtkVersionConfig.h" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/srtkVersionConfig.h"
  DESTINATION "${SimpleRTK_INSTALL_INCLUDE_DIR}"
  COMPONENT Development)
