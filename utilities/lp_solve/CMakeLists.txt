set(LPSOLVE_VERSION_MAJOR 5)
set(LPSOLVE_VERSION_MINOR 5)
set(LPSOLVE_VERSION_PATCH 2)
set(LPSOLVE_VERSION ${LPSOLVE_VERSION_MAJOR}.${LPSOLVE_VERSION_MINOR}.${LPSOLVE_VERSION_PATCH})

if(NOT ITK_SOURCE_DIR)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleMacros)
endif()
itk_module_warnings_disable(C CXX)

# Set up install directories
add_library(lpsolve55
	lp_MDO.c
	shared/commonlib.c
	shared/mmio.c
	shared/myblas.c
	ini.c
	colamd/colamd.c
	lp_rlp.c
	lp_crash.c
	bfp/bfp_LUSOL/lp_LUSOL.c
	bfp/bfp_LUSOL/LUSOL/lusol.c
	lp_Hash.c
	lp_lib.c
	lp_wlp.c
	lp_matrix.c
	lp_mipbb.c
	lp_MPS.c
	lp_params.c
	lp_presolve.c
	lp_price.c
	lp_pricePSE.c
	lp_report.c
	lp_scale.c
	lp_simplex.c
	lp_SOS.c
	lp_utils.c
	yacc_read.c
)

set_property(TARGET lpsolve55 PROPERTY PUBLIC_HEADER
	lp_bit.h
	lp_crash.h
	lp_explicit.h
	lp_fortify.h
	lp_Hash.h
	lp_lib.h
	lp_matrix.h
	lp_MDO.h
	lp_mipbb.h
	lp_MPS.h
	lp_presolve.h
	lp_price.h
	lp_pricePSE.h
	lp_report.h
	lp_rlp.h
	lp_scale.h
	lp_simplex.h
	lp_solveDLL.h
	lp_SOS.h
	lp_types.h
	lp_utils.h
	lp_wlp.h
)

include_directories(
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shared>"
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bfp>"
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bfp/bfp_LUSOL>"
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bfp/bfp_LUSOL/LUSOL>"
	"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/colamd>"
)

set_property(TARGET lpsolve55 APPEND PROPERTY COMPILE_DEFINITIONS
	YY_NEVER_INTERACTIVE
	PARSER_LP
	INVERSE_ACTIVE=INVERSE_LUSOL
	RoleIsExternalInvEngine
)

if(WIN32)
    set_property(TARGET lpsolve55 APPEND PROPERTY COMPILE_DEFINITIONS
                INLINE=static
		_USRDLL
		WIN32
    )
endif()

include(CheckSymbolExists)
check_symbol_exists(isnan math.h HAVE_ISNAN)
if(NOT HAVE_ISNAN)
	set_property(TARGET lpsolve55 APPEND PROPERTY COMPILE_DEFINITIONS NOISNAN)
endif()

# Set version of library
set_target_properties(lpsolve55 PROPERTIES VERSION ${LPSOLVE_VERSION}
                                           SOVERSION ${LPSOLVE_VERSION_MAJOR})

#=========================================================
if(NOT RTK_INSTALL_NO_LIBRARIES)
  target_include_directories(lpsolve55 PUBLIC $<INSTALL_INTERFACE:${RTK_INSTALL_INCLUDE_DIR}/lpsolve>)
  install(TARGETS lpsolve55 EXPORT RTK
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}/lpsolve"
  )
  install(TARGETS lpsolve55 EXPORT ITKTargets
    RUNTIME DESTINATION ${RTK_INSTALL_RUNTIME_DIR} COMPONENT Runtime
    LIBRARY DESTINATION ${RTK_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${RTK_INSTALL_ARCHIVE_DIR} COMPONENT Development
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_INCLUDE_DIR}/lpsolve"
  )
endif()
