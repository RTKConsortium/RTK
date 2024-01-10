set(LPSOLVE_VERSION_MAJOR 5)
set(LPSOLVE_VERSION_MINOR 5)
set(LPSOLVE_VERSION_PATCH 2)
set(LPSOLVE_VERSION ${LPSOLVE_VERSION_MAJOR}.${LPSOLVE_VERSION_MINOR}.${LPSOLVE_VERSION_PATCH})

rtk_module_warnings_disable(C CXX)

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
        LoadInverseLib=FALSE
        LoadLanguageLib=FALSE
        LoadableBlasLib=0
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

if (NOT ITK_SOURCE_DIR)
  # Export to RTK build targets must be done after RTK has been loaded by ITK.
  # When building externally we must do this after including ITKModuleExternal.
  # See RTK/CMakeLists.txt.
else()
  itk_module_target(lpsolve55)
endif()
