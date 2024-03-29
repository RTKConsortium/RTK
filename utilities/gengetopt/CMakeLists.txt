# Gengetopt: https://www.gnu.org/software/gengetopt/

ADD_DEFINITIONS(-DPACKAGE=\"gengetopt\")
ADD_DEFINITIONS(-DVERSION=\"2.22.4\")
ADD_DEFINITIONS(-DHAVE_SSTREAM)
ADD_DEFINITIONS(-DHAVE_NAMESPACES)

rtk_module_warnings_disable(C CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-fno-var-tracking-assignments HAS_NO_VAR_TRACKING_ASSIGNMENTS)
if (HAS_NO_VAR_TRACKING_ASSIGNMENTS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-var-tracking-assignments")
endif()

# MSVC specific issues
ADD_DEFINITIONS(-DHAVE_STRING_H)
IF(MSVC)
  ADD_DEFINITIONS(-DYY_NO_UNISTD_H)
  ADD_DEFINITIONS(-Disatty=!)
ENDIF(MSVC)

ADD_EXECUTABLE(gengetopt
  ./cmdline.c
  ./ggos.cpp
  ./scanner.cc
  ./skels/required_option.cc
  ./skels/handle_version.cc
  ./skels/generic_option.cc
  ./skels/clear_arg.cc
  ./skels/update_given.cc
  ./skels/dependant_option.cc
  ./skels/custom_getopt_gen.cc
  ./skels/handle_help.cc
  ./skels/given_field.cc
  ./skels/init_args_info.cc
  ./skels/multiple_opt_list.cc
  ./skels/free_list.cc
  ./skels/exit_failure.cc
  ./skels/clear_given.cc
  ./skels/free_multiple.cc
  ./skels/enum_decl.cc
  ./skels/reset_group.cc
  ./skels/c_source.cc
  ./skels/group_option.cc
  ./skels/file_save.cc
  ./skels/check_modes.cc
  ./skels/copyright.cc
  ./skels/option_arg.cc
  ./skels/group_counter.cc
  ./skels/free_string.cc
  ./skels/header.cc
  ./skels/print_help_string.cc
  ./skels/multiple_fill_array.cc
  ./skels/file_save_multiple.cc
  ./fileutils.cpp
  ./getopt.c
  ./gm.cc
  ./globals.cpp
  ./yywrap.c
  ./getopt1.c
  ./argsdef.c
  ./gm_utils.cpp
  ./acceptedvalues.cpp
  ./gengetopt.cc
  ./yyerror.cc
  ./parser.cc)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(skels)
#INCLUDE_DIRECTORIES(includes)
