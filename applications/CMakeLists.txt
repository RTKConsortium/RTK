FIND_PACKAGE(Gengetopt)

# Common revision info between applications
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if(GIT_SHA1 MATCHES ".*NOTFOUND") 
  SET(RTK_VERSION_HASH "")
else()
  SET(RTK_VERSION_HASH ", git hash ${GIT_SHA1}")
endif()
CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/rtkVersion.ggo.in ${RTK_BINARY_DIR}/rtkVersion.ggo)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# add path to RTK libraries to the install path
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")
#=========================================================
# Executables
ADD_SUBDIRECTORY(rtkprojections)
ADD_SUBDIRECTORY(rtkparkershortscanweighting)
ADD_SUBDIRECTORY(rtkwangdisplaceddetectorweighting)
ADD_SUBDIRECTORY(rtkfdktwodweights)
ADD_SUBDIRECTORY(rtkramp)
ADD_SUBDIRECTORY(rtkfdk)
ADD_SUBDIRECTORY(rtkiterativefdk)
ADD_SUBDIRECTORY(rtkfourdfdk)
ADD_SUBDIRECTORY(rtkinlinefdk)
ADD_SUBDIRECTORY(rtkamsterdamshroud)
ADD_SUBDIRECTORY(rtkforwardprojections)
ADD_SUBDIRECTORY(rtkbackprojections)
ADD_SUBDIRECTORY(rtkrayboxintersection)
ADD_SUBDIRECTORY(rtksart)
ADD_SUBDIRECTORY(rtkfourdsart)
ADD_SUBDIRECTORY(rtkrayquadricintersection)
ADD_SUBDIRECTORY(rtkprojectgeometricphantom)
ADD_SUBDIRECTORY(rtkdrawgeometricphantom)
ADD_SUBDIRECTORY(rtkprojectshepploganphantom)
ADD_SUBDIRECTORY(rtkfieldofview)
ADD_SUBDIRECTORY(rtksubselect)
ADD_SUBDIRECTORY(rtkdrawshepploganphantom)
ADD_SUBDIRECTORY(rtkextractshroudsignal)
ADD_SUBDIRECTORY(rtkoverlayphaseandshroud)
ADD_SUBDIRECTORY(rtkmedian)
ADD_SUBDIRECTORY(rtklut)
ADD_SUBDIRECTORY(rtktotalvariationdenoising)
ADD_SUBDIRECTORY(rtkadmmwavelets)
ADD_SUBDIRECTORY(rtkadmmtotalvariation)
ADD_SUBDIRECTORY(rtkfourdrooster)
ADD_SUBDIRECTORY(rtkmcrooster)
ADD_SUBDIRECTORY(rtkfourdconjugategradient)
ADD_SUBDIRECTORY(rtkconjugategradient)
ADD_SUBDIRECTORY(rtkregularizedconjugategradient)
ADD_SUBDIRECTORY(rtkwaveletsdenoising)
ADD_SUBDIRECTORY(rtki0estimation)
ADD_SUBDIRECTORY(rtktutorialapplication)
ADD_SUBDIRECTORY(rtkscatterglarecorrection)
ADD_SUBDIRECTORY(rtklastdimensionl0gradientdenoising)
ADD_SUBDIRECTORY(rtkwarpedforwardprojectsequence)
ADD_SUBDIRECTORY(rtkwarpedbackprojectsequence)
ADD_SUBDIRECTORY(rtkmotioncompensatedfourdconjugategradient)
ADD_SUBDIRECTORY(rtklagcorrection)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
ADD_SUBDIRECTORY(rtkvarianobigeometry)
ADD_SUBDIRECTORY(rtksimulatedgeometry)
ADD_SUBDIRECTORY(rtkelektasynergygeometry)
ADD_SUBDIRECTORY(rtkdigisensgeometry)
ADD_SUBDIRECTORY(rtkxradgeometry)
ADD_SUBDIRECTORY(rtkimagxgeometry)
#=========================================================

# Testing of the applications
if(BUILD_TESTING)

  add_test(rtkappsimulatedgeometrytest ${EXECUTABLE_OUTPUT_PATH}/rtksimulatedgeometry -n 180 --sid 1000 --sdd 1500 -o geo)

  add_test(rtkappprojectshepploganphantomtest ${EXECUTABLE_OUTPUT_PATH}/rtkprojectshepploganphantom  -o sheppy.mha -g geo --phantomscale 40 --dimension 128)
  set_tests_properties(rtkappprojectshepploganphantomtest PROPERTIES DEPENDS rtkappsimulatedgeometrytest)
 
  if(RTK_USE_CUDA AND CUDA_HAVE_GPU)
    add_test(rtkappfdkcudatest ${EXECUTABLE_OUTPUT_PATH}/rtkfdk -g geo -p . -r sheppy.mha -o fdk_gpu.mha --hardware cuda)
	set_tests_properties(rtkappfdkcudatest PROPERTIES DEPENDS rtkappprojectshepploganphantomtest)
    add_test(rtkappfdkchecktest ${EXECUTABLE_OUTPUT_PATH}/rtkcheckimagequality fdk_gpu.mha)
    set_tests_properties(rtkappfdkchecktest PROPERTIES DEPENDS rtkappfdkcudatest)
  else()
    # Only run the software version if 
    if(NOT FAST_TESTS_NO_CHECKS)
      add_test(rtkappfdktest ${EXECUTABLE_OUTPUT_PATH}/rtkfdk -g geo -p . -r sheppy.mha -o fdk_gpu.mha)
	  set_tests_properties(rtkappfdktest PROPERTIES DEPENDS rtkappprojectshepploganphantomtest)
      add_test(rtkappfdkchecktest ${EXECUTABLE_OUTPUT_PATH}/rtkcheckimagequality fdk_gpu.mha)
      set_tests_properties(rtkappfdkchecktest PROPERTIES DEPENDS rtkappfdktest)
	endif()
  endif()
 


endif()
