find_package(Gengetopt)

# Common revision info between applications
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
if(GIT_SHA1 MATCHES ".*NOTFOUND") 
  set(RTK_VERSION_HASH "")
else()
  set(RTK_VERSION_HASH ", git hash ${GIT_SHA1}")
endif()
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/rtkVersion.ggo.in ${RTK_BINARY_DIR}/rtkVersion.ggo)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# add path to RTK libraries to the install path
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")
#=========================================================
# Executables
add_subdirectory(rtkprojections)
add_subdirectory(rtkparkershortscanweighting)
add_subdirectory(rtkwangdisplaceddetectorweighting)
add_subdirectory(rtkfdktwodweights)
add_subdirectory(rtkramp)
add_subdirectory(rtkfdk)
add_subdirectory(rtkiterativefdk)
add_subdirectory(rtkfourdfdk)
add_subdirectory(rtkinlinefdk)
add_subdirectory(rtkamsterdamshroud)
add_subdirectory(rtkforwardprojections)
add_subdirectory(rtkbackprojections)
add_subdirectory(rtkrayboxintersection)
add_subdirectory(rtksart)
add_subdirectory(rtkfourdsart)
add_subdirectory(rtkrayquadricintersection)
add_subdirectory(rtkprojectgeometricphantom)
add_subdirectory(rtkdrawgeometricphantom)
add_subdirectory(rtkprojectshepploganphantom)
add_subdirectory(rtkfieldofview)
add_subdirectory(rtksubselect)
add_subdirectory(rtkdrawshepploganphantom)
add_subdirectory(rtkextractshroudsignal)
add_subdirectory(rtkoverlayphaseandshroud)
add_subdirectory(rtkmedian)
add_subdirectory(rtklut)
add_subdirectory(rtktotalvariationdenoising)
add_subdirectory(rtktotalnuclearvariationdenoising)
add_subdirectory(rtkadmmwavelets)
add_subdirectory(rtkadmmtotalvariation)
add_subdirectory(rtkfourdrooster)
add_subdirectory(rtkmcrooster)
add_subdirectory(rtkfourdconjugategradient)
add_subdirectory(rtkconjugategradient)
add_subdirectory(rtkregularizedconjugategradient)
add_subdirectory(rtkwaveletsdenoising)
add_subdirectory(rtki0estimation)
add_subdirectory(rtktutorialapplication)
add_subdirectory(rtkscatterglarecorrection)
add_subdirectory(rtklastdimensionl0gradientdenoising)
add_subdirectory(rtkwarpedforwardprojectsequence)
add_subdirectory(rtkwarpedbackprojectsequence)
add_subdirectory(rtkmotioncompensatedfourdconjugategradient)
add_subdirectory(rtklagcorrection)
add_subdirectory(rtkgaincorrection)
add_subdirectory(rtkspectralsimplexdecomposition)
add_subdirectory(rtkdualenergysimplexdecomposition)
add_subdirectory(rtkspectralrooster)
add_subdirectory(rtkspectralforwardmodel)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
add_subdirectory(rtkvarianobigeometry)
add_subdirectory(rtkvarianprobeamgeometry)
add_subdirectory(rtksimulatedgeometry)
add_subdirectory(rtkelektasynergygeometry)
add_subdirectory(rtkdigisensgeometry)
add_subdirectory(rtkxradgeometry)
add_subdirectory(rtkimagxgeometry)
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
