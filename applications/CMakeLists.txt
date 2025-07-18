find_package(Gengetopt)

configure_file (${CMAKE_CURRENT_SOURCE_DIR}/rtkVersion.ggo.in ${RTK_BINARY_DIR}/rtkVersion.ggo)

#-----------------------------------------------------------------------------
# Setup RPATH :
#
# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# add path to RTK libraries to the install path
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${RTK_INSTALL_LIB_DIR}")

#-----------------------------------------------------------------------------
# Find ITK.
# Required to include ITK_USE_FILE in order to Register IO factories
# Force requested modules to be RTK dependencies only, otherwise all
# available factories will try to register themselves.
if (NOT ITK_DIR)
  set(ITK_DIR ${ITK_BINARY_DIR}/CMakeTmp)
endif()
find_package(ITK REQUIRED COMPONENTS ${ITK_MODULE_RTK_DEPENDS})
include(${ITK_USE_FILE})

#-----------------------------------------------------------------------------
# Executables
add_subdirectory(rtkamsterdamshroud)
add_subdirectory(rtkbackprojections)
add_subdirectory(rtkfdk)
add_subdirectory(rtkfdktwodweights)
add_subdirectory(rtkfieldofview)
add_subdirectory(rtkforwardprojections)
add_subdirectory(rtkfourdfdk)
add_subdirectory(rtkfourdsart)
add_subdirectory(rtkiterativefdk)
add_subdirectory(rtkosem)
add_subdirectory(rtkparkershortscanweighting)
add_subdirectory(rtkprojectgeometricphantom)
add_subdirectory(rtkprojections)
add_subdirectory(rtkprojectshepploganphantom)
add_subdirectory(rtkramp)
add_subdirectory(rtkrayboxintersection)
add_subdirectory(rtkrayquadricintersection)
add_subdirectory(rtksart)
add_subdirectory(rtksubselect)
add_subdirectory(rtkwangdisplaceddetectorweighting)

add_subdirectory(rtkdrawgeometricphantom)
add_subdirectory(rtkdrawshepploganphantom)
add_subdirectory(rtkextractshroudsignal)
add_subdirectory(rtkextractphasesignal)
add_subdirectory(rtkoverlayphaseandshroud)
add_subdirectory(rtklut)
add_subdirectory(rtktotalvariationdenoising)
add_subdirectory(rtktotalnuclearvariationdenoising)
add_subdirectory(rtkadmmwavelets)
add_subdirectory(rtkadmmtotalvariation)
add_subdirectory(rtkfourdrooster)
add_subdirectory(rtkmcrooster)
add_subdirectory(rtkfourdconjugategradient)
add_subdirectory(rtkspectralonestep)
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
add_subdirectory(rtkmaskcollimation)
add_subdirectory(rtkspectraldenoiseprojections)
add_subdirectory(rtkprojectionmatrix)
add_subdirectory(rtkvectorconjugategradient)
add_subdirectory(rtkdualenergyforwardmodel)

add_subdirectory(rtkcheckimagequality)

#All the executables below are meant to create RTK ThreeDCircularProjectionGeometry files
add_subdirectory(rtkvarianobigeometry)
add_subdirectory(rtkvarianprobeamgeometry)
add_subdirectory(rtksimulatedgeometry)
add_subdirectory(rtkelektasynergygeometry)
add_subdirectory(rtkdigisensgeometry)
add_subdirectory(rtkxradgeometry)
add_subdirectory(rtkimagxgeometry)
add_subdirectory(rtkorageometry)
add_subdirectory(rtkbioscangeometry)
#=========================================================

#-----------------------------------------------------------------------------
# Testing of the applications
if(BUILD_TESTING)
  # Reference data (geometry, reference Shepp Logan and corresponding projections)
  add_test(rtkappsimulatedgeometrytest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtksimulatedgeometry -n 45 --sid 1000 --sdd 1500 -o geo)

  add_test(rtkappdrawshepploganphantomtest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkdrawshepploganphantom -o reference.mha --size 21 --phantomscale 10)

  add_test(rtkappprojectshepploganphantomtest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkprojectshepploganphantom  -o sheppy.mha -g geo --phantomscale 10 --size 32)
  set_tests_properties(rtkappprojectshepploganphantomtest PROPERTIES DEPENDS rtkappsimulatedgeometrytest)

  # FDK test
  if(RTK_USE_CUDA)
    add_test(rtkappfdktest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkfdk -g geo -p . -r sheppy.mha -o fdk.mha --hardware cuda --size 21)
  else()
    add_test(rtkappfdktest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkfdk -g geo -p . -r sheppy.mha -o fdk.mha --size 21)
  endif()
  set_tests_properties(rtkappfdktest PROPERTIES DEPENDS rtkappprojectshepploganphantomtest)

  add_test(rtkappfdkchecktest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkcheckimagequality -i reference.mha -j fdk.mha -t 400)
  set_tests_properties(rtkappfdkchecktest PROPERTIES DEPENDS "rtkappdrawshepploganphantomtest;rtkappfdktest")

  # Iteration reporting testing
  add_test(rtkapposemtest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkosem -g geo -p . -r sheppy.mha -o osem.mha -n 3 --size 21 --output-every 1 --nprojpersubset 15 --iteration-file-name osem%d.mha)
  set_tests_properties(rtkapposemtest PROPERTIES DEPENDS rtkappprojectshepploganphantomtest)

  add_test(rtkapposemchecktest ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rtkcheckimagequality -i reference.mha -j osem1.mha,osem2.mha,osem3.mha -t 800,550,455)
  set_tests_properties(rtkapposemchecktest PROPERTIES DEPENDS "rtkapposemtest;rtkappdrawshepploganphantomtest")
endif()
