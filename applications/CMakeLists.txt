set(
  GENGETOPT_HEADER
  "version \\\"${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}.${RTK_VERSION_PATCH}${RTK_VERSION_HASH}\\\"
option \\\"verbose\\\" v \\\"Verbose execution\\\" flag off
"
)

find_package(Gengetopt)

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
if(NOT ITK_DIR)
  set(ITK_DIR ${ITK_BINARY_DIR}/CMakeTmp)
endif()
find_package(ITK REQUIRED COMPONENTS ${ITK_MODULE_RTK_DEPENDS})
include(${ITK_USE_FILE})

#-----------------------------------------------------------------------------
# Executables
add_subdirectory(rtkamsterdamshroud)
add_subdirectory(rtkbackprojections)
add_subdirectory(rtkfdk)
add_subdirectory(rtkfieldofview)
add_subdirectory(rtkforwardprojections)
add_subdirectory(rtkfourdfdk)
add_subdirectory(rtkfourdsart)
add_subdirectory(rtkiterativefdk)
add_subdirectory(rtkosem)
add_subdirectory(rtkprojectgeometricphantom)
add_subdirectory(rtkprojections)
add_subdirectory(rtkprojectshepploganphantom)
add_subdirectory(rtksart)
add_subdirectory(rtksubselect)

add_subdirectory(rtkdrawgeometricphantom)
add_subdirectory(rtkdrawshepploganphantom)
add_subdirectory(rtkextractshroudsignal)
add_subdirectory(rtkextractphasesignal)
add_subdirectory(rtkoverlayphaseandshroud)
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
add_subdirectory(rtkmotioncompensatedfourdconjugategradient)
add_subdirectory(rtklagcorrection)
add_subdirectory(rtkgaincorrection)
add_subdirectory(rtkspectralsimplexdecomposition)
add_subdirectory(rtkspectralrooster)
add_subdirectory(rtkspectralforwardmodel)
add_subdirectory(rtkmaskcollimation)
add_subdirectory(rtkspectraldenoiseprojections)
add_subdirectory(rtkprojectionmatrix)
add_subdirectory(rtkvectorconjugategradient)

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
  add_test(
    rtkappsimulatedgeometrytest
    ${CMAKE_BINARY_DIR}/bin/rtksimulatedgeometry
    -n
    45
    --sid
    1000
    --sdd
    1500
    -o
    geo
  )

  add_test(
    rtkappdrawshepploganphantomtest
    ${CMAKE_BINARY_DIR}/bin/rtkdrawshepploganphantom
    -o
    reference.mha
    --size
    21
    --phantomscale
    10
  )

  add_test(
    rtkappprojectshepploganphantomtest
    ${CMAKE_BINARY_DIR}/bin/rtkprojectshepploganphantom
    -o
    sheppy.mha
    -g
    geo
    --phantomscale
    10
    --size
    32
  )
  set_tests_properties(
    rtkappprojectshepploganphantomtest
    PROPERTIES
      DEPENDS
        rtkappsimulatedgeometrytest
  )

  # FDK test
  if(RTK_USE_CUDA)
    add_test(
      rtkappfdktest
      ${CMAKE_BINARY_DIR}/bin/rtkfdk
      -g
      geo
      -p
      .
      -r
      sheppy.mha
      -o
      fdk.mha
      --hardware
      cuda
      --size
      21
    )
  else()
    add_test(
      rtkappfdktest
      ${CMAKE_BINARY_DIR}/bin/rtkfdk
      -g
      geo
      -p
      .
      -r
      sheppy.mha
      -o
      fdk.mha
      --size
      21
    )
  endif()
  set_tests_properties(
    rtkappfdktest
    PROPERTIES
      DEPENDS
        rtkappprojectshepploganphantomtest
  )

  add_test(
    rtkappfdkchecktest
    ${CMAKE_BINARY_DIR}/bin/rtkcheckimagequality
    -i
    reference.mha
    -j
    fdk.mha
    -t
    400
  )
  set_tests_properties(
    rtkappfdkchecktest
    PROPERTIES
      DEPENDS
        "rtkappdrawshepploganphantomtest;rtkappfdktest"
  )

  # Iteration reporting testing
  add_test(
    rtkapposemtest
    ${CMAKE_BINARY_DIR}/bin/rtkosem
    -g
    geo
    -p
    .
    -r
    sheppy.mha
    -o
    osem.mha
    -n
    3
    --size
    21
    --output-every
    1
    --nprojpersubset
    15
    --iteration-file-name
    osem%d.mha
  )
  set_tests_properties(
    rtkapposemtest
    PROPERTIES
      DEPENDS
        rtkappprojectshepploganphantomtest
  )

  add_test(
    rtkapposemchecktest
    ${CMAKE_BINARY_DIR}/bin/rtkcheckimagequality
    -i
    reference.mha
    -j
    osem1.mha,osem2.mha,osem3.mha
    -t
    800,550,455
  )
  set_tests_properties(
    rtkapposemchecktest
    PROPERTIES
      DEPENDS
        "rtkapposemtest;rtkappdrawshepploganphantomtest"
  )
endif()
