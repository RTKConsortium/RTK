WRAP_GGO(rtkramp_GGO_C rtkramp.ggo)
ADD_EXECUTABLE(rtkramp rtkramp.cxx ${rtkramp_GGO_C})
TARGET_LINK_LIBRARIES(rtkramp RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkramp)

