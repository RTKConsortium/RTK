WRAP_GGO(rtkamsterdamshroud_GGO_C rtkamsterdamshroud.ggo)
ADD_EXECUTABLE(rtkamsterdamshroud rtkamsterdamshroud.cxx ${rtkamsterdamshroud_GGO_C})
TARGET_LINK_LIBRARIES(rtkamsterdamshroud RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkamsterdamshroud)

