WRAP_GGO(rtksubselect_GGO_C rtksubselect.ggo)
ADD_EXECUTABLE(rtksubselect rtksubselect.cxx ${rtksubselect_GGO_C})
TARGET_LINK_LIBRARIES(rtksubselect RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtksubselect)

