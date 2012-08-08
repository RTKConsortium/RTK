WRAP_GGO(rtkfdktwodweights_GGO_C rtkfdktwodweights.ggo)
ADD_EXECUTABLE(rtkfdktwodweights rtkfdktwodweights.cxx ${rtkfdktwodweights_GGO_C})
TARGET_LINK_LIBRARIES(rtkfdktwodweights RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkfdktwodweights)

