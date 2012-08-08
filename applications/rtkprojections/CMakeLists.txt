WRAP_GGO(rtkprojections_GGO_C rtkprojections.ggo)
ADD_EXECUTABLE(rtkprojections rtkprojections.cxx ${rtkprojections_GGO_C})
TARGET_LINK_LIBRARIES(rtkprojections RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkprojections)

