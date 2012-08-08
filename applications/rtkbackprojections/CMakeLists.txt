WRAP_GGO(rtkbackprojections_GGO_C rtkbackprojections.ggo)
ADD_EXECUTABLE(rtkbackprojections rtkbackprojections.cxx ${rtkbackprojections_GGO_C})
TARGET_LINK_LIBRARIES(rtkbackprojections RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtkbackprojections)

