WRAP_GGO(rtksart_GGO_C rtksart.ggo)
ADD_EXECUTABLE(rtksart rtksart.cxx ${rtksart_GGO_C})
TARGET_LINK_LIBRARIES(rtksart RTK)
SET(RTK_EXECUTABLE_TARGETS ${RTK_EXECUTABLE_TARGETS} rtksart)

