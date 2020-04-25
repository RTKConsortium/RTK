macro(rtk_module_warnings_disable)
  # Remove compiler warnings flags for the languages sent as argument.
  #
  # Mirrors itk_module_warnings_disable to avoid inclusion of ITKModuleMacros
  # where the macro is defined. ITKModuleMacros has the side effect of adding
  # KWStyle/ClangFormat external projects so it should only be included once
  # per ITK External/Remote module through the inclusion of ITKModuleExternal.
  foreach(lang ${ARGN})
    if(MSVC)
      string(REGEX REPLACE "(^| )[/-]W[0-4]( |$)" " "
        CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS}")
      set(CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS} /W0")
    elseif(BORLAND)
      set(CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS} -w-")
    else()
      set(CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS} -w")
    endif()
  endforeach()
endmacro()
