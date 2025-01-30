if(RTK_BUILD_SPHINX)

  # Build sphinx documentation in the build directory
  set(RTK_DOC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/sphinx)

  # Copy the source tree documentation to the build tree before calling sphinx.
  # The images are downloaded when conf.py is run by sphinx.
  add_custom_target(copy_sources ALL
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${RTK_DOC_OUTPUT_DIR}/examples"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${RTK_DOC_OUTPUT_DIR}/applications"
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${RTK_DOC_OUTPUT_DIR}/documentation"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RTK_SOURCE_DIR}/examples" "${RTK_DOC_OUTPUT_DIR}/examples"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RTK_SOURCE_DIR}/applications" "${RTK_DOC_OUTPUT_DIR}/applications"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${RTK_SOURCE_DIR}/documentation" "${RTK_DOC_OUTPUT_DIR}/documentation"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/conf.py" "${RTK_DOC_OUTPUT_DIR}/conf.py"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/index.md" "${RTK_DOC_OUTPUT_DIR}/index.md"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/GettingStarted.md" "${RTK_DOC_OUTPUT_DIR}/GettingStarted.md"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/INSTALLATION.md" "${RTK_DOC_OUTPUT_DIR}/INSTALLATION.md"
    COMMAND ${CMAKE_COMMAND} -E copy "${RTK_SOURCE_DIR}/CodeContribution.md" "${RTK_DOC_OUTPUT_DIR}/CodeContribution.md"
    COMMENT "Copying documentation sources"
    )

  find_program(SPHINX_EXECUTABLE
    NAMES sphinx-build sphinx-build.exe
    DOC "Sphinx documentation generator"
  )

  add_custom_target( sphinx_doc ALL
    COMMAND ${SPHINX_EXECUTABLE}
    -c ${RTK_DOC_OUTPUT_DIR}                   # conf.py directory
    ${RTK_DOC_OUTPUT_DIR}                      # Source directory containing root index.md
    "${RTK_DOC_OUTPUT_DIR}/_build"             # Output directory
    COMMENT "Generating sphinx documentation"
    DEPENDS copy_sources
    WORKING_DIRECTORY ${RTK_DOC_OUTPUT_DIR}
    )

  set_property(
    DIRECTORY APPEND PROPERTY
    ADDITIONAL_MAKE_CLEAN_FILES
    "${RTK_DOC_OUTPUT_DIR}/_build"
    )

endif()
