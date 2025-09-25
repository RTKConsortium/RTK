if(RTK_BUILD_SPHINX)
  # Build sphinx documentation in the build directory
  set(RTK_DOC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/sphinx)

  # Copy the source tree documentation to the build tree before calling sphinx.
  # The images are downloaded when conf.py is run by sphinx.
  add_custom_target(
    copy_sources
    ALL
    COMMAND
      ${CMAKE_COMMAND} -DRTK_SOURCE_DIR:PATH="${RTK_SOURCE_DIR}"
      -DRTK_DOC_OUTPUT_DIR:PATH="${RTK_DOC_OUTPUT_DIR}" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/copy_and_fetch_sphinx_doc_files.cmake"
    COMMENT "Copying documentation sources"
  )

  find_program(
    SPHINX_EXECUTABLE
    NAMES
      sphinx-build
      sphinx-build.exe
    DOC "Sphinx documentation generator"
  )

  add_custom_target(
    sphinx_doc
    ALL
    COMMAND
      ${SPHINX_EXECUTABLE} -c ${RTK_DOC_OUTPUT_DIR} # conf.py directory
      ${RTK_DOC_OUTPUT_DIR} # Source directory containing root index.md
      "${RTK_DOC_OUTPUT_DIR}/_build" # Output directory
    COMMENT "Generating sphinx documentation"
    DEPENDS
      copy_sources
    WORKING_DIRECTORY ${RTK_DOC_OUTPUT_DIR}
  )

  set_property(
    DIRECTORY
    APPEND
    PROPERTY
      ADDITIONAL_MAKE_CLEAN_FILES
        "${RTK_DOC_OUTPUT_DIR}/_build"
  )
endif()
