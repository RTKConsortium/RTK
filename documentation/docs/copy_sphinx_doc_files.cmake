# List of files to be copied to RTK_DOC_OUTPUT_DIR
list(APPEND DOCSRCFILES
      "conf.py"
      "index.md"
      "GettingStarted.md"
      "INSTALLATION.md"
      "CodeContribution.md"
    )

# List of subdirectories to be copied to RTK_DOC_OUTPUT_DIR
set(DOCSUBDIRS
      "examples"
      "applications"
      "documentation"
   )

# Append the files in the subdirectories
foreach(subdir ${DOCSUBDIRS})
  file(GLOB_RECURSE SUBSUBDIR LIST_DIRECTORIES false RELATIVE "${RTK_SOURCE_DIR}" "${RTK_SOURCE_DIR}/${subdir}/*")
  list(APPEND DOCSRCFILES ${SUBSUBDIR})
endforeach()

# Make the directory and copy the file if they are different
foreach(file ${DOCSRCFILES})
  get_filename_component(dir "${RTK_DOC_OUTPUT_DIR}/${file}" DIRECTORY)
  file(MAKE_DIRECTORY ${dir})
  file(COPY_FILE "${RTK_SOURCE_DIR}/${file}" "${RTK_DOC_OUTPUT_DIR}/${file}" ONLY_IF_DIFFERENT)
endforeach()

# Clean-up: remove all files which are not in the input files or in the _build and documentation/docs/ExternalData/ subfolders
file(GLOB_RECURSE filesInOutput LIST_DIRECTORIES false RELATIVE "${RTK_DOC_OUTPUT_DIR}" "${RTK_DOC_OUTPUT_DIR}/*")
list(REMOVE_ITEM filesInOutput ${DOCSRCFILES})
file(GLOB_RECURSE filesInOutputBuild LIST_DIRECTORIES false RELATIVE "${RTK_DOC_OUTPUT_DIR}" "${RTK_DOC_OUTPUT_DIR}/_build/*")
list(REMOVE_ITEM filesInOutput ${filesInOutputBuild})
foreach(file ${filesInOutput})
  file(REMOVE "${RTK_DOC_OUTPUT_DIR}/${file}")
endforeach()
