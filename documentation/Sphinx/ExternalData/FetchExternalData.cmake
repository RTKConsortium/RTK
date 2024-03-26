# Fetch external data from https://data.kitware.com
# Used by Sphinx conf.py during setup.
#
set(link_content sha512)

if(NOT EXTERNAL_DATA_INPUT_ROOT)
  set(EXTERNAL_DATA_INPUT_ROOT "${CMAKE_SOURCE_DIR}")
endif()

if(NOT EXTERNAL_DATA_OUTPUT_ROOT)
  set(EXTERNAL_DATA_OUTPUT_ROOT "${CMAKE_SOURCE_DIR}")
endif()

file(GLOB_RECURSE content_files
  "${EXTERNAL_DATA_INPUT_ROOT}/*${link_content}")

foreach(content_file ${content_files})

  file(RELATIVE_PATH content_file_rel_path ${EXTERNAL_DATA_INPUT_ROOT} ${content_file})
  get_filename_component(last_ext "${content_file_rel_path}" LAST_EXT)
  string(REPLACE "${last_ext}" "" content_file_rel_path "${content_file_rel_path}")

  file(READ "${content_file}" hash)
  set(URL "https://data.kitware.com:443/api/v1/file/hashsum/${link_content}/${hash}/download")
  string(REGEX REPLACE "\n" "" URL "${URL}")
  file(DOWNLOAD "${URL}" "${EXTERNAL_DATA_OUTPUT_ROOT}/${content_file_rel_path}")

endforeach()
