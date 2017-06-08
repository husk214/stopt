cmake_minimum_required(VERSION 2.8.12)
project(stopt CXX)

message("${CMAKE_CXX_FLAGS}")

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
# add_library(stopt STATIC ${SRCS})
target_link_libraries(stopt ${CommonLibs})
set_target_properties(stopt PROPERTIES LINKER_LANGUAGE CXX)
