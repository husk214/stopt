cmake_minimum_required(VERSION 2.8.12)
project(stopt CXX)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRCS)
message(${SRCS})
find_library(MY_LIB NAMES stopt PATHS "${CMAKE_SOURCE_DIR}/src")
message(${MY_LIB})

foreach(SRC ${SRCS})
  get_filename_component(MY_EXE "${SRC}" NAME_WE)
  add_executable(${MY_EXE} ${SRC})
  # target_link_libraries(${MY_EXE} ${CommonLibs} stopt)
endforeach(SRC)
