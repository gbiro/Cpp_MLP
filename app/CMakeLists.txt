cmake_minimum_required(VERSION 3.0.0)
project(app VERSION 0.0.0)

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*)
file(GLOB headers ${PROJECT_SOURCE_DIR}/inc/*)

macro(build_executables EXECUTABLES)
foreach(EXECUTABLE ${EXECUTABLES})
    get_filename_component(TARGET_NAME ${EXECUTABLE} NAME_WE)
    add_executable(${TARGET_NAME} ${EXECUTABLE})
    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    target_link_libraries(${TARGET_NAME} 
        PUBLIC Framework
        PUBLIC ${CMAKE_DL_LIBS}
        )
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                                                        CXX_STANDARD 14
                                                        CXX_STANDARD_REQUIRED ON)
    set(EXECUTABLES ${EXECUTABLES} ${TARGET_NAME})
endforeach()
endmacro()

build_executables("${sources}")
