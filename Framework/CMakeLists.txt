cmake_minimum_required (VERSION 3.0.0)

project(Framework LANGUAGES CXX)

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*)
file(GLOB headers ${PROJECT_SOURCE_DIR}/inc/*)

add_library (${PROJECT_NAME} ${headers} ${sources})

target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC Threads::Threads
)


set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${LOCAL_LIB}
                                                 CXX_STANDARD 14
                                                 CXX_STANDARD_REQUIRED ON)