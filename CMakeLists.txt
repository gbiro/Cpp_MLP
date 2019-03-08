cmake_minimum_required(VERSION 3.0.0)
project(NumberRecognizer VERSION 1.0.0)

# Link external CRT dependeincies
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
    
# Turn on all architecture-specific tuning
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# If compiling with GCC
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")

    # Enforce strict ISO C++ conformance
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=auto")

    set(ERROR_LIMIT 5 CACHE STRING
    "Limit number of errors output by diagnostics.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=${ERROR_LIMIT}")

    # Turn on ALL warnings for debug builds
    set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Weffc++ -ggdb")

endif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")


set(LOCAL_LIB ${LOCAL_LIB} ${PROJECT_SOURCE_DIR}/lib)

####################################### required packages

find_package(Threads REQUIRED)

add_subdirectory (Framework)
add_subdirectory (app)