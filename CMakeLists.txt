cmake_minimum_required(VERSION 2.8.11)
project(mlsmpm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
set(LIBRARIES ${LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)
find_package(OpenMP)
if(${OpenMP_FOUND})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(UNIX)
    if(APPLE)
        set(CMAKE_FIND_FRAMEWORK ONLY)
        find_library(COCOA_LIBRARY Cocoa)
        set(LIBRARIES ${LIBRARIES} ${COCOA_LIBRARY})
    else()
        find_package(X11 REQUIRED)
        set(LIBRARIES ${LIBRARIES} ${X11_LIBRARIES})
        set(INCLUDES ${INCLUDES} ${X11_INCLUDE_DIR})
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(SOURCES
    mls-mpm.cpp
)

add_executable(mls-mpm ${SOURCES})
target_link_libraries(mls-mpm ${LIBRARIES})
target_include_directories(mls-mpm PRIVATE ${INCLUDES})
target_compile_definitions(mls-mpm PRIVATE GLM_FORCE_CXX14=1 _USE_MATH_DEFINES=1)
