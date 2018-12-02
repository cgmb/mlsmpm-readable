cmake_minimum_required(VERSION 2.8.11)
project(mlsmpm)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(LIBRARIES ${LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

if(UNIX)
    find_package(X11 REQUIRED)
    set(LIBRARIES ${LIBRARIES}
        ${X11_LIBRARIES}
    )
    set(INCLUDES ${INCLUDES}
        ${X11_INCLUDE_DIR}
    )
endif(UNIX)

set(SOURCES
    mls-mpm88.cpp
)

add_executable(mls-mpm88 ${SOURCES})
target_link_libraries(mls-mpm88 ${LIBRARIES})
target_include_directories(mls-mpm88 PRIVATE ${INCLUDES})
target_compile_definitions(mls-mpm88 PRIVATE GLM_FORCE_CXX14=1 _USE_MATH_DEFINES=1)
