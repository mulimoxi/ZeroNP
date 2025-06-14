project(ZERONP C)
cmake_minimum_required(VERSION 3.5.1)
set(LINK_MKL 0)
if (LINK_MKL)
    message("Linking MKL")
    set(BLA_VENDOR Intel10_64lp)
else()
    message("Linking LAPACK")
endif()

if (CMAKE_HOST_WIN32)
    set(WINDOWS 1)
    set(CMAKE_OSX_ARCHITECTURES x86_64)
    message("Compile ZeroNP on Windows")
    # set(BLA_STATIC ON)
    find_package(LAPACK)
elseif(CMAKE_HOST_APPLE)
    set(MACOS 1)
    set(CMAKE_OSX_ARCHITECTURES arm64)
    message("Compile ZeroNP on MacOS")
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    set(LAPACK_FOUND TRUE)
elseif(CMAKE_HOST_UNIX)
    set(LINUX 1)
    set(CMAKE_OSX_ARCHITECTURES x86_64)
    message("Compile ZeroNP on Linux")
    # set(BLA_STATIC ON)
    find_package(LAPACK)
endif()
#set(ZERONP_COMPILER GCC)

set(CMAKE_C_STANDARD 11)

SET(CMAKE_BUILD_TYPE "Release")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

find_package(Python3)

if (NOT LAPACK_FOUND)
    message(FATAL_ERROR "LAPACK not found, please specify the path")
    # If LAPACK can not be found automatically, 
    # please comment the line above and
    # set LAPACK or MKL root manually below
    set(ENV{MKL_ROOT} YOUR_MKL_PATH)
    set(ENV{LAPACK_ROOT} YOUR_LAPACK_PATH) 
endif()

# # Configure MKL paths
set(ENV{MKL_LIB_PATH} $ENV{MKL_ROOT}/lib/intel64)
set(ENV{MKL_include_PATH} $ENV{MKL_ROOT}/include)

# Configure osqp paths
include(FindOSQPConf.cmake)
if (OSQP_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "OSQP not found")
endif()

# Include project directory
include_directories(./include)
include_directories($ENV{OSQP_include_PATH})
include_directories($ENV{MKL_include_PATH})

# file(GLOB_RECURSE mysrc ./source/*.c)
file(GLOB_RECURSE mysrc 
                  ./source/der_info.c
                  ./source/linalg.c
                  ./source/linsys.c
                  ./source/qp_solver.c
                  ./source/Rescue.c
                  ./source/zeronp_py.c
                  ./source/zeronp.c
                  ./source/subnp_qp.c
                  ./source/zeronp_util.c
                  ./source/zeronp_c_interface.c
                  )

file(GLOB_RECURSE myinclude ./include/*.h)

add_library(zeronp SHARED ${mysrc} ${myinclude})


target_link_libraries(zeronp PRIVATE $ENV{OSQP_LIB_PATH}/libosqp.a)
if(MACOS)
    target_link_libraries(zeronp PRIVATE "${ACCELERATE_FRAMEWORK}")
else()
    if (LAPACK_FOUND)
        target_link_libraries(zeronp PRIVATE $ENV{LAPACK_LIBRARIES})
    else()
        if(WINDOWS)
            if(LINK_MKL)
                target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_core.lib)
                target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_intel_lp64.lib)
                target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_sequential.lib)
            else()
                target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/liblapack.a)
                target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/libblas.a)
                target_link_libraries(zeronp gfortran)
                target_link_libraries(zeronp quadmath)
                # Add definition for name manging
                add_definitions("-DADD_")
            endif()
        else()
            if(LINK_MKL)
                target_link_libraries(zeronp -Wl,--start-group
                                            $ENV{MKL_LIB_PATH}/libmkl_core.a
                                            $ENV{MKL_LIB_PATH}/libmkl_intel_lp64.a
                                            $ENV{MKL_LIB_PATH}/libmkl_sequential.a
                                            -Wl,--end-group)
            else()
                target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/liblapack.a)
            endif()
        endif()
    endif()
endif()

# if(WINDOWS)
#     if(LINK_MKL)
#         target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_core.lib)
#         target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_intel_lp64.lib)
#         target_link_libraries(zeronp $ENV{MKL_LIB_PATH}/mkl_sequential.lib)
#     else()
#         target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/liblapack.a)
#         target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/libblas.a)
#         target_link_libraries(zeronp gfortran)
#         target_link_libraries(zeronp quadmath)
#         # Add definition for name manging
#         add_definitions("-DADD_")
#     endif()
#     target_link_libraries(zeronp $ENV{OSQP_LIB_PATH}/libosqp.a)

# elseif(MACOS)
#     target_link_libraries(zeronp PRIVATE $ENV{OSQP_LIB_PATH}/libosqp.a)
#     if(LINK_MKL)
#         message( FATAL_ERROR "MKL not supported on MACOS" )
#     else()
#         target_link_libraries(zeronp PRIVATE "${ACCELERATE_FRAMEWORK}")
#     endif()
# else()
#     if(LINK_MKL)
#         target_link_libraries(zeronp -Wl,--start-group
#                                     $ENV{MKL_LIB_PATH}/libmkl_core.a
#                                     $ENV{MKL_LIB_PATH}/libmkl_intel_lp64.a
#                                     $ENV{MKL_LIB_PATH}/libmkl_sequential.a
#                                     -Wl,--end-group)
#     else()
#         # target_link_libraries(zeronp $ENV{LAPACK_ROOT}/lib/liblapack.a)
#         target_link_libraries(zeronp $ENV{LAPACK_LIBRARIES})
#     endif()
#     target_link_libraries(zeronp $ENV{OSQP_LIB_PATH}/libosqp.a)
# endif()

