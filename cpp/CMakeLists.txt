cmake_minimum_required (VERSION 3.12)

cmake_policy(SET CMP0075 NEW)

include(CheckIncludeFileCXX)

project (radMC_planar)

enable_testing()

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build options
option(BUILD_TESTS "Building tests" ON)
# Link tests dynamically
add_definitions( -DBOOST_TEST_DYN_LINK )
# absolute path to testing data
add_definitions( -DDATA_PATH="${PROJECT_SOURCE_DIR}/tests/data/" )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# configure compiler warnings
set( COMPILER_WARNINGS "-Wall -Wpedantic -Wextra -Wcast-align -Wcast-qual -Winit-self -Wlogical-op")
set( COMPILER_WARNINGS "${COMPILER_WARNINGS} -Wsign-promo -Wstrict-null-sentinel -Wundef")
set( COMPILER_WARNINGS "${COMPILER_WARNINGS} -Wno-unused-parameter -Wno-unused-local-typedefs")
set( COMPILER_WARNINGS "${COMPILER_WARNINGS} -Wno-unused-local-typedefs -Wno-deprecated-declarations")
set( COMPILER_WARNINGS "${COMPILER_WARNINGS} -Wno-ignored-attributes")

if (MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /EHsc ${OpenMP_CXX_FLAGS}  /fp:precise" )
  set( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE )
elseif (CYGWIN)
  set( CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native --std=gnu++17 ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_CXX_FLAGS_RELEASE_DEF "-O3 -DNDEBUG")
elseif(MINGW)
  set( CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native --std=gnu++17 ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_CXX_FLAGS_RELEASE_DEF "-O3 -DNDEBUG")
else(MSVC)
  set( CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -march=native --std=c++17 ${OpenMP_CXX_FLAGS}" )
  set( CMAKE_CXX_FLAGS_RELEASE_DEF "-O3 -DNDEBUG")
endif (MSVC)

# add compiler warnings for all but msvc
if(NOT MSVC)
   set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILER_WARNINGS}")
endif(NOT MSVC)

if("${CMAKE_CXX_FLAGS_RELEASE}" STREQUAL "_UNSET")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_DEF}" CACHE STRING "Flags used by the compiler during release builds." FORCE)
endif()

if("${CMAKE_CXX_FLAGS_DEBUG}" STREQUAL "_UNSET")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
endif()

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#custom cmake modules
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake/")

#opencl source dir
add_definitions( -DCL_SOURCE_DIR="${CMAKE_SOURCE_DIR}/../opencl/" )

#find packages
find_package( Boost 1.72.0 COMPONENTS filesystem unit_test_framework REQUIRED )
set( BOOST_TEST_LIBS ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )

find_package (NetCDF REQUIRED)

include_directories(SYSTEM ${PROJECT_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${Boost_INCLUDE_DIRS} )

find_package( OpenCL 1.2 REQUIRED)

if(OpenCL_FOUND)
    include_directories(SYSTEM ${OpenCL_INCLUDE_DIRS} )
    list( APPEND CMAKE_REQUIRED_LIBRARIES ${OpenCL_LIBRARIES} )
    # OpenCL DEFINEs
    add_compile_definitions(CL_HPP_MINIMUM_OPENCL_VERSION=120)
    add_compile_definitions(CL_HPP_TARGET_OPENCL_VERSION=120)
    add_compile_definitions(CL_TARGET_OPENCL_VERSION=120)
    add_compile_definitions(CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY)
    check_include_file_cxx( "CL/opencl.hpp" HAVE_OPENCL_HPP )
    check_include_file_cxx( "CL/cl2.hpp" HAVE_CL2_HPP )
    check_include_file_cxx( "CL/cl.hpp" HAVE_CL_HPP )
    IF(NOT HAVE_OPENCL_HPP)
        if(NOT HAVE_CL2_HPP)
            if(NOT HAVE_CL_HPP)
                message(FATAL_ERROR "opencl.hpp or cl2.hpp or cl.hpp needed but not found.")
            else(NOT HAVE_CL_HPP)
                set(OPENCL_CPP_HEADER_NAME "CL/cl.hpp")
                set(OPENCL_CPP_HEADER_TYPE 1)
                message(STATUS "CL/cl.hpp header used.")
            endif(NOT HAVE_CL_HPP)
        else(NOT HAVE_CL2_HPP)
            set(OPENCL_CPP_HEADER_NAME "CL/cl2.hpp")
            set(OPENCL_CPP_HEADER_TYPE 2)
            message(STATUS "CL/cl2.hpp header used.")
        endif(NOT HAVE_CL2_HPP)
    else(NOT HAVE_OPENCL_HPP)
        set(OPENCL_CPP_HEADER_NAME "CL/opencl.hpp")
        set(OPENCL_CPP_HEADER_TYPE 3)
        message(STATUS "CL/opencl.hpp header used.")
    endif(NOT HAVE_OPENCL_HPP)
    add_compile_definitions(OPENCL_CPP_HEADER_TYPE=${OPENCL_CPP_HEADER_TYPE})
endif(OpenCL_FOUND)

link_directories( ${Boost_LIBRARY_DIRS} )

add_library( radMC config.cpp clsim.cpp netcdf_interface_mc.cpp)
target_link_libraries( radMC ${NETCDF_LIBRARIES_C} ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})

# build executables
add_executable( radMC_planar radMC_planar.cpp )
target_link_libraries( radMC_planar radMC )

if(BUILD_TESTS)

add_executable( test_welford tests/test_welford.cpp )
target_link_libraries( test_welford ${BOOST_TEST_LIBS} )
add_test( Welford_algorithm test_welford )

add_executable( test_mc_semi_lf tests/test_semi_lf.cpp )
target_link_libraries( test_mc_semi_lf radMC )
add_test( MC_semi_lf test_mc_semi_lf )

add_executable( test_mc_semi_int tests/test_semi_int.cpp )
target_link_libraries( test_mc_semi_int radMC )
add_test( MC_semi_int test_mc_semi_int )

add_executable( test_mc_slab_lf tests/test_slab_lf.cpp )
target_link_libraries( test_mc_slab_lf radMC )
add_test( MC_slab_lf test_mc_slab_lf )

add_executable( test_mc_slab_int tests/test_slab_int.cpp )
target_link_libraries( test_mc_slab_int radMC )
add_test( MC_slab_int test_mc_slab_int )

add_executable( test_convergence tests/test_convergence.cpp )
target_link_libraries( test_convergence radMC )

endif(BUILD_TESTS)
