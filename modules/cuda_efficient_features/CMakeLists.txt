cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME cuda_efficient_features)
set(INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INTERNAL_INCLUDE_DIR ${INTERFACE_INCLUDE_DIR})

option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(USE_HIP)
	project(${PROJECT_NAME} LANGUAGES CXX HIP)
	if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
		set(CMAKE_HIP_ARCHITECTURES "gfx90a")
	endif()
else()
	project(${PROJECT_NAME} LANGUAGES CXX CUDA)
endif()

# dependent packages
if(USE_HIP)
	# For HIP build, we only need OpenCV core modules (no CUDA modules)
	find_package(OpenCV REQUIRED core features2d)
	find_package(hip REQUIRED)
	find_package(hipblas REQUIRED)
else()
	find_package(OpenCV REQUIRED core features2d cudev cudafilters cudaimgproc cudawarping)
	find_package(CUDAToolkit REQUIRED)
endif()

# target configuration
if(USE_HIP)
	file(GLOB CUDA_SOURCES src/*.cu)
	file(GLOB CPP_SOURCES src/*.cpp)
	file(GLOB HEADERS src/*.h* ${INTERNAL_INCLUDE_DIR}/*.h*)
	set(SRCS ${CPP_SOURCES} ${CUDA_SOURCES} ${HEADERS})
else()
	file(GLOB SRCS src/*.cpp src/*.cu src/*.h* ${INTERNAL_INCLUDE_DIR}/*.h*)
endif()

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SRCS})

if(USE_HIP)
	# Add hip_compat directory BEFORE OpenCV includes to shim cuda_runtime.h
	# Also add it to INTERFACE for consumers (tests, samples)
	target_include_directories(${PROJECT_NAME} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/src/hip_compat
		${INTERNAL_INCLUDE_DIR}
		${OpenCV_INCLUDE_DIRS})
	target_include_directories(${PROJECT_NAME} INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/src/hip_compat)
	target_compile_definitions(${PROJECT_NAME} PRIVATE USE_HIP)
	set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
	set_target_properties(${PROJECT_NAME} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
	set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
	if(NOT WIN32)
		target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fPIE>)
	endif()
	target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} hip::host roc::hipblas)
else()
	target_include_directories(${PROJECT_NAME} PRIVATE ${INTERNAL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
	target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} CUDA::cudart CUDA::cublas)
	set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})
endif()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${INTERFACE_INCLUDE_DIR})

target_compile_options(${PROJECT_NAME} PRIVATE
$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wall -O3>
$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/wd4819>
$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-lineinfo -Xcompiler="\"/wd4819 /wd4244\"">
)

install(
	TARGETS ${PROJECT_NAME}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)

install(
	DIRECTORY ${INTERFACE_INCLUDE_DIR}
	DESTINATION ${CMAKE_INSTALL_PREFIX}
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
