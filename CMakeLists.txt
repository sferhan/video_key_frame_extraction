cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(key_frame_extraction LANGUAGES CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(FFMPEG_LIBRARIES /global/homes/f/farhan/FFmpeg/install/lib/libavcodec.dylib /global/homes/f/farhan/FFmpeg/install/lib/libavformat.dylib /global/homes/f/farhan/FFmpeg/install/lib/libavutil.dylib)
set(FFMPEG_INCLUDE_DIRS /global/homes/f/farhan/FFmpeg/install/include)

option(DISTRIBUTED_IMPL "Distributed Memory Implementation" OFF)

# 19 oct 2023: we are assuming the use of the NVHPC compiler on Perlmutter
# the compilation flags baked in below reflect that assumption
# set up your environment on Perlmutter with:
# module load PrgEnv-nvidia

add_executable(key_frame_extraction key_frame_extraction.cpp)
target_compile_definitions(key_frame_extraction PRIVATE DISTRIBUTED_IMPL=0)
target_include_directories(key_frame_extraction PRIVATE ${FFMPEG_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(key_frame_extraction PRIVATE ${FFMPEG_LIBRARIES} ${MPI_LIBRARIES})
target_compile_options(key_frame_extraction PRIVATE -fopenmp)
target_link_options(key_frame_extraction PRIVATE -fopenmp)

add_executable(key_frame_extraction_dist key_frame_extraction.cpp)
target_compile_definitions(key_frame_extraction_dist PRIVATE DISTRIBUTED_IMPL=1)
target_include_directories(key_frame_extraction_dist PRIVATE ${FFMPEG_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(key_frame_extraction_dist PRIVATE ${FFMPEG_LIBRARIES} ${MPI_LIBRARIES})
target_compile_options(key_frame_extraction_dist PRIVATE -fopenmp)
target_link_options(key_frame_extraction_dist PRIVATE -fopenmp)


