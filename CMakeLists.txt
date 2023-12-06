cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(key_frame_extraction LANGUAGES CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(FFMPEG_LIBRARIES /usr/local/lib/libavcodec.dylib /usr/local/lib/libavformat.dylib /usr/local/lib/libavutil.dylib)
set(FFMPEG_INCLUDE_DIRS /usr/local/include)

# 19 oct 2023: we are assuming the use of the NVHPC compiler on Perlmutter
# the compilation flags baked in below reflect that assumption
# set up your environment on Perlmutter with:
# module load PrgEnv-nvidia

add_executable(key_frame_extraction key_frame_extraction.cpp)

target_include_directories(key_frame_extraction PRIVATE ${FFMPEG_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(key_frame_extraction PRIVATE ${FFMPEG_LIBRARIES} ${MPI_LIBRARIES})

target_compile_options(key_frame_extraction PRIVATE -fopenmp -fast)
target_link_options(key_frame_extraction PRIVATE -fopenmp)

# add_executable(sobel_cpu_omp_offload sobel_cpu_omp_offload.cpp)
# target_compile_options(sobel_cpu_omp_offload PRIVATE -fast -mp=gpu -Minfo=mp,accel )
# target_link_options(sobel_cpu_omp_offload PRIVATE -fast -mp=gpu -Minfo=mp,accel)


project(scratch LANGUAGES CXX)

# 19 oct 2023: we are assuming the use of the NVHPC compiler on Perlmutter
# the compilation flags baked in below reflect that assumption
# set up your environment on Perlmutter with:
# module load PrgEnv-nvidia

add_executable(scratch test.cpp)

target_include_directories(scratch PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_libraries(scratch PRIVATE ${FFMPEG_LIBRARIES})

target_compile_options(scratch PRIVATE -fopenmp -fast)
target_link_options(scratch PRIVATE -fopenmp)