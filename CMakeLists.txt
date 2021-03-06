cmake_minimum_required(VERSION 2.8)
 
if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

PROJECT(VolumeRender)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_FLAGS "-w -fopenmp")

#########################################################
# Include GLM
#########################################################
#include_directories("/home/soumya/")
#set(GLM_DIR "/home/user/glm" CACHE PATH "Directory to GLM Library")
#add_definitions(-DGLM_PATH=\"${GLM_DIR}\")

#include_directories(${GLM_DIR})

#########################################################
# Include OSUFlow
#########################################################
#set(OSUFLOW_DIR "/usr/local/" CACHE PATH "OSUFlow dir")
#set(OSUFLOW_INCLUDE_DIR "${OSUFLOW_DIR}/include/osuflow")
#set(OSUFLOW_LIB_DIR "${OSUFLOW_DIR}/lib/osuflow")
#include_directories(${OSUFLOW_INCLUDE_DIR} ${OSUFLOW_INCLUDE_DIR}/VTK)
#link_directories(${OSUFLOW_LIB_DIR})

#########################################################
# Include OPENGL
#########################################################
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})

#########################################################
# Include GLEW
#########################################################
find_package(GLEW)
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})
#add_definitions(${GLEW_DEFINITIONS})

#if(NOT GLEW_FOUND)
#    message(ERROR " GLEW not found!")
#endif(NOT GLEW_FOUND)


########################################################
# CUDA : should be put before VTK
########################################################
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30 -Xptxas -v)
find_package(CUDA QUIET REQUIRED)
include_directories( ${CUDA_TOOLKIT_ROOT_DIR}/include  ${CUDA_SDK_ROOT_DIR}/common/inc )
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 )# ${CUDA_SDK_ROOT_DIR}/common/lib/x64 )
#LINUX:
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
CUDA_ADD_LIBRARY(volumeRender_kernel ${CMAKE_SOURCE_DIR}/volRender_kernel.cu)


#########################################################
# Include VTK
#########################################################
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#########################################################
# Include QT4
#########################################################
find_package(Qt4 COMPONENTS QtCore QtOpenGL QtGui  REQUIRED )
include(${QT_USE_FILE})


########################################################
# Include Boost
########################################################
#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIR})

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
########################################################



file(GLOB UI_FILES *.ui)
file(GLOB QT_WRAP *.h)
file(GLOB CXX_FILES *.cpp *.cxx)

QT4_WRAP_UI(UISrcs ${UI_FILES})
QT4_WRAP_CPP(MOCSrcs ${QT_WRAP})

add_executable(VolumeRender ${CXX_FILES} ${UISrcs} ${MOCSrcs} )


#########################################################
# Data path
#########################################################
set(DATA_DIR "/data" CACHE PATH "Directory to data")
add_definitions(-DDATA_DIR=\"${DATA_DIR}/\")
#set(OUTPUT_DIR "/data" CACHE PATH "Directory to the output")
#add_definitions(-DOUTPUT_DIR=\"${OUTPUT_DIR}/\")
#set(SAVED_OUTPUT_DIR "/data/turbine_Stg/" CACHE PATH "Directory to the saved output")
#add_definitions(-DSAVED_OUTPUT_DIR=\"${SAVED_OUTPUT_DIR}/\")
set(MIN_TIMESTEP "1" CACHE STRING "Initial time step")
add_definitions(-DMIN_TIMESTEP=${MIN_TIMESTEP})
set(MAX_TIMESTEP "10" CACHE STRING "Last time step")
add_definitions(-DMAX_TIMESTEP=${MAX_TIMESTEP})

#########################################################
# Link libraries
#########################################################
#target_link_libraries(VTK_GL ${QT_LIBRARIES} OSUFlowVTK_d OSUFlow_d ${VTK_LIBRARIES} ${OPENGL_LIBRARIES} ${LIBPCA_LIBRARIES})

target_link_libraries(VolumeRender ${QT_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${VTK_LIBRARIES} ${GLEW_LIBRARIES} volumeRender_kernel)


