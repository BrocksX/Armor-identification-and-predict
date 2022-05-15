# project name
PROJECT(HELLO)

# using C++11
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 ")

# cmake version
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

# find OpenCV
FIND_PACKAGE(OpenCV REQUIRED)

# show the message of OpenCV
message(STATUS "OpenCV library status:")
message(STATUS "    version: 	${OpenCV_VERSION}")
message(STATUS "    headers: 	${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: 	${OpenCV_LIBS}")

# link headers
INCLUDE_DIRECTORIES({OpenCV_INCLUDE_DIRS})

# 添加源代码文件到SRC_LIST变量中
AUX_SOURCE_DIRECTORY(. SRC_LIST)

# 生成可执行文件
ADD_EXECUTABLE(hello ${SRC_LIST})

# after ADD_EXECUTABLE，为生成文件target添加库
TARGET_LINK_LIBRARIES(hello ${OpenCV_LIBS})
