
all: target

# Name of the shared object
TARGET = gbdtmo.so

# Compiler settings
CPP = clang++
CPP_FLAGS = -std=c++11
CPP_FLAGS += -fPIC -Ofast
CPP_FLAGS += -Wall -Wfatal-errors
CPP_FLAGS += -m64

BUILD_DIR = build
SOURCES_DIR = src

SOURCES = api.cpp booster.cpp dataStruct.cpp io.cpp loss.cpp tree.cpp
OBJECTS = $(SOURCES:.cpp=.o)
# SOURCES := $(addprefix ${SOURCES_DIR}/,${SOURCES})
# OBJECTS := $(addprefix ${BUILD_DIR}/,${OBJECTS})

target: ${BUILD_DIR}/${TARGET}

# Build objects for each compilation unit
${BUILD_DIR}/%.o: ${SOURCES_DIR}/%.cpp
	@ [ -d "${BUILD_DIR}" ] || mkdir ${BUILD_DIR};
	${CPP} ${CPP_FLAGS} -c $< -o $@

# Link the shared objects
${BUILD_DIR}/${TARGET}: $(addprefix ${BUILD_DIR}/,${OBJECTS})
	@ [ -d "${BUILD_DIR}" ] || mkdir ${BUILD_DIR};
	${CPP} ${CPP_FLAGS} -shared $^ -o $@

.PHONY: clean

clean:
	rm -rf "${BUILD_DIR}"