
all: target python_copy

# Name of the shared object
TARGET = gbdtmo.so

# Compiler settings
CPP = clang++
CPP_FLAGS = -std=c++11
CPP_FLAGS += -fPIC -Ofast
CPP_FLAGS += -Wall -Wfatal-errors -pedantic
CPP_FLAGS += -m64
# CPP_FLAGS += -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize

BUILD_DIR = build
SOURCES_DIR = src

SOURCES = api.cpp booster_single.cpp booster_multi.cpp booster_base.cpp histogram.cpp io.cpp loss.cpp tree.cpp string_utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)
# SOURCES := $(addprefix ${SOURCES_DIR}/,${SOURCES})
# OBJECTS := $(addprefix ${BUILD_DIR}/,${OBJECTS})

# Name of library in the python package
# Python package directory
PYTHON_TARGET = lib.so
PYTHON_DIR = gbdtmo

# TODO: Add this:
# https://www.gnu.org/software/make/manual/html_node/Automatic-Prerequisites.html

# Compile target
target: ${BUILD_DIR}/${TARGET}
python_copy: ${PYTHON_DIR}/${PYTHON_TARGET}

# Build objects for each compilation unit
${BUILD_DIR}/%.o: ${SOURCES_DIR}/%.cpp
	@ [ -d "${BUILD_DIR}" ] || mkdir ${BUILD_DIR};
	@ echo " CXX  $(notdir $@) <- $(notdir $<)"
	@ ${CPP} ${CPP_FLAGS} -c $< -o $@

# Link the shared objects
${BUILD_DIR}/${TARGET}: $(addprefix ${BUILD_DIR}/,${OBJECTS})
	@ echo " CXX  $(notdir $@) <- $(notdir $^)"
	@ ${CPP} ${CPP_FLAGS} -shared $^ -o $@

${PYTHON_DIR}/${PYTHON_TARGET}: ${BUILD_DIR}/${TARGET}
	@ echo " CP   $(notdir $@) <- $(notdir $^)"
	@ cp ${BUILD_DIR}/${TARGET} ${PYTHON_DIR}/${PYTHON_TARGET}

.PHONY: clean

clean:
	@ rm -rf ${BUILD_DIR}
	@ rm -rf ${PYTHON_DIR}/${PYTHON_TARGET}