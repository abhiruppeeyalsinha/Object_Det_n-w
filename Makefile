# ============================================================================
# Name        : Makefile
# Date        : 16 Aug 2026
# Description : Robust / Dynamic C++ Build System
#
# Features:
#   - Automatic architecture detection
#   - Automatic compiler/tool detection
#   - Automatic OpenCV detection through pkg-config
#   - Automatic SDL2 detection
#   - Automatic CUDA detection
#   - Optional TensorRT detection
#   - Optional cuDNN detection
#   - Optional OpenCV CUDA module detection
#   - Automatic dependency generation
#   - Parallel-build safe
#   - Debug / Release configurations
#   - Verbose build support
#   - Clean / rebuild / info targets
#   - No hard dependency on unavailable libraries
# ============================================================================

# ============================================================================
# USER CONFIGURATION
# ============================================================================

SRC_FILE ?= hybrid_classical_ai_det.cpp

INC_DIR  ?= ./Inc
OBJ_DIR  ?= ./Obj/$(BUILD)
BIN_DIR  ?= .

TARGET := $(BIN_DIR)/Exe-$(basename $(notdir $(SRC_FILE)))

# Build type:
#   make BUILD=release
#   make BUILD=debug
BUILD ?= release

# Set to 1 for detailed commands:
#   make V=1
V ?= 0

# Enable optional dependencies automatically when detected.
ENABLE_TENSORRT ?= auto
ENABLE_CUDNN    ?= auto
ENABLE_CUDA_OPENCV ?= auto

# ============================================================================
# TOOLS
# ============================================================================

CXX  ?= g++
NVCC ?= nvcc
PKG_CONFIG ?= pkg-config

# ============================================================================
# ARCHITECTURE
# ============================================================================

ARCH := $(shell uname -m)

ifeq ($(ARCH),x86_64)
    GPU_ARCH ?= sm_86
    JSON_INC :=
else ifeq ($(ARCH),aarch64)
    GPU_ARCH ?= sm_72
    JSON_INC := -I/home/nano/Json/include
else
    $(error Unsupported architecture: $(ARCH))
endif

# ============================================================================
# PROJECT FILES
# ============================================================================

SRC_BASE := $(basename $(notdir $(SRC_FILE)))

OBJ_FILE  := $(OBJ_DIR)/$(SRC_BASE).o
DEPS_FILE := $(OBJ_DIR)/$(SRC_BASE).d

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Check whether a command exists.
command_exists = $(shell command -v $(1) >/dev/null 2>&1 && echo yes || echo no)

# Check pkg-config package.
pkg_exists = $(shell $(PKG_CONFIG) --exists "$(1)" 2>/dev/null && echo yes || echo no)

# ============================================================================
# COMPILER CHECK
# ============================================================================

ifeq ($(call command_exists,$(CXX)),no)
    $(error C++ compiler '$(CXX)' was not found)
endif

# ============================================================================
# CUDA DETECTION
# ============================================================================

CUDA_CANDIDATES := \
    $(CUDA_HOME) \
    $(CUDA_PATH) \
    /usr/local/cuda \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.5 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12.3 \
    /usr/local/cuda-12.2 \
    /usr/local/cuda-12.1 \
    /usr/local/cuda-12.0 \
    /usr/local/cuda-11.8 \
    /usr/local/cuda-11.7 \
    /usr/local/cuda-11.6

CUDA_PATH := $(firstword $(foreach d,$(CUDA_CANDIDATES),\
    $(if $(wildcard $(d)/bin/nvcc),$(d))))

ifneq ($(strip $(CUDA_PATH)),)

    NVCC := $(CUDA_PATH)/bin/nvcc
    CUDA_INCLUDE := $(CUDA_PATH)/include
    CUDA_LIBDIR  := $(CUDA_PATH)/lib64

    ifneq ($(wildcard $(CUDA_LIBDIR)/libcudart.so*),)
        HAVE_CUDA_RUNTIME := yes
    else
        HAVE_CUDA_RUNTIME := no
    endif

else

    HAVE_CUDA_RUNTIME := no

endif

# ============================================================================
# OPENMP
# ============================================================================

OPENMP_FLAGS := -fopenmp

# ============================================================================
# SDL2 DETECTION
# ============================================================================

HAVE_SDL2 := $(call pkg_exists,sdl2)

ifeq ($(HAVE_SDL2),yes)

    SDL2_CFLAGS := $(shell $(PKG_CONFIG) --cflags sdl2)
    SDL2_LIBS   := $(shell $(PKG_CONFIG) --libs sdl2)

else

    $(warning SDL2 was not found through pkg-config)

    SDL2_CFLAGS :=
    SDL2_LIBS   := -lSDL2

endif

# ============================================================================
# OPENCV DETECTION
# ============================================================================

HAVE_OPENCV := $(call pkg_exists,opencv4)

ifeq ($(HAVE_OPENCV),yes)

    OPENCV_CFLAGS := $(shell $(PKG_CONFIG) --cflags opencv4)
    OPENCV_LIBS   := $(shell $(PKG_CONFIG) --libs opencv4)

else

    $(error OpenCV 4 was not found. Install opencv4 development files.)

endif

# ============================================================================
# OPTIONAL TENSORRT DETECTION
# ============================================================================

TENSORRT_CANDIDATES := \
    $(TENSORRT_HOME) \
    $(TENSORRT_PATH) \
    /usr/local/TensorRT \
    /usr/local/TensorRT-* \
    /usr/lib/x86_64-linux-gnu

TENSORRT_LIB := $(firstword $(foreach d,$(TENSORRT_CANDIDATES),\
    $(if $(wildcard $(d)/lib/libnvinfer.so*),$(d)/lib,\
    $(if $(wildcard $(d)/lib64/libnvinfer.so*),$(d)/lib64,\
    $(if $(wildcard $(d)/libnvinfer.so*),$(d)) ))))

ifneq ($(strip $(TENSORRT_LIB)),)

    HAVE_TENSORRT := yes
    TENSORRT_INCLUDE := $(dir $(firstword $(wildcard \
        $(dir $(TENSORRT_LIB))/../include/NvInfer.h \
        /usr/include/NvInfer.h \
        /usr/include/x86_64-linux-gnu/NvInfer.h)))

else

    HAVE_TENSORRT := no

endif

ifeq ($(ENABLE_TENSORRT),yes)
    ifeq ($(HAVE_TENSORRT),no)
        $(error TensorRT requested but not found)
    endif
endif

# ============================================================================
# OPTIONAL cuDNN DETECTION
# ============================================================================

CUDNN_CANDIDATES := \
    $(CUDNN_HOME) \
    $(CUDNN_PATH) \
    /usr/local/cudnn \
    /usr/local/cuda \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib

CUDNN_LIBDIR := $(firstword $(foreach d,$(CUDNN_CANDIDATES),\
    $(if $(wildcard $(d)/lib64/libcudnn.so*),$(d)/lib64,\
    $(if $(wildcard $(d)/lib/libcudnn.so*),$(d)/lib,\
    $(if $(wildcard $(d)/libcudnn.so*),$(d)) ))))

ifneq ($(strip $(CUDNN_LIBDIR)),)

    HAVE_CUDNN := yes

else

    HAVE_CUDNN := no

endif

ifeq ($(ENABLE_CUDNN),yes)
    ifeq ($(HAVE_CUDNN),no)
        $(error cuDNN requested but not found)
    endif
endif

# ============================================================================
# OPENCL / CUDA OPENCV MODULE DETECTION
# ============================================================================

ifeq ($(HAVE_OPENCV),yes)

    OPENCV_CUDA_LIBS := \
        opencv_cudaimgproc \
        opencv_cudaarithm \
        opencv_cudawarping \
        opencv_cudafilters \
        opencv_cudafeatures2d \
        opencv_cudacodec \
        opencv_cudabgsegm

    HAVE_OPENCV_CUDA := yes

    # Check all CUDA OpenCV libraries.
    $(foreach lib,$(OPENCV_CUDA_LIBS),\
        $(if $(shell ldconfig -p 2>/dev/null | grep -q "lib$(lib).so" && echo yes),,\
            $(eval HAVE_OPENCV_CUDA := no)))

else

    HAVE_OPENCV_CUDA := no

endif

ifeq ($(ENABLE_CUDA_OPENCV),yes)
    ifeq ($(HAVE_OPENCV_CUDA),no)
        $(error OpenCV CUDA modules requested but not found)
    endif
endif

# ============================================================================
# OPTIONAL LIBRARIES
# ============================================================================

OPTIONAL_LIBS :=
OPTIONAL_LDFLAGS :=
OPTIONAL_RPATHS :=
OPTIONAL_INCLUDES :=

ifeq ($(HAVE_TENSORRT),yes)

    ifneq ($(ENABLE_TENSORRT),no)

        OPTIONAL_LIBS += \
            -lnvinfer \
            -lnvinfer_plugin \
            -lnvonnxparser

        OPTIONAL_LDFLAGS += -L$(TENSORRT_LIB)
        OPTIONAL_RPATHS  += -Wl,-rpath,$(TENSORRT_LIB)

        ifneq ($(strip $(TENSORRT_INCLUDE)),)
            OPTIONAL_INCLUDES += -I$(TENSORRT_INCLUDE)
        endif

    endif

endif

ifeq ($(HAVE_CUDNN),yes)

    ifneq ($(ENABLE_CUDNN),no)

        OPTIONAL_LIBS += -lcudnn

        OPTIONAL_LDFLAGS += -L$(CUDNN_LIBDIR)
        OPTIONAL_RPATHS  += -Wl,-rpath,$(CUDNN_LIBDIR)

    endif

endif

ifeq ($(HAVE_OPENCV_CUDA),yes)

    ifneq ($(ENABLE_CUDA_OPENCV),no)

        OPTIONAL_LIBS += \
            -lopencv_cudaimgproc \
            -lopencv_cudaarithm \
            -lopencv_cudawarping \
            -lopencv_cudafilters \
            -lopencv_cudafeatures2d \
            -lopencv_cudacodec \
            -lopencv_cudabgsegm

    endif

endif

# ============================================================================
# INCLUDE FLAGS
# ============================================================================

INCLUDES := \
    -I$(INC_DIR) \
    $(JSON_INC) \
    $(OPENCV_CFLAGS) \
    $(SDL2_CFLAGS) \
    $(OPTIONAL_INCLUDES)

ifneq ($(HAVE_CUDA_RUNTIME),no)
    INCLUDES += -I$(CUDA_INCLUDE)
endif

# ============================================================================
# BUILD FLAGS
# ============================================================================

COMMON_FLAGS := \
    -Wall \
    -Wextra \
    -Wno-unused-result \
    -Wno-parentheses \
    -Wno-unused-label \
    -MMD \
    -MP \
    -std=c++17 \
    -fpermissive

ifeq ($(BUILD),debug)

    OPT_FLAGS := \
        -O0 \
        -g3

else ifeq ($(BUILD),release)

    OPT_FLAGS := \
        -O2 \
        -DNDEBUG

else

    $(error Unknown BUILD='$(BUILD)'. Use BUILD=debug or BUILD=release)

endif

CXX_FLAGS := \
    $(INCLUDES) \
    $(COMMON_FLAGS) \
    $(OPT_FLAGS) \
    $(OPENMP_FLAGS)

# ============================================================================
# LIBRARY PATHS
# ============================================================================

LIB_PATHS :=
RPATHS :=

ifneq ($(HAVE_CUDA_RUNTIME),no)

    LIB_PATHS += -L$(CUDA_LIBDIR)
    RPATHS += -Wl,-rpath,$(CUDA_LIBDIR)

endif

LIB_PATHS += $(OPTIONAL_LDFLAGS)
RPATHS += $(OPTIONAL_RPATHS)

# ============================================================================
# LIBRARIES
# ============================================================================

LIBS := \
    $(OPENCV_LIBS) \
    $(SDL2_LIBS) \
    -lavformat \
    -lavcodec \
    -lavutil \
    -lswscale \
    -lswresample \
    -lavdevice \
    $(OPTIONAL_LIBS) \
    -lpthread \
    -ldl \
    -lrt

ifneq ($(HAVE_CUDA_RUNTIME),no)

    LIBS += \
        -lcudart \
        -lcuda

endif

# ============================================================================
# VERBOSE / SILENT BUILD
# ============================================================================

ifeq ($(V),1)

    Q :=

else

    Q := @

endif

# ============================================================================
# TARGETS
# ============================================================================

.PHONY: all clean rebuild debug release info help

all: $(TARGET)

# ============================================================================
# LINK
# ============================================================================

$(TARGET): $(OBJ_FILE)

	$(Q)echo ""
	$(Q)echo "=============================================="
	$(Q)echo " LINKING: $@"
	$(Q)echo "=============================================="

	$(Q)$(CXX) \
	    -o $@ \
	    $^ \
	    $(LIB_PATHS) \
	    $(RPATHS) \
	    $(LIBS) \
	    $(OPENMP_FLAGS)

	$(Q)echo ""
	$(Q)echo "Build successful: $@"

# ============================================================================
# COMPILE
# ============================================================================

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)

	$(Q)echo "[C++] $<"

	$(Q)$(CXX) \
	    $(CXX_FLAGS) \
	    -c $< \
	    -o $@

# ============================================================================
# OBJECT DIRECTORY
# ============================================================================

$(OBJ_DIR):

	$(Q)mkdir -p -m 775 $(OBJ_DIR)

# ============================================================================
# DEPENDENCY FILES
# ============================================================================

-include $(DEPS_FILE)

# ============================================================================
# CLEAN
# ============================================================================

clean:

	$(Q)echo "Cleaning build artifacts..."
	$(Q)rm -rf $(OBJ_DIR) $(TARGET)

# ============================================================================
# REBUILD
# ============================================================================

rebuild: clean all

# ============================================================================
# DEBUG BUILD
# ============================================================================

debug:

	$(MAKE) BUILD=debug V=$(V)

# ============================================================================
# RELEASE BUILD
# ============================================================================

release:

	$(MAKE) BUILD=release V=$(V)

# ============================================================================
# INFORMATION
# ============================================================================

info:

	@echo ""
	@echo "=============================================="
	@echo " Build Configuration"
	@echo "=============================================="
	@echo "Architecture       : $(ARCH)"
	@echo "GPU Architecture   : $(GPU_ARCH)"
	@echo "Compiler           : $(CXX)"
	@echo "Source             : $(SRC_FILE)"
	@echo "Target             : $(TARGET)"
	@echo "Build Type         : $(BUILD)"
	@echo ""
	@echo "CUDA Path          : $(CUDA_PATH)"
	@echo "CUDA Runtime       : $(HAVE_CUDA_RUNTIME)"
	@echo "CUDA OpenCV        : $(HAVE_OPENCV_CUDA)"
	@echo "OpenCV             : $(HAVE_OPENCV)"
	@echo "SDL2               : $(HAVE_SDL2)"
	@echo "TensorRT           : $(HAVE_TENSORRT)"
	@echo "cuDNN              : $(HAVE_CUDNN)"
	@echo ""
	@echo "OpenCV Flags       : $(OPENCV_CFLAGS)"
	@echo "OpenCV Libraries   : $(OPENCV_LIBS)"
	@echo "SDL2 Flags         : $(SDL2_CFLAGS)"
	@echo "SDL2 Libraries     : $(SDL2_LIBS)"
	@echo "=============================================="
	@echo ""

# ============================================================================
# HELP
# ============================================================================

help:

	@echo ""
	@echo "Usage:"
	@echo ""
	@echo "  make                 Build release version"
	@echo "  make BUILD=debug     Build debug version"
	@echo "  make BUILD=release   Build release version"
	@echo "  make V=1             Show complete compiler commands"
	@echo "  make clean           Remove build artifacts"
	@echo "  make rebuild         Clean and rebuild"
	@echo "  make info            Show detected dependencies"
	@echo "  make debug           Debug build"
	@echo "  make release         Release build"
	@echo ""
	@echo "Optional dependencies:"
	@echo ""
	@echo "  ENABLE_TENSORRT=yes"
	@echo "  ENABLE_CUDNN=yes"
	@echo "  ENABLE_CUDA_OPENCV=yes"
	@echo ""
