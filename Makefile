
TARGET_EXEC := kcore_debug

BUILD_DIR := ./build
SRC_DIRS := ./src
NVCC := nvcc
CXX := g++

# 查找所有源码文件
SRCS := main.cu $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.cu')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS := $(INC_FLAGS) -MMD -MP

# GPU 架构参数
ComputeCapability := -arch=sm_86

# OpenMP 参数
CXXFLAGS += -fopenmp
CFLAGS += -fopenmp
# 给 nvcc 传递给主机编译器的 openmp 参数（必须用 -Xcompiler）
NVCCFLAGS += -Xcompiler -fopenmp 

# 链接时启用 openmp
LDFLAGS += -lgomp

# 链接目标
$(TARGET_EXEC): $(OBJS)
	$(NVCC) $(ComputeCapability) $(OBJS) -o $@ $(LDFLAGS) $(NVCCFLAGS)

# CUDA 编译规则
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(ComputeCapability) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# C++ 编译规则
$(BUILD_DIR)/%.cpp.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -r $(BUILD_DIR)

# 自动依赖文件
-include $(DEPS)
