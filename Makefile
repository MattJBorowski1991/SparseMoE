NVCC ?= nvcc

NVCC_FLAGS = -O3 -lineinfo -Xcompiler -Wall

MAXRREGCOUNT ?= 
ifeq($MAXRREGCOUNT,)
	NVCC_REGCOUNT = 
else
	NVCC_REGCOUNT = -maxrregcount=$(MAXRREGCOUNT)
endif

NVCC_ARCH ?= 89

NVCC_GENCODE = 	-gencode arch=compute_$(NVCC_ARCH),code=sm_$(NVCC_ARCH) \
				-gencode arch=compute_$(NVCC_ARCH),code=compute_$(NVCC_ARCH)

INCLUDE_FLAGS = -I. -I./include

DRIVERS_DIR = drivers
KERNELS_DIR = kernels
INPUTS_DIR = inputs
BIN_DIR = bin
UTILS_DIR = utils

KERNEL ?= baseline

PROFILE_WARMUPS ?= 3
PROFILE_RUNS ?= 15
NCU ?= ncu
NCU_SET ?= full
NCU_FLAGS ?=

TARGET = $(BIN_DIR)/profile_$(KERNEL)

BASE_SRCS = $(DRIVERS_DIR)/main.cu \
			$(INPUTS_DIR)/data.cu

KERNEL_SRCS = $(KERNELS_DIR)/$(KERNEL).cu

SRC = $(BASE_SRCS) $(KERNEL_SRCS)

.PHONY: all run profile clean

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SRC) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_REGCOUNT) $(NVCC_GENCODE) $(INCLUDE_FLAGS) $(SRC) -o $(TARGET)

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) --kernel=$(KERNEL)

profile: $(TARGET)
	$(NCU) --set $(NCU_SET) $(NCU_FLAGS) ./$(TARGET) --kernel=$(KERNEL) --warmups=$(PROFILE_WARMUPS) --runs=$(PROFILE_RUNS)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(KERNELS_DIR)/*.o $(INPUTS_DIR)/*.o $(DRIVERS_DIR)/*.o $(UTILS_DIR)/*.o