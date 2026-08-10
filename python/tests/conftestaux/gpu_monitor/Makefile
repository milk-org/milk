# Makefile — build libcupti_spy.so
#
# Usage:
#   make                     # auto-detect CUDA_ROOT
#   make CUDA_ROOT=/opt/cuda  # explicit path
#
# Typical CUDA Toolkit layout assumed:
#   $CUDA_ROOT/include/          — cuda_runtime_api.h
#   $CUDA_ROOT/extras/CUPTI/include/ — cupti.h
#   $CUDA_ROOT/lib64/            — libcudart.so
#   $CUDA_ROOT/extras/CUPTI/lib64/  — libcupti.so
#
# If your distro splits CUPTI into a separate package (e.g. libcupti-dev
# on Ubuntu), cupti.h may land in /usr/local/cuda/extras/CUPTI/include or
# /usr/include.  Adjust CUPTI_INC / CUPTI_LIB below as needed.

CUDA_ROOT  ?= /usr/local/cuda
CUPTI_ROOT ?= $(CUDA_ROOT)/extras/CUPTI

CC      = gcc
CFLAGS  = -fPIC -O2 -Wall -Wextra -std=c11

INC     = -I$(CUDA_ROOT)/include \
          -I$(CUPTI_ROOT)/include

LDFLAGS = -shared \
          -L$(CUDA_ROOT)/lib64 \
          -L$(CUPTI_ROOT)/lib64 \
          -Wl,-rpath,$(CUDA_ROOT)/lib64 \
          -Wl,-rpath,$(CUPTI_ROOT)/lib64

LIBS    = -lcupti -lcudart

TARGET  = libcupti_spy.so

# ---------------------------------------------------------------------------

$(TARGET): cupti_spy.c
	$(CC) $(CFLAGS) $(INC) $(LDFLAGS) -o $@ $< $(LIBS)
	@echo "Built $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: clean
