# Override ARCH if -arch=native is not supported by your nvcc (needs CUDA >= 11.6)
#   e.g.  make ARCH=-arch=sm_75
NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS  = -O3 -std=c++17 $(ARCH) -lineinfo -Xptxas -v
LIBS   = -lcublas

# -lineinfo lets Nsight Compute map metrics to source lines.
# -Xptxas -v prints registers / shared memory / spills per kernel. Read it!

sgemm: src/main.cu src/kernels.cuh src/common.cuh
	$(NVCC) $(FLAGS) -o $@ src/main.cu $(LIBS) 2> ptxas_info.txt || (cat ptxas_info.txt; false)
	@grep -E "Compiling entry|registers|spill" ptxas_info.txt | sed 's/ptxas info    : //' || true

# Instructor only: build against the reference solutions.
solution: src/main.cu instructor/solution_kernels.cuh src/common.cuh
	$(NVCC) $(FLAGS) -DUSE_SOLUTION -o sgemm_solution src/main.cu $(LIBS)

clean:
	rm -f sgemm sgemm_solution ptxas_info.txt

.PHONY: clean solution
