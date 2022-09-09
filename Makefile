
include make.inc

install_cpu:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=cpu
	@$(MAKE) -C src install TARGET=cpu

install_cuda:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=cuda CXX=
	@$(MAKE) -C src install TARGET=cuda CXX=

install_hip:
	@mkdir -p $(BUILDDIR)
	@$(MAKE) -C src clean TARGET=hip CXX=
	@$(MAKE) -C src install TARGET=hip CXX=

test_cpu: install_cpu
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_cpu_lib

test_cuda: install_cuda
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_cuda_lib

test_hip: install_hip
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_hip_lib

test_header_only_cpu:
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_cpu

test_header_only_cuda:
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_cuda

test_header_only_hip:
	@$(MAKE) -C tests clean
	@$(MAKE) -C tests all_hip

test_cpu: export PREFIX := ${CURDIR}/test_lib_cpu
test_cpu: export BUILDDIR := ${CURDIR}/test_lib_cpu
test_cuda: export PREFIX := ${CURDIR}/test_lib_cuda
test_cuda: export BUILDDIR := ${CURDIR}/test_lib_cuda
test_hip: export PREFIX := ${CURDIR}/test_lib_hip
test_hip: export BUILDDIR := ${CURDIR}/test_lib_hip
test_header_only_cpu test_header_only_cuda test_header_only_hip: export SB_LDFLAGS :=
test_header_only_cpu test_header_only_cuda test_header_only_hip: export SB_INCLUDE := -I../include

format:
	${MAKE} -C src format

.PHONY: install_cpu install_cuda install_hip test_cpu test_cuda test_hip test_header_only_cpu test_header_only_cuda test_header_only_hip format
.NOTPARALLEL:
