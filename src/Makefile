include ../make.inc

MAIN_HEADER := ../include/superbblas.h
OTHER_HEADERS := \
  ../include/superbblas/superbblas_flags.h \
  ../include/superbblas/superbblas_lib.h \
  ../include/superbblas/blas_cblas_tmpl.hpp \
  ../include/superbblas/blas_ftn_tmpl.hpp \
  ../include/superbblas/blas.h \
  ../include/superbblas/cblas.h \
  ../include/superbblas/dist.h \
  ../include/superbblas/performance.h \
  ../include/superbblas/platform.h \
  ../include/superbblas/runtime_features.h \
  ../include/superbblas/template.h \
  ../include/superbblas/template_types.h \
  ../include/superbblas/template_undef.h \
  ../include/superbblas/tensor.h

SOURCE := $(MAIN_HEADER) $(OTHER_HEADERS)

INCLUDE := -I../include

$(BUILDDIR)/superbblas_flags.h: $(SOURCE)
	echo "// This file is generated automatically. Please don't modify" > $@
	echo "#ifndef __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@
	echo "#define __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@
ifeq ($(TARGET), cpu)
	$(CXX) -E $(INCLUDE) ${CXXFLAGS} -DSUPERBBLAS_CREATING_FLAGS ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -E ${MPISBFLAG} ${NVCCFLAGS} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' -DSUPERBBLAS_CREATING_FLAGS ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
	echo "#endif // __SUPERBBLAS_SUPERBBLAS_FLAGS__" >> $@

$(BUILDDIR)/superbblas_lib.cpp: $(SOURCE)
	echo "// This file is generated automatically. Please, do not modify" > $@
	echo "#include \"superbblas.h\"" >> $@
	echo "using namespace superbblas; using namespace detail;" >> $@
ifeq ($(TARGET), cpu)
	$(CXX) -E $(INCLUDE) ${CXXFLAGS} -DSUPERBBLAS_CREATING_LIB ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -E ${MPISBFLAG} ${NVCCFLAGS} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' -DSUPERBBLAS_CREATING_LIB ../include/superbblas.h | $(PYTHON) tools/ctemplate >> $@
endif

%.o: %.cpp
ifeq ($(TARGET), cpu)
	${CXX} -c $(INCLUDE) ${CXXFLAGS} $< -o $@
endif
ifeq ($(TARGET), cuda)
	${NVCC} -c ${INCLUDE} ${MPISBFLAG} ${NVCCFLAGS} ${NVCCSTDLANG} -Xcompiler '${CXXFLAGS} ${MPIINCFLAGS}' $< -o $@
endif

$(BUILDDIR)/superbblas_lib.o: $(SOURCE) $(BUILDDIR)/superbblas_flags.h

install: $(BUILDDIR)/superbblas_lib.o install_headers 
	@rm -f $(LIBRARY)
	install -d $(libdir)
ifeq ($(UNAME), Darwin)
	libtool -static -o $(LIBRARY) $< -no_warning_for_no_symbols
else
	$(AR) r $(LIBRARY) $<
	$(RANLIB) $(LIBRARY)
endif
	install -m 644 $(LIBRARY) $(libdir)

install_headers: $(BUILDDIR)/superbblas_flags.h
	install -d $(includedir)
	install -d $(includedir)/superbblas
	install -m 644 $(MAIN_HEADER) $(includedir)
	install -m 644 $(OTHER_HEADERS) $(includedir)/superbblas
	install -m 644 $(BUILDDIR)/superbblas_flags.h $(includedir)/superbblas

format:
	clang-format -i ${MAIN_HEADER} ${OTHER_HEADERS}

clean:
	rm -f $(BUILDDIR)/superbblas_flags.h $(BUILDDIR)/superbblas_lib.cpp $(BUILDDIR)/superbblas_lib.o $(LIBRARY)

.PHONY: install install_headers format clean
