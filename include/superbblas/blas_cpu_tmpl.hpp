
#ifndef THIS_FILE
#    define THIS_FILE "blas_cpu_tmpl.hpp"
#endif

#include "platform.h"
#include "template.h"
#ifdef SUPERBBLAS_USE_MKL
#    include "mkl.h"
#    include "mkl_spblas.h"
#    ifndef SUPERBBLAS_USE_CBLAS
#        define SUPERBBLAS_USE_CBLAS
#    endif
#    define MKL_SCALAR ARITH(, , float, MKL_Complex8, double, MKL_Complex16, , )
#elif defined(SUPERBBLAS_USE_CBLAS)
#    include "cblas.h"
#endif // SUPERBBLAS_USE_MKL

#ifdef SUPERBBLAS_USE_MKL
#    define LAPACK_SCALAR MKL_SCALAR
#else
#    define LAPACK_SCALAR SCALAR
#endif

#include <complex>
#include <vector>

#if !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)

namespace superbblas {

    namespace detail {

#    ifdef SUPERBBLAS_USE_MKL
#        define BLASINT MKL_INT
#    else
#        define BLASINT int
#    endif

#    define REAL ARITH(, , float, float, double, double, , )
#    define SCALAR ARITH(, , float, std::complex<float>, double, std::complex<double>, , )

        //
        // Basic BLAS
        //

        using BLASSTRING = const char *;

#    ifndef SUPERBBLAS_USE_CBLAS

// clang-format off
#define XCOPY     FORTRAN_FUNCTION(ARITH(hcopy , kcopy , scopy , ccopy , dcopy , zcopy , , ))
#define XSWAP     FORTRAN_FUNCTION(ARITH(hswap , kswap , sswap , cswap , dswap , zswap , , ))
#define XGEMM     FORTRAN_FUNCTION(ARITH(hgemm , kgemm , sgemm , cgemm , dgemm , zgemm , , ))
#define XTRMM     FORTRAN_FUNCTION(ARITH(htrmm , ktrmm , strmm , ctrmm , dtrmm , ztrmm , , ))
#define XTRSM     FORTRAN_FUNCTION(ARITH(htrsm , ktrsm , strsm , ctrsm , dtrsm , ztrsm , , ))
#define XHEMM     FORTRAN_FUNCTION(ARITH(hsymm , khemm , ssymm , chemm , dsymm , zhemm , , ))
#define XHEMV     FORTRAN_FUNCTION(ARITH(hsymv , khemv , ssymv , chemv , dsymv , zhemv , , ))
#define XAXPY     FORTRAN_FUNCTION(ARITH(haxpy , kaxpy , saxpy , caxpy , daxpy , zaxpy , , ))
#define XGEMV     FORTRAN_FUNCTION(ARITH(hgemv , kgemv , sgemv , cgemv , dgemv , zgemv , , ))
#define XDOT      FORTRAN_FUNCTION(ARITH(hdot  ,       , sdot  ,       , ddot  ,       , , ))
#define XSCAL     FORTRAN_FUNCTION(ARITH(hscal , kscal , sscal , cscal , dscal , zscal , , ))
        // clang-format on

        extern "C" {

        void XCOPY(BLASINT *n, const SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
        void XSWAP(BLASINT *n, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
        void XGEMM(BLASSTRING transa, BLASSTRING transb, BLASINT *m, BLASINT *n, BLASINT *k,
                   SCALAR *alpha, const SCALAR *a, BLASINT *lda, const SCALAR *b, BLASINT *ldb,
                   SCALAR *beta, SCALAR *c, BLASINT *ldc);
        void XGEMV(BLASSTRING transa, BLASINT *m, BLASINT *n, SCALAR *alpha, const SCALAR *a,
                   BLASINT *lda, const SCALAR *x, BLASINT *incx, SCALAR *beta, SCALAR *y,
                   BLASINT *incy);
        void XTRMM(BLASSTRING side, BLASSTRING uplo, BLASSTRING transa, BLASSTRING diag, BLASINT *m,
                   BLASINT *n, SCALAR *alpha, SCALAR *a, BLASINT *lda, SCALAR *b, BLASINT *ldb);
        void XTRSM(BLASSTRING side, BLASSTRING uplo, BLASSTRING transa, BLASSTRING diag, BLASINT *m,
                   BLASINT *n, SCALAR *alpha, const SCALAR *a, BLASINT *lda, SCALAR *b,
                   BLASINT *ldb);
        void XHEMM(BLASSTRING side, BLASSTRING uplo, BLASINT *m, BLASINT *n, SCALAR *alpha,
                   SCALAR *a, BLASINT *lda, SCALAR *b, BLASINT *ldb, SCALAR *beta, SCALAR *c,
                   BLASINT *ldc);
        void XHEMV(BLASSTRING uplo, BLASINT *n, SCALAR *alpha, SCALAR *a, BLASINT *lda, SCALAR *x,
                   BLASINT *lncx, SCALAR *beta, SCALAR *y, BLASINT *lncy);
        void XAXPY(BLASINT *n, SCALAR *alpha, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
// NOTE: avoid calling Fortran functions that return complex values
#        ifndef __SUPERBBLAS_USE_COMPLEX
        SCALAR XDOT(BLASINT *n, SCALAR *x, BLASINT *incx, SCALAR *y, BLASINT *incy);
#        endif // __SUPERBBLAS_USE_COMPLEX
        void XSCAL(BLASINT *n, SCALAR *alpha, SCALAR *x, BLASINT *incx);
        }

#    else //  SUPERBBLAS_USE_CBLAS

// Pass constant values by value for non-complex types, and by reference otherwise
#        define PASS_SCALAR(X) ARITH(X, &(X), X, &(X), X, &(X), X, &(X))
#        define PASS_SCALARpp(X)                                                                   \
            ARITH(X, (void **)(X), X, (void **)(X), X, (void **)(X), X, (void **)(X))
#        define PASS_SCALARcpp(X)                                                                  \
            ARITH(X, (const void **)(X), X, (const void **)(X), X, (const void **)(X), X,          \
                  (const void **)(X))

#        define CBLAS_FUNCTION(X) CONCAT(cblas_, X)

// clang-format off
#define XCOPY     CBLAS_FUNCTION(ARITH(hcopy , kcopy , scopy , ccopy , dcopy , zcopy , , ))
#define XSWAP     CBLAS_FUNCTION(ARITH(hswap , kswap , sswap , cswap , dswap , zswap , , ))
#define XGEMM     CBLAS_FUNCTION(ARITH(hgemm , kgemm , sgemm , cgemm , dgemm , zgemm , , ))
#define XTRMM     CBLAS_FUNCTION(ARITH(htrmm , ktrmm , strmm , ctrmm , dtrmm , ztrmm , , ))
#define XTRSM     CBLAS_FUNCTION(ARITH(htrsm , ktrsm , strsm , ctrsm , dtrsm , ztrsm , , ))
#define XHEMM     CBLAS_FUNCTION(ARITH(hsymm , khemm , ssymm , chemm , dsymm , zhemm , , ))
#define XHEMV     CBLAS_FUNCTION(ARITH(hsymv , khemv , ssymv , chemv , dsymv , zhemv , , ))
#define XAXPY     CBLAS_FUNCTION(ARITH(haxpy , kaxpy , saxpy , caxpy , daxpy , zaxpy , , ))
#define XGEMV     CBLAS_FUNCTION(ARITH(hgemv , kgemv , sgemv , cgemv , dgemv , zgemv , , ))
#define XSCAL     CBLAS_FUNCTION(ARITH(hscal , kscal , sscal , cscal , dscal , zscal , , ))
#define XDOT      CBLAS_FUNCTION(ARITH(hdot  , kdotc_sub, sdot, cdotc_sub, ddot, zdotc_sub, , ))
        // clang-format on

#        ifndef __SUPERBBLAS_BLAS_CPU_PRIVATE
#            define __SUPERBBLAS_BLAS_CPU_PRIVATE
        inline CBLAS_TRANSPOSE toCblasTrans(char trans) {
            switch (trans) {
            case 'n':
            case 'N': return CblasNoTrans;
            case 't':
            case 'T': return CblasTrans;
            case 'c':
            case 'C': return CblasConjTrans;
            default: throw std::runtime_error("Not valid value of trans");
            }
        }

        inline CBLAS_SIDE toCblasSide(char side) {
            switch (side) {
            case 'l':
            case 'L': return CblasLeft;
            case 'r':
            case 'R': return CblasRight;
            default: throw std::runtime_error("Not valid value of side");
            }
        }

        inline CBLAS_UPLO toCblasUplo(char uplo) {
            switch (uplo) {
            case 'u':
            case 'U': return CblasUpper;
            case 'l':
            case 'L': return CblasLower;
            default: throw std::runtime_error("Not valid value of uplo");
            }
        }

        inline CBLAS_DIAG toCblasDiag(char diag) {
            switch (diag) {
            case 'n':
            case 'N': return CblasNonUnit;
            case 'u':
            case 'U': return CblasUnit;
            default: throw std::runtime_error("Not valid value of diag");
            }
        }
#        endif // __SUPERBBLAS_BLAS_CPU_PRIVATE

#    endif //  SUPERBBLAS_USE_CBLAS

// clang-format off
#define XPOTRF    FORTRAN_FUNCTION(ARITH(hpotrf, kpotrf, spotrf, cpotrf, dpotrf, zpotrf, , ))
#define XGETRF    FORTRAN_FUNCTION(ARITH(hgetrf, kgetrf, sgetrf, cgetrf, dgetrf, zgetrf, , ))
#define XGETRI    FORTRAN_FUNCTION(ARITH(hgetri, kgetri, sgetri, cgetri, dgetri, zgetri, , ))
#define XGETRS    FORTRAN_FUNCTION(ARITH(hgetrs, kgetrs, sgetrs, cgetrs, dgetrs, zgetrs, , ))
        // clang-format on

#    ifndef SUPERBBLAS_USE_MKL
        extern "C" {
        void XPOTRF(BLASSTRING uplo, BLASINT *n, SCALAR *a, BLASINT *lda, BLASINT *info);
        void XGETRF(BLASINT *m, BLASINT *n, SCALAR *a, BLASINT *lda, BLASINT *ipivot,
                    BLASINT *info);
        void XGETRS(BLASSTRING trans, BLASINT *n, BLASINT *m, SCALAR *a, BLASINT *lda,
                    BLASINT *ipivot, SCALAR *b, BLASINT *ldb, BLASINT *info);
        void XGETRI(BLASINT *n, SCALAR *a, BLASINT *lda, BLASINT *piv, SCALAR *work, BLASINT *lwork,
                    BLASINT *info);
        }
#    endif // SUPERBBLAS_USE_MKL

        inline void xcopy(BLASINT n, const SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XCOPY(&n, x, &incx, y, &incy);
#    else
            XCOPY(n, x, incx, y, incy);
#    endif
        }

        inline void xswap(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XSWAP(&n, x, &incx, y, &incy);
#    else
            XSWAP(n, x, incx, y, incy);
#    endif
        }

        inline void xgemm(char transa, char transb, BLASINT m, BLASINT n, BLASINT k, SCALAR alpha,
                          const SCALAR *a, BLASINT lda, const SCALAR *b, BLASINT ldb, SCALAR beta,
                          SCALAR *c, BLASINT ldc, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XGEMM(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#    else
            XGEMM(CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n, k,
                  PASS_SCALAR(alpha), a, lda, b, ldb, PASS_SCALAR(beta), c, ldc);
#    endif
        }

        inline void xgemv(char transa, BLASINT m, BLASINT n, SCALAR alpha, const SCALAR *a,
                          BLASINT lda, const SCALAR *x, BLASINT incx, SCALAR beta, SCALAR *y,
                          BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XGEMV(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#    else
            XGEMV(CblasColMajor, toCblasTrans(transa), m, n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
#    endif
        }

        inline void xtrmm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XTRMM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#    else
            XTRMM(CblasColMajor, toCblasSide(side), toCblasUplo(uplo), toCblasTrans(transa),
                  toCblasDiag(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
#    endif
        }

        inline void xtrsm(char side, char uplo, char transa, char diag, BLASINT m, BLASINT n,
                          SCALAR alpha, const SCALAR *a, BLASINT lda, SCALAR *b, BLASINT ldb, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XTRSM(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
#    else
            XTRSM(CblasColMajor, toCblasSide(side), toCblasUplo(uplo), toCblasTrans(transa),
                  toCblasDiag(diag), m, n, PASS_SCALAR(alpha), a, lda, b, ldb);
#    endif
        }

        inline void xhemm(char side, char uplo, BLASINT m, BLASINT n, SCALAR alpha, SCALAR *a,
                          BLASINT lda, SCALAR *b, BLASINT ldb, SCALAR beta, SCALAR *c, BLASINT ldc,
                          Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XHEMM(&side, &uplo, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
#    else
            XHEMM(CblasColMajor, toCblasSide(side), toCblasUplo(uplo), m, n, PASS_SCALAR(alpha), a,
                  lda, b, ldb, PASS_SCALAR(beta), c, ldc);
#    endif
        }

        inline void xhemv(char uplo, BLASINT n, SCALAR alpha, SCALAR *a, BLASINT lda, SCALAR *x,
                          BLASINT incx, SCALAR beta, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XHEMV(&uplo, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
#    else
            XHEMV(CblasColMajor, toCblasUplo(uplo), n, PASS_SCALAR(alpha), a, lda, x, incx,
                  PASS_SCALAR(beta), y, incy);
#    endif
        }

        inline void xaxpy(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy,
                          Cpu) {
#    ifndef SUPERBBLAS_USE_CBLAS
            XAXPY(&n, &alpha, x, &incx, y, &incy);
#    else
            XAXPY(n, PASS_SCALAR(alpha), x, incx, y, incy);
#    endif
        }

        inline SCALAR xdot(BLASINT n, SCALAR *x, BLASINT incx, SCALAR *y, BLASINT incy, Cpu) {
#    ifndef __SUPERBBLAS_USE_COMPLEX
#        ifndef SUPERBBLAS_USE_CBLAS
            return XDOT(&n, x, &incx, y, &incy);
#        else
            return XDOT(n, x, incx, y, incy);
#        endif
#    else
            SCALAR r = (SCALAR)0.0;
#        ifdef _OPENMP
#            pragma omp for
#        endif
            for (int i = 0; i < n; i++) r += std::conj(x[i * incx]) * y[i * incy];
            return r;
#    endif // __SUPERBBLAS_USE_COMPLEX
        }

        inline void xscal(BLASINT n, SCALAR alpha, SCALAR *x, BLASINT incx, Cpu) {
            if (std::fabs(alpha) == SCALAR{0.0}) {
#    ifdef _OPENMP
#        pragma omp for
#    endif
                for (BLASINT i = 0; i < n; ++i) x[i * incx] = SCALAR{0};
                return;
            }
            if (alpha == SCALAR{1.0}) return;
#    ifndef SUPERBBLAS_USE_CBLAS
            XSCAL(&n, &alpha, x, &incx);
#    else
            XSCAL(n, PASS_SCALAR(alpha), x, incx);
#    endif
        }

        inline BLASINT xpotrf(char uplo, BLASINT n, SCALAR *a, BLASINT lda, Cpu) {
            /* Zero dimension matrix may cause problems */
            if (n == 0) return 0;

            BLASINT linfo = 0;
            XPOTRF(&uplo, &n, (LAPACK_SCALAR *)a, &lda, &linfo);
            return linfo;
        }

        int xgetrf(BLASINT m, BLASINT n, SCALAR *a, BLASINT lda, std::int64_t *ipivot, Cpu) {
            /* Zero dimension matrix may cause problems */
            if (m == 0 || n == 0) return 0;

            BLASINT linfo = 0;
            XGETRF(&m, &n, (LAPACK_SCALAR *)a, &lda, (BLASINT *)ipivot, &linfo);
            return linfo;
        }

        int xgetri(BLASINT n, SCALAR *a, BLASINT lda, BLASINT *ipivot, SCALAR *work, BLASINT lwork,
                   Cpu) {
            /* Zero dimension matrix may cause problems */
            if (n == 0) return 0;

            BLASINT info = 0;
            XGETRI(&n, (LAPACK_SCALAR *)a, &lda, ipivot, (LAPACK_SCALAR *)work, &lwork, &info);
            return info;
        }

        int xgetrs(char trans, BLASINT n, BLASINT nrhs, SCALAR *a, BLASINT lda,
                   std::int64_t *ipivot, SCALAR *b, BLASINT ldb, Cpu) {
            /* Zero dimension matrix may cause problems */
            if (n == 0 || nrhs == 0) return 0;

            BLASINT info = 0;
            XGETRS(&trans, &n, &nrhs, (LAPACK_SCALAR *)a, &lda, (BLASINT *)ipivot,
                   (LAPACK_SCALAR *)b, &ldb, &info);
            return info;
        }

#    undef XCOPY
#    undef XSWAP
#    undef XGEMM
#    undef XTRMM
#    undef XTRSM
#    undef XHEMM
#    undef XHEMV
#    undef XAXPY
#    undef XGEMV
#    undef XDOT
#    undef XSCAL
#    undef XPOTRF
#    undef XGETRF
#    undef XGETRI
#    undef XGETRS

        //
        // Batched GEMM
        //

#    ifdef SUPERBBLAS_USE_MKL

        inline void xgemm_batch_strided(char transa, char transb, int m, int n, int k, SCALAR alpha,
                                        const SCALAR *a, int lda, int stridea, const SCALAR *b,
                                        int ldb, int strideb, SCALAR beta, SCALAR *c, int ldc,
                                        int stridec, int batch_size, Cpu) {
#        if INTEL_MKL_VERSION >= 20210000
            if (lda <= stridea && ldb <= strideb && ldc <= stridec) {
                CONCAT(cblas_, CONCAT(ARITH(, , s, c, d, z, , ), gemm_batch_strided))
                (CblasColMajor, toCblasTrans(transa), toCblasTrans(transb), m, n, k,
                 PASS_SCALAR(alpha), a, lda, stridea, b, ldb, strideb, PASS_SCALAR(beta), c, ldc,
                 stridec, batch_size);
                return;
            }
#        endif

            CBLAS_TRANSPOSE transa_ = toCblasTrans(transa), transb_ = toCblasTrans(transb);
            std::vector<const SCALAR *> av(batch_size), bv(batch_size);
            std::vector<SCALAR *> cv(batch_size);
            for (int i = 0; i < batch_size; ++i) av[i] = a + i * stridea;
            for (int i = 0; i < batch_size; ++i) bv[i] = b + i * strideb;
            for (int i = 0; i < batch_size; ++i) cv[i] = c + i * stridec;
            CONCAT(cblas_, CONCAT(ARITH(, , s, c, d, z, , ), gemm_batch))
            (CblasColMajor, &transa_, &transb_, &m, &n, &k, &alpha, PASS_SCALARcpp(av.data()), &lda,
             PASS_SCALARcpp(bv.data()), &ldb, &beta, PASS_SCALARpp(cv.data()), &ldc, 1,
             &batch_size);
        }

#    else // SUPERBBLAS_USE_MKL

        inline void xgemm_batch_strided(char transa, char transb, int m, int n, int k, SCALAR alpha,
                                        const SCALAR *a, int lda, int stridea, const SCALAR *b,
                                        int ldb, int strideb, SCALAR beta, SCALAR *c, int ldc,
                                        int stridec, int batch_size, Cpu cpu) {

#        ifdef _OPENMP
#            pragma omp for
#        endif
            for (int i = 0; i < batch_size; ++i) {
                xgemm(transa, transb, m, n, k, alpha, a + stridea * i, lda, b + strideb * i, ldb,
                      beta, c + stridec * i, ldc, cpu);
            }
        }

#    endif // SUPERBBLAS_USE_MKL

#    ifdef SUPERBBLAS_USE_CBLAS
#        undef PASS_SCALAR
#        undef CBLAS_FUNCTION
#    endif

        //
        // MKL Sparse
        //

#    ifdef SUPERBBLAS_USE_MKL
// Change types to MKL
// #        define PASS_SCALAR(X) *(MKL_SCALAR *)&(X)
#        define PASS_SCALAR(X)                                                                     \
            ARITH(, , (X), (MKL_SCALAR{std::real(X), std::imag(X)}), (X),                          \
                  (MKL_SCALAR{std::real(X), std::imag(X)}), , )

#        define MKL_SP_FUNCTION(X) CONCAT(mkl_sparse_, X)

// clang-format off
#define XSPCREATEBSR    MKL_SP_FUNCTION(ARITH( , , s_create_bsr , c_create_bsr , d_create_bsr , z_create_bsr , , ))
#define XSPMM           MKL_SP_FUNCTION(ARITH( , , s_mm , c_mm , d_mm , z_mm , , ))
        // clang-format on

        inline sparse_status_t mkl_sparse_create_bsr(sparse_matrix_t *A,
                                                     sparse_index_base_t indexing,
                                                     sparse_layout_t block_layout, BLASINT rows,
                                                     BLASINT cols, BLASINT block_size,
                                                     BLASINT *rows_start, BLASINT *rows_end,
                                                     BLASINT *col_indx, SCALAR *values) {
            static_assert(sizeof(SCALAR) == sizeof(MKL_SCALAR));
            return XSPCREATEBSR(A, indexing, block_layout, rows, cols, block_size, rows_start,
                                rows_end, col_indx, (MKL_SCALAR *)values);
        }

        inline sparse_status_t mkl_sparse_mm(const sparse_operation_t operation, SCALAR alpha,
                                             sparse_matrix_t A, struct matrix_descr descr,
                                             sparse_layout_t layout, const SCALAR *B,
                                             BLASINT columns, BLASINT ldb, SCALAR beta, SCALAR *C,
                                             BLASINT ldc) {
            static_assert(sizeof(SCALAR) == sizeof(MKL_SCALAR));
            return XSPMM(operation, PASS_SCALAR(alpha), A, descr, layout, (MKL_SCALAR *)B, columns,
                         ldb, PASS_SCALAR(beta), (MKL_SCALAR *)C, ldc);
        }

#        undef PASS_SCALAR
#        undef MKL_SCALAR
#        undef MKL_SP_FUNCTION

#    endif // SUPERBBLAS_USE_MKL

#    if defined(SUPERBBLAS_USE_CUDA)

#        define CUSPARSE_SCALAR ARITH(, , float, cuComplex, double, cuDoubleComplex, , )

        inline cusparseStatus_t
        cusparseXbsrmm(cusparseHandle_t handle, cusparseDirection_t dirA,
                       cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n,
                       int kb, int nnzb, SCALAR alpha, const cusparseMatDescr_t descrA,
                       const SCALAR *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
                       int blockDim, const SCALAR *B, int ldb, SCALAR beta, SCALAR *C, int ldc) {
            return ARITH(, , cusparseSbsrmm, cusparseCbsrmm, cusparseDbsrmm, cusparseZbsrmm, , )(
                handle, dirA, transA, transB, mb, n, kb, nnzb, (const CUSPARSE_SCALAR *)&alpha,
                descrA, (const CUSPARSE_SCALAR *)bsrValA, bsrRowPtrA, bsrColIndA, blockDim,
                (const CUSPARSE_SCALAR *)B, ldb, (const CUSPARSE_SCALAR *)&beta,
                (CUSPARSE_SCALAR *)C, ldc);
        }

#        undef CUSPARSE_SCALAR

#    elif defined(SUPERBBLAS_USE_HIP)

#        define HIPSPARSE_SCALAR ARITH(, , float, hipComplex, double, hipDoubleComplex, , )

        inline hipsparseStatus_t
        hipsparseXbsrmm(hipsparseHandle_t handle, hipsparseDirection_t dirA,
                        hipsparseOperation_t transA, hipsparseOperation_t transB, int mb, int n,
                        int kb, int nnzb, SCALAR alpha, const hipsparseMatDescr_t descrA,
                        const SCALAR *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA,
                        int blockDim, const SCALAR *B, int ldb, SCALAR beta, SCALAR *C, int ldc) {
            return ARITH(, , hipsparseSbsrmm, hipsparseCbsrmm, hipsparseDbsrmm, hipsparseZbsrmm,
                         , )(handle, dirA, transA, transB, mb, n, kb, nnzb,
                             (const HIPSPARSE_SCALAR *)&alpha, descrA,
                             (const HIPSPARSE_SCALAR *)bsrValA, bsrRowPtrA, bsrColIndA, blockDim,
                             (const HIPSPARSE_SCALAR *)B, ldb, (const HIPSPARSE_SCALAR *)&beta,
                             (HIPSPARSE_SCALAR *)C, ldc);
        }

#        undef HIPSPARSE_SCALAR

#    endif

#    undef BLASINT
#    undef REAL
#    undef SCALAR
    }
}
#endif // !defined(__SUPERBBLAS_USE_HALF) && !defined(__SUPERBBLAS_USE_HALFCOMPLEX)
