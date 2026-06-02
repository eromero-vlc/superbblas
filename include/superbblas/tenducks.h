/// TENsor Dynamic User-defined Contraction K-dimensional Subroutine (TenDuCKS)

/// Proposed strategy for matrix-matrix multiplication:
/// - Make coordinate transformation on the fly

#ifndef __SUPERBBLAS_TENDUCKS__
#define __SUPERBBLAS_TENDUCKS__

#include "blas.h"

#if defined(SUPERBBLAS_CREATING_LIB)
#    define COOR_DIMS_MULT_4 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48

/// Generate template instantiations for sptensor_tensor_product_gpu functions with template parameter N, T, IndexType

#    define DECL_SPTENSOR_TENSOR_PRODUCT_GPU_N_T_IDX(...)                                          \
        EMIT REPLACE1(sptensor_tensor_product, superbblas::detail::sptensor_tensor_product<N, T>)  \
            REPLACE_T REPLACE(N, COOR_DIMS_MULT_4) template __VA_ARGS__;

#else
#    define DECL_SPTENSOR_TENSOR_PRODUCT_GPU_N_T_IDX(...) __VA_ARGS__
#endif

namespace superbblas {
    namespace detail {
        namespace aux_sptensor_tensor_product {
            template <typename IndexType> IndexType volume(int N, const IndexType *dim) {
                IndexType vol = 1;
                for (int i = 0; i < N; ++i) vol *= dim[i];
                return vol;
            }

            template <typename IndexType>
            IndexType volume(int N, const IndexType *p, const IndexType *dim) {
                IndexType vol = 1;
                for (int i = 0; i < N; ++i) vol *= dim[p[i]];
                return vol;
            }

            template <typename IndexType>
            void copy_coor(int N, const IndexType *dim, IndexType *coor) {
                for (int i = 0; i < N; ++i) { coor[i] = dim[i]; }
            }

            template <typename IndexType>
            void copy_coor(int N, const IndexType *dim, const IndexType *p, IndexType *coor) {
                for (int i = 0; i < N; ++i) { coor[i] = dim[p[i]]; }
            }

            template <typename IndexType>
            bool is_compatible(int N, const IndexType *p, const IndexType *coor_p,
                               const IndexType *coor_b) {
                for (int i = 0; i < N; ++i) {
                    if (coor_p[p[i]] != coor_b[i]) return false;
                }
                return true;
            }

            template <typename IndexType>
            void copy_stride(int N, const IndexType *dim, IndexType *stride) {
                IndexType step = 1;
                for (int i = 0; i < N; ++i) {
                    stride[i] = step;
                    step *= dim[i];
                }
            }

            template <typename IndexType>
            void copy_stride(int N, const IndexType *dim, const IndexType *p, IndexType *stride) {
                IndexType step = 1;
                for (int i = 0; i < N; ++i) {
                    stride[i] = step;
                    step *= dim[p[i]];
                }
            }

            std::tuple<std::vector<char>, std::vector<IndexType>,                         //
                       std::vector<char>, std::vector<IndexType>, std::vector<IndexType>, //
                       std::vector<char>, std::vector<IndexType>, std::vector<IndexType>, //
                       IndexType, IndexType, IndexType>
            sptensor_tensor_product_preparation(
                int s_rows_N, const IndexType *s_dim_rows, int s_cols_N,
                const IndexType *s_dim_cols, const IndexType *s_from_rows,
                const IndexType *s_from_cols, const IndexType *s_size_rows,
                const IndexType *s_size_cols, //
                int x_N, const IndexType *x_dim, const IndexType *x_from,
                const IndexType *x_size, //
                int y_N, const IndexType *y_dim, const IndexType *y_from,
                const IndexType *y_size, //
                const IndexType *p_ABT, const IndexType *p_CBT, const IndexType *p_ACT, int NA,
                int NB, int NC, int NT) {

                (void)s_from_rows;
                (void)x_from;
                (void)y_from;

                // Some basic checks
                for (const auto &[N, size, dim] :
                     std::array<std::tuple<int, const IndexType *, const IndexType *>, 4>{
                         {{s_rows_N, s_size_rows, s_dim_rows},
                          {s_cols_N, s_size_cols, s_dim_cols},
                          {x_N, x_size, x_dim},
                          {y_N, y_size, y_dim}}}) {
                    for (int i = 0; i < N; ++i) {
                        if (size[i] > dim[i]) throw std::runtime_error("invalid input");
                    }
                }
                if (NA + NB + NT != s_rows_N + s_cols_N || NC + NB + NT != x_N ||
                    NA + NC + NT != y_N)
                    throw std::runtime_error("invalid input");

                // Find the volume for each part
                std::vector<IndexType> size_B(NB);
                copy_coor(NB, x_size, p_CBT + NC, size_B.data());
                std::vector<IndexType> size_C(NC);
                copy_coor(NC, x_size, p_CBT, size_C.data());
                {
                    std::vector<IndexType> size_A(NA);
                    copy_coor(NA, y_size, p_ACT, size_A.data());
                    std::vector<IndexType> size_T(NT);
                    copy_coor(NT, x_size, p_CBT + NC + NB, size_T.data());
                    std::vector<IndexType> s_size(s_rows_N + s_cols_N);
                    copy_coor(s_rows_N, s_size_rows, s_size.data());
                    copy_coor(s_cols_N, s_size_cols, s_size.data() + s_rows_N);
                    if (!is_compatible(NA, p_ABT, s_size.data(), size_A.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NB, p_ABT + NA, s_size.data(), size_B.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_ABT + NA + NB, s_size.data(), size_T.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_CBT + NC + NB, x_size, size_T.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NC, p_ACT + NA, y_size, size_C.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_ACT + NA + NC, y_size, size_T.data()))
                        throw std::runtime_error("invalid input");
                }

                // Split A and T coordinates into the sparse rows and columns coordinates
                // Find the "a" coor rows in B
                std::vector<IndexType> p_A_rows, p_A_cols;
                for (int i = 0; i < NA; ++i) {
                    if (p_ABT[i] < s_rows_N)
                        p_A_rows.push_back(p_ABT[i]);
                    else
                        p_A_cols.push_back(p_ABT[i] - s_rows_N);
                }
                for (int i = 0; i < NB; ++i) {
                    if (p_ABT[NA + i] < s_rows_N) {
                        // FIXME: support "a" row coors in B
                        throw std::runtime_error("unsupported");
                    }
                }
                std::vector<IndexType> p_T_rows, p_T_cols;
                for (int i = 0; i < NT; ++i) {
                    if (p_ABT[NA + NB + i] < s_rows_N) {
                        p_T_rows.push_back(p_ABT[NA + NB + i]);
                    } else {
                        p_T_cols.push_back(p_ABT[NA + NB + i] - s_rows_N);
                    }
                }

                // Get the strides and permutations
                std::vector<IndexType> stride_A_rows(p_A_rows.size());
                copy_stride(p_A_rows.size(), s_size_rows, p_A_rows.data(), stride_A_rows.data());
                std::vector<IndexType> stride_s_dim_cols(s_cols_N);
                copy_stride(s_cols_N, s_dim_cols, stride_s_dim_cols.data());

                std::vector<IndexType> stride_C(NC);
                copy_stride(NC, size_C.data(), stride_C.data());

                std::vector<IndexType> stride_T_rows(p_T_rows.size());
                copy_stride(p_T_rows.size(), s_size_rows, p_T_rows.data(), stride_T_rows.data());

                std::vector<IndexType> stride_s_rows(s_rows_N);
                copy_stride(s_rows_N, s_size_rows, stride_s_rows.data());

                // Get strides for the rows of the sparse matrix
                std::vector<IndexType> perm_stride_rows(s_rows_N);
                std::vector<char> perm_index_rows(s_rows_N);
                for (int i = 0; i < s_rows_N; ++i) {
                    const std::size_t j =
                        std::find(p_A_rows.begin(), p_A_rows.end(), i) - p_A_rows.begin();
                    if (j < p_A_rows.size()) {
                        perm_stride_rows.at(i) = stride_A_rows.at(j);
                        perm_index_rows.at(i) = 0;
                    } else {
                        const auto j =
                            std::find(p_T_rows.begin(), p_T_rows.end(), i) - p_T_rows.begin();
                        perm_stride_rows.at(i) = stride_T_rows.at(j);
                        perm_index_rows.at(i) = 1;
                    }
                }

                // Get strides for the input dense tensor, x
                std::vector<IndexType> perm_stride_x(x_N);
                std::vector<IndexType> perm_from_x(x_N);
                std::vector<char> perm_index_x(x_N);
                for (int i = 0; i < x_N; ++i) {
                    const int j = std::find(p_CBT, p_CBT + x_N, i) - p_CBT;
                    if (j < NC) {
                        perm_from_x.at(i) = 0;
                        perm_stride_x.at(i) = stride_C.at(j);
                        perm_index_x.at(i) = 0;
                    } else if (j - NC < NB) {
                        if (p_ABT[NA + j - NC] < s_rows_N) throw std::runtime_error("unsupported");
                        const auto js_cols = p_ABT[NA + j - NC] - s_rows_N;
                        perm_from_x.at(i) =
                            (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                        perm_stride_x.at(i) = stride_s_dim_cols.at(js_cols);
                        perm_index_x.at(i) = 1;
                    } else if (j - NC - NB < NT) {
                        const std::size_t ja = std::find(p_T_rows.begin(), p_T_rows.end(),
                                                         p_ABT[NA + NB + j - NC - NB]) -
                                               p_T_rows.begin();
                        if (ja < p_T_rows.size()) {
                            perm_from_x.at(i) = 0;
                            perm_stride_x.at(i) = stride_T_rows.at(ja);
                            perm_index_x.at(i) = 2;
                        } else {
                            const auto js_cols = p_ABT[NA + NB + j - NC - NB] - s_rows_N;
                            perm_from_x.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_x.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_x.at(i) = 1;
                        }
                    }
                }

                // Get strides for the output dense tensor, y
                std::vector<IndexType> perm_stride_y(y_N);
                std::vector<IndexType> perm_from_y(y_N);
                std::vector<char> perm_index_y(y_N);
                for (int i = 0; i < y_N; ++i) {
                    const int j = std::find(p_ACT, p_ACT + y_N, i) - p_ACT;
                    if (j < NA) {
                        const std::size_t ja =
                            std::find(p_A_rows.begin(), p_A_rows.end(), p_ABT[j]) -
                            p_A_rows.begin();
                        if (ja < p_A_rows.size()) {
                            perm_from_y.at(i) = 0;
                            perm_stride_y.at(i) = stride_A_rows.at(ja);
                            perm_index_y.at(i) = 0;
                        } else {
                            const auto js_cols = p_ABT[j] - s_rows_N;
                            perm_from_y.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_y.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_y.at(i) = 1;
                        }
                    } else if (j - NA < NC) {
                        perm_from_y.at(i) = 0;
                        perm_stride_y.at(i) = stride_C.at(j - NA);
                        perm_index_y.at(i) = 2;
                    } else if (j - NA - NC < NT) {
                        const std::size_t ja = std::find(p_T_rows.begin(), p_T_rows.end(),
                                                         p_ABT[NA + NB + j - NA - NC]) -
                                               p_T_rows.begin();
                        if (ja < p_T_rows.size()) {
                            perm_from_y.at(i) = 0;
                            perm_stride_y.at(i) = stride_T_rows.at(ja);
                            perm_index_y.at(i) = 3;
                        } else {
                            const auto js_cols = p_ABT[NA + NB + j - NA - NC] - s_rows_N;
                            perm_from_y.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_y.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_y.at(i) = 1;
                        }
                    }
                }

                const auto vol_T_rows =
                    NT == 0 ? 1 : volume(p_T_rows.size(), p_T_rows.data(), s_size_rows);
                const auto vol_A_rows =
                    NA == 0 ? 1 : volume(p_A_rows.size(), p_A_rows.data(), s_size_rows);
                const auto vol_C = NC == 0 ? 1 : volume(NC, p_CBT, x_size);

                return {perm_index_rows, perm_stride_rows, perm_index_x, perm_from_x,
                        perm_stride_x,   perm_index_y,     perm_from_y,  perm_stride_y,
                        vol_A_rows,      vol_T_rows,       vol_C};
            }

            /// Contracts a sparse subtensor `S` and a dense subtensor `X` resulting in a dense subtensor `Y`:
            ///   S * X -> Y .
            /// The sparse tensor has `s_rows_N` dense dimensions and `s_cols_N` dimensions and it represented
            /// as a CSR matrix where the dense dimensions are the "rows" and the sparse dimensions are the "columns".
            /// That is, `s_i[i]..s_i[i+1]` are the indices in `s_j` and `s_values` of the nonzero values for the i-th
            /// dense element.
            /// The user should indicate which coordinates to contract by giving permutations of the coordinates of
            /// `S`, `X` and `Y` such that they have the form `[A, B, T]` for `S`, `[C, B, T]` for `X` and `[A, C, T]` for `S`,
            /// where:
            /// - `A` are the common directions of the tensors `S` and `Y`,
            /// - `B` are the common directions of the tensors `S` and `X`,
            /// - `C` are the common directions of the tensors `X` and `Y`,
            /// - `T` are the common directions of the tensors `S`, `X` and `Y`.
            /// \param s_rows_N: number of dense dimensions of the sparse tensor
            /// \param s_dim_rows: number of elements in each of the dense dimensions of the sparse tensor
            /// \param s_cols_N: number of sparse dimensions of the sparse tensor
            /// \param s_dim_cols: number of elements in each of the sparse dimensions of the sparse tensor
            /// \param s_i: `s_i[i]..s_i[i+1]` indices in `s_j` and `s_values` with the nonzero values for the i-th dense element
            /// \param s_j: `s_j[i]` index of the position of the i-th nonzero value
            /// \param s_values: `s_values[i]` value for the i-th nonzero value
            /// \param s_from_rows: first dense coordinate of the sparse tensor to contract
            /// \param s_from_cols: first sparse coordinate of the sparse tensor to contract
            /// \param s_size_rows: number of elements in each dense direction of the sparse tensor to contract
            /// \param s_size_cols: number of elements in each sparse direction of the sparse tensor to contract
            /// \param x_N: number of dimensions of input dense tensor `X`
            /// \param x_dim: number of elements of the input tensor in each direction
            /// \param x_values: pointer to the values of the dense tensor
            /// \param x_from: coordinates of first element of the subtensor of the dense tensor to contract
            /// \param x_size: number of elements in each direction of the subtensor of the dense tensor to contract
            /// \param y_N: number of dimensions of output dense tensor `Y`
            /// \param y_dim: number of elements of the output tensor in each direction
            /// \param y_values: pointer to the values of the output dense tensor
            /// \param y_from: coordinates of first element of the subtensor of the dense output tensor
            /// \param y_size: number of elements in each direction of the subtensor of the dense output tensor
            /// \param p_ABT: permutation of the coordinates of sparse tensor (first all the sparse dimensions, then the dense dimensions)
            ///           for having the form (A, B, T)
            /// \param p_CBT: permutation of the dense input tensor coordinates to have the form (C, B, T)
            /// \param p_ACT: permutation of the dense output tensor coordinates to have the form (A, C, T)

            template <typename T, typename IndexType>
            void sptensor_tensor_product(
                int s_rows_N, const IndexType *SB_RESTRICT s_dim_rows, int s_cols_N,
                const IndexType *SB_RESTRICT s_dim_cols, const IndexType *SB_RESTRICT s_i,
                const IndexType *SB_RESTRICT s_j, const T *SB_RESTRICT s_values,
                const IndexType *SB_RESTRICT s_from_rows, const IndexType *SB_RESTRICT s_from_cols,
                const IndexType *SB_RESTRICT s_size_rows,
                const IndexType *SB_RESTRICT s_size_cols, //
                int x_N, const IndexType *SB_RESTRICT x_dim, const T *SB_RESTRICT x_values,
                const IndexType *SB_RESTRICT x_from,
                const IndexType *SB_RESTRICT x_size, //
                int y_N, const IndexType *SB_RESTRICT y_dim, T *SB_RESTRICT y_values,
                const IndexType *SB_RESTRICT y_from, const IndexType *SB_RESTRICT y_size, //
                const IndexType *SB_RESTRICT p_ABT, const IndexType *SB_RESTRICT p_CBT,
                const IndexType *SB_RESTRICT p_ACT, int NA, int NB, int NC, int NT, Cpu) {

		// Preparation
                std::vector<char> perm_index_rows;
                std::vector<IndexType> perm_stride_rows;
                std::vector<char> perm_index_x;
                std::vector<IndexType> perm_stride_x;
                std::vector<IndexType> perm_from_x;
                std::vector<char> perm_index_y;
                std::vector<IndexType> perm_stride_y;
                std::vector<IndexType> perm_from_y;
                IndexType vol_A_rows, vol_T_rows, vol_C;
                std::tie(perm_index_rows, perm_stride_rows,        //
                         perm_index_x, perm_stride_x, perm_from_x, //
                         perm_index_y, perm_stride_y, perm_from_y, //
                         vol_A_rows, vol_T_rows, vol_C) =
                    sptensor_tensor_product_preparation(
                        s_rows_N, s_dim_rows, s_cols_N, s_dim_cols, s_from_rows, s_from_cols,
                        s_size_rows, s_size_cols, x_N, x_dim, x_from, x_size, y_N, y_dim, y_from,
                        y_size, p_ABT, p_CBT, p_ACT, NA, NB, NC, NT);

                // Deal with trivial cases
                if ((s_rows_N == 0 && s_cols_N == 0) || x_N == 0 || y_N == 0 ||
                    (s_rows_N > 0 && volume(s_rows_N, s_size_rows) == 0) ||
                    (s_cols_N > 0 && volume(s_cols_N, s_size_cols) == 0) ||
                    volume(x_N, x_size) == 0 || volume(y_N, y_size) == 0) {
                    return;
                }

                const auto get_sp_row = [](int sp_N, const IndexType *sp_from,
                                           const IndexType *sp_size, const IndexType *sp_dim,
                                           const char *perm_index, const IndexType *perm_stride,
                                           IndexType Ai, IndexType Ti) {
                    IndexType row = 0;
                    IndexType step = 1;
                    for (int i = 0; i < sp_N; ++i) {
                        IndexType coor_i =
                            ((perm_index[i] == 0 ? Ai : Ti) / perm_stride[i]) % sp_size[i];
                        row += ((coor_i + sp_from[i]) % sp_dim[i]) * step;
                        step *= sp_dim[i];
                    }
                    return row;
                };

                const auto check_sp_col = [](IndexType abs_index, int sp_N,
                                             const IndexType *sp_from, const IndexType *sp_size,
                                             const IndexType *sp_dim) {
                    IndexType step = 1;
                    for (int i = 0; i < sp_N; ++i) {
                        if ((abs_index / step - sp_from[i] + sp_dim[i]) % sp_dim[i] >= sp_size[i])
                            return false;
                        step *= sp_dim[i];
                    }
                    return true;
                };

                const auto get_x_index = [](int x_N, const IndexType *x_from,
                                            const IndexType *x_size, const IndexType *x_dim,
                                            const char *perm_index, const IndexType *perm_from,
                                            const IndexType *perm_stride, IndexType Ci,
                                            IndexType col, IndexType Trowsi) {
                    IndexType index = 0;
                    IndexType step = 1;
                    for (int i = 0; i < x_N; ++i) {
                        IndexType coor_i =
                            ((perm_index[i] == 0 ? Ci : (perm_index[i] == 1 ? col : Trowsi)) /
                                 perm_stride[i] +
                             perm_from[i]) %
                            x_size[i];
                        index += ((coor_i + x_from[i]) % x_dim[i]) * step;
                        step *= x_dim[i];
                    }
                    return index;
                };

                const auto get_y_index = [](int y_N, const IndexType *y_from,
                                            const IndexType *y_size, const IndexType *y_dim,
                                            const char *perm_index, const IndexType *perm_from,
                                            const IndexType *perm_stride, IndexType Arowsi,
                                            IndexType col, IndexType Ci, IndexType Trowsi) {
                    IndexType index = 0;
                    IndexType step = 1;
                    for (int i = 0; i < y_N; ++i) {
                        IndexType coor_i =
                            ((perm_index[i] == 0
                                  ? Arowsi
                                  : (perm_index[i] == 1 ? col
                                                        : (perm_index[i] == 2 ? Ci : Trowsi))) /
                                 perm_stride[i] +
                             perm_from[i]) %
                            y_size[i];
                        index += ((coor_i + y_from[i]) % y_dim[i]) * step;
                        step *= y_dim[i];
                    }
                    return index;
                };

                // Do the products
                // ABT x CBT->ACT
                // (rows, cols) x CBT->ACT
                // rows in AT, cols in ABT
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) collapse(2)
#endif
                for (IndexType Trowsi = 0; Trowsi < vol_T_rows; ++Trowsi) {
                    for (IndexType Arowsi = 0; Arowsi < vol_A_rows; ++Arowsi) {
                        const auto s_row = get_sp_row(s_rows_N, s_from_rows, s_size_rows,
                                                      s_dim_rows, perm_index_rows.data(),
                                                      perm_stride_rows.data(), Arowsi, Trowsi);
                        for (IndexType ji = s_i[s_row], jn = s_i[s_row + 1]; ji < jn; ++ji) {
                            const auto col = s_j[ji];
                            if (!check_sp_col(col, s_cols_N, s_from_cols, s_size_cols, s_dim_cols))
                                continue;
                            const auto s_val = s_values[ji];
                            for (IndexType Ci = 0; Ci < vol_C; ++Ci) {
                                const auto xi = get_x_index(x_N, x_from, x_size, x_dim,
                                                            perm_index_x.data(), perm_from_x.data(),
                                                            perm_stride_x.data(), Ci, col, Trowsi);
                                const auto yi =
                                    get_y_index(y_N, y_from, y_size, y_dim, perm_index_y.data(),
                                                perm_from_y.data(), perm_stride_y.data(), Arowsi,
                                                col, Ci, Trowsi);

                                const auto x_val = x_values[xi];
                                y_values[yi] += s_val * x_val;
                            }
                        }
                    }
                }
            }

#ifdef SUPERBBLAS_USE_GPU
#    ifdef SUPERBBLAS_GENERATE_KERNELS
            template <std::size_t N, typename IndexType> struct BSRExtKernelParams {
                IndexType vol_T_rows;
                IndexType vol_A_rows;
                IndexType vol_C;
                int s_rows_N;
                IndexType s_from_rows[N]; // max 8 row dims – adjust if needed
                IndexType s_size_rows[N];
                IndexType s_dim_rows[N];
                char perm_index_rows[N];
                IndexType perm_stride_rows[N];
                int s_cols_N;
                IndexType s_from_cols[N];
                IndexType s_size_cols[N];
                IndexType s_dim_cols[N];
                int x_N;
                IndexType x_from[N];
                IndexType x_size[N];
                IndexType x_dim[N];
                char perm_index_x[N];
                IndexType perm_from_x[N];
                IndexType perm_stride_x[N];
                int y_N;
                IndexType y_from[N];
                IndexType y_size[N];
                IndexType y_dim[N];
                char perm_index_y[N];
                IndexType perm_from_y[N];
                IndexType perm_stride_y[N];
            };

            template <typename IndexType>
            __device__ __forceinline__ IndexType dev_get_sp_row(int sp_N, const IndexType *sp_from,
                                                                const IndexType *sp_size,
                                                                const IndexType *sp_dim,
                                                                const char *perm_index,
                                                                const IndexType *perm_stride,
                                                                IndexType Ai, IndexType Ti) {
                IndexType row = 0;
                IndexType step = 1;
                for (int i = 0; i < sp_N; ++i) {
                    IndexType coor_i =
                        ((perm_index[i] == 0 ? Ai : Ti) / perm_stride[i]) % sp_size[i];
                    row += ((coor_i + sp_from[i]) % sp_dim[i]) * step;
                    step *= sp_dim[i];
                }
                return row;
            }

            template <typename IndexType>
            __device__ __forceinline__ bool
            dev_check_sp_col(IndexType abs_index, int sp_N, const IndexType *sp_from,
                             const IndexType *sp_size, const IndexType *sp_dim) {
                IndexType step = 1;
                for (int i = 0; i < sp_N; ++i) {
                    if ((abs_index / step - sp_from[i] + sp_dim[i]) % sp_dim[i] >= sp_size[i])
                        return false;
                    step *= sp_dim[i];
                }
                return true;
            }

            template <typename IndexType>
            __device__ __forceinline__ IndexType dev_get_x_index(
                int x_N, const IndexType *x_from, const IndexType *x_size, const IndexType *x_dim,
                const char *perm_index, const IndexType *perm_from, const IndexType *perm_stride,
                IndexType Ci, IndexType col, IndexType Trowsi) {
                IndexType index = 0;
                IndexType step = 1;
                for (int i = 0; i < x_N; ++i) {
                    IndexType coor_i =
                        ((perm_index[i] == 0 ? Ci : (perm_index[i] == 1 ? col : Trowsi)) /
                             perm_stride[i] +
                         perm_from[i]) %
                        x_size[i];
                    index += ((coor_i + x_from[i]) % x_dim[i]) * step;
                    step *= x_dim[i];
                }
                return index;
            }

            template <typename IndexType>
            __device__ __forceinline__ IndexType dev_get_y_index(
                int y_N, const IndexType *y_from, const IndexType *y_size, const IndexType *y_dim,
                const char *perm_index, const IndexType *perm_from, const IndexType *perm_stride,
                IndexType Arowsi, IndexType col, IndexType Ci, IndexType Trowsi) {
                IndexType index = 0;
                IndexType step = 1;
                for (int i = 0; i < y_N; ++i) {
                    IndexType coor_i =
                        ((perm_index[i] == 0
                              ? Arowsi
                              : (perm_index[i] == 1 ? col : (perm_index[i] == 2 ? Ci : Trowsi))) /
                             perm_stride[i] +
                         perm_from[i]) %
                        y_size[i];
                    index += ((coor_i + y_from[i]) % y_dim[i]) * step;
                    step *= y_dim[i];
                }
                return index;
            }

            template <std::size_t N, typename T, typename IndexType>
            __global__ void sptensor_tensor_product_kernel(BSRExtKernelParams<N, IndexType> p,
                                                           const IndexType *SB_RESTRICT s_i,
                                                           const IndexType *SB_RESTRICT s_j,
                                                           const T *SB_RESTRICT s_values,
                                                           const T *SB_RESTRICT x_values,
                                                           T *SB_RESTRICT y_values

            ) {
                // Flatten the 3-D index space (Trowsi, Arowsi, Ci) into a 1-D grid.
                const IndexType tid = (IndexType)blockIdx.x * blockDim.x + threadIdx.x;
                const IndexType total = p.vol_T_rows * p.vol_A_rows * p.vol_C;
                if (tid >= total) return;

                // Recover the three logical indices
                const IndexType Ci = tid % p.vol_C;
                const IndexType Arowsi = (tid / p.vol_C) % p.vol_A_rows;
                const IndexType Trowsi = tid / (p.vol_C * p.vol_A_rows);

                // CSR row for this (Arowsi, Trowsi) pair
                const IndexType s_row =
                    dev_get_sp_row(p.s_rows_N, p.s_from_rows, p.s_size_rows, p.s_dim_rows,
                                   p.perm_index_rows, p.perm_stride_rows, Arowsi, Trowsi);

                // Walk the nonzeros of that CSR row
                for (IndexType ji = s_i[s_row], jn = s_i[s_row + 1]; ji < jn; ++ji) {
                    const IndexType col = s_j[ji];
                    if (!dev_check_sp_col(col, p.s_cols_N, p.s_from_cols, p.s_size_cols,
                                          p.s_dim_cols))
                        continue;

                    const IndexType xi =
                        dev_get_x_index(p.x_N, p.x_from, p.x_size, p.x_dim, p.perm_index_x,
                                        p.perm_from_x, p.perm_stride_x, Ci, col, Trowsi);

                    const IndexType yi =
                        dev_get_y_index(p.y_N, p.y_from, p.y_size, p.y_dim, p.perm_index_y,
                                        p.perm_from_y, p.perm_stride_y, Arowsi, col, Ci, Trowsi);

                    y_values[yi] += s_values[ji] * x_values[xi];
                }
            }
#    endif // SUPERBBLAS_GENERATE_KERNELS

            template <std::size_t N, typename T>
            DECL_SPTENSOR_TENSOR_PRODUCT_GPU_N_T_IDX(void sptensor_tensor_product(
                int s_rows_N, const IndexType *s_dim_rows, int s_cols_N,
                const IndexType *s_dim_cols,
                const IndexType *s_i, // device
                const IndexType *s_j, // device
                const T *s_values,    // device
                const IndexType *s_from_rows, const IndexType *s_from_cols,
                const IndexType *s_size_rows, const IndexType *s_size_cols, int x_N,
                const IndexType *x_dim,
                const T *x_values, // device
                const IndexType *x_from, const IndexType *x_size, int y_N, const IndexType *y_dim,
                T *y_values, // device
                const IndexType *y_from, const IndexType *y_size, const IndexType *p_ABT,
                const IndexType *p_CBT, const IndexType *p_ACT, int NA, int NB, int NC, int NT,
                Gpu xpu))
            IMPL({
                if ((std::size_t)std::max({s_rows_N, s_cols_N, x_N, y_N}) > N)
                    throw std::runtime_error("invalid input");

                for (const auto &[Ni, size, dim] :
                     std::array<std::tuple<int, const IndexType *, const IndexType *>, 4>{
                         {{s_rows_N, s_size_rows, s_dim_rows},
                          {s_cols_N, s_size_cols, s_dim_cols},
                          {x_N, x_size, x_dim},
                          {y_N, y_size, y_dim}}}) {
                    for (int i = 0; i < Ni; ++i)
                        if (size[i] > dim[i]) throw std::runtime_error("invalid input");
                }
                if (NA + NB + NT != s_rows_N + s_cols_N || NC + NB + NT != x_N ||
                    NA + NC + NT != y_N)
                    throw std::runtime_error("invalid input");

                if ((s_rows_N == 0 && s_cols_N == 0) || x_N == 0 || y_N == 0 ||
                    (s_rows_N > 0 && volume(s_rows_N, s_size_rows) == 0) ||
                    (s_cols_N > 0 && volume(s_cols_N, s_size_cols) == 0) ||
                    volume(x_N, x_size) == 0 || volume(y_N, y_size) == 0)
                    return;

                // Find the volume for each part
                std::vector<IndexType> size_B(NB);
                copy_coor(NB, x_size, p_CBT + NC, size_B.data());
                std::vector<IndexType> size_C(NC);
                copy_coor(NC, x_size, p_CBT, size_C.data());
                {
                    std::vector<IndexType> size_A(NA);
                    copy_coor(NA, y_size, p_ACT, size_A.data());
                    std::vector<IndexType> size_T(NT);
                    copy_coor(NT, x_size, p_CBT + NC + NB, size_T.data());
                    std::vector<IndexType> s_size(s_rows_N + s_cols_N);
                    copy_coor(s_rows_N, s_size_rows, s_size.data());
                    copy_coor(s_cols_N, s_size_cols, s_size.data() + s_rows_N);
                    if (!is_compatible(NA, p_ABT, s_size.data(), size_A.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NB, p_ABT + NA, s_size.data(), size_B.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_ABT + NA + NB, s_size.data(), size_T.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_CBT + NC + NB, x_size, size_T.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NC, p_ACT + NA, y_size, size_C.data()))
                        throw std::runtime_error("invalid input");
                    if (!is_compatible(NT, p_ACT + NA + NC, y_size, size_T.data()))
                        throw std::runtime_error("invalid input");
                }

                std::vector<IndexType> p_A_rows, p_A_cols;
                for (int i = 0; i < NA; ++i) {
                    if (p_ABT[i] < s_rows_N)
                        p_A_rows.push_back(p_ABT[i]);
                    else
                        p_A_cols.push_back(p_ABT[i] - s_rows_N);
                }
                for (int i = 0; i < NB; ++i) {
                    if (p_ABT[NA + i] < s_rows_N) {
                        // FIXME: support "a" row coors in B
                        throw std::runtime_error("unsupported");
                    }
                }
                std::vector<IndexType> p_T_rows, p_T_cols;
                for (int i = 0; i < NT; ++i) {
                    if (p_ABT[NA + NB + i] < s_rows_N)
                        p_T_rows.push_back(p_ABT[NA + NB + i]);
                    else
                        p_T_cols.push_back(p_ABT[NA + NB + i] - s_rows_N);
                }

                std::vector<IndexType> stride_A_rows(p_A_rows.size());
                copy_stride(p_A_rows.size(), s_size_rows, p_A_rows.data(), stride_A_rows.data());
                std::vector<IndexType> stride_s_dim_cols(s_cols_N);
                copy_stride(s_cols_N, s_dim_cols, stride_s_dim_cols.data());
                std::vector<IndexType> stride_C(NC);
                copy_stride(NC, size_C.data(), stride_C.data());
                std::vector<IndexType> stride_T_rows(p_T_rows.size());
                copy_stride(p_T_rows.size(), s_size_rows, p_T_rows.data(), stride_T_rows.data());
                std::vector<IndexType> stride_s_rows(s_rows_N);
                copy_stride(s_rows_N, s_size_rows, stride_s_rows.data());

                // perm_stride_rows / perm_index_rows
                std::vector<IndexType> perm_stride_rows(s_rows_N);
                std::vector<char> perm_index_rows(s_rows_N);
                for (int i = 0; i < s_rows_N; ++i) {
                    const std::size_t j =
                        std::find(p_A_rows.begin(), p_A_rows.end(), i) - p_A_rows.begin();
                    if (j < p_A_rows.size()) {
                        perm_stride_rows.at(i) = stride_A_rows.at(j);
                        perm_index_rows.at(i) = 0;
                    } else {
                        const auto j =
                            std::find(p_T_rows.begin(), p_T_rows.end(), i) - p_T_rows.begin();
                        perm_stride_rows.at(i) = stride_T_rows.at(j);
                        perm_index_rows.at(i) = 1;
                    }
                }

                // perm for x
                std::vector<IndexType> perm_stride_x(x_N), perm_from_x(x_N);
                std::vector<char> perm_index_x(x_N);
                for (int i = 0; i < x_N; ++i) {
                    const int j = std::find(p_CBT, p_CBT + x_N, i) - p_CBT;
                    if (j < NC) {
                        perm_from_x.at(i) = 0;
                        perm_stride_x.at(i) = stride_C.at(j);
                        perm_index_x.at(i) = 0;
                    } else if (j - NC < NB) {
                        if (p_ABT[NA + j - NC] < s_rows_N) throw std::runtime_error("unsupported");
                        const auto js_cols = p_ABT[NA + j - NC] - s_rows_N;
                        perm_from_x.at(i) =
                            (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                        perm_stride_x.at(i) = stride_s_dim_cols.at(js_cols);
                        perm_index_x.at(i) = 1;
                    } else if (j - NC - NB < NT) {
                        const std::size_t ja = std::find(p_T_rows.begin(), p_T_rows.end(),
                                                         p_ABT[NA + NB + j - NC - NB]) -
                                               p_T_rows.begin();
                        if (ja < p_T_rows.size()) {
                            perm_from_x.at(i) = 0;
                            perm_stride_x.at(i) = stride_T_rows.at(ja);
                            perm_index_x.at(i) = 2;
                        } else {
                            const auto js_cols = p_ABT[NA + NB + j - NC - NB] - s_rows_N;
                            perm_from_x.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_x.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_x.at(i) = 1;
                        }
                    }
                }

                // perm for y
                std::vector<IndexType> perm_stride_y(y_N), perm_from_y(y_N);
                std::vector<char> perm_index_y(y_N);
                for (int i = 0; i < y_N; ++i) {
                    const int j = std::find(p_ACT, p_ACT + y_N, i) - p_ACT;
                    if (j < NA) {
                        const std::size_t ja =
                            std::find(p_A_rows.begin(), p_A_rows.end(), p_ABT[j]) -
                            p_A_rows.begin();
                        if (ja < p_A_rows.size()) {
                            perm_from_y.at(i) = 0;
                            perm_stride_y.at(i) = stride_A_rows.at(ja);
                            perm_index_y.at(i) = 0;
                        } else {
                            const auto js_cols = p_ABT[j] - s_rows_N;
                            perm_from_y.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_y.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_y.at(i) = 1;
                        }
                    } else if (j - NA < NC) {
                        perm_from_y.at(i) = 0;
                        perm_stride_y.at(i) = stride_C.at(j - NA);
                        perm_index_y.at(i) = 2;
                    } else if (j - NA - NC < NT) {
                        const std::size_t ja = std::find(p_T_rows.begin(), p_T_rows.end(),
                                                         p_ABT[NA + NB + j - NA - NC]) -
                                               p_T_rows.begin();
                        if (ja < p_T_rows.size()) {
                            perm_from_y.at(i) = 0;
                            perm_stride_y.at(i) = stride_T_rows.at(ja);
                            perm_index_y.at(i) = 3;
                        } else {
                            const auto js_cols = p_ABT[NA + NB + j - NA - NC] - s_rows_N;
                            perm_from_y.at(i) =
                                (s_dim_cols[js_cols] - s_from_cols[js_cols]) % s_dim_cols[js_cols];
                            perm_stride_y.at(i) = stride_s_dim_cols.at(js_cols);
                            perm_index_y.at(i) = 1;
                        }
                    }
                }

                const IndexType vol_T_rows =
                    NT == 0 ? 1 : volume(p_T_rows.size(), p_T_rows.data(), s_size_rows);
                const IndexType vol_A_rows =
                    NA == 0 ? 1 : volume(p_A_rows.size(), p_A_rows.data(), s_size_rows);
                const IndexType vol_C_val = NC == 0 ? 1 : volume(NC, p_CBT, x_size);

                BSRExtKernelParams<N, IndexType> kp{};
                kp.vol_T_rows = vol_T_rows;
                kp.vol_A_rows = vol_A_rows;
                kp.vol_C = vol_C_val;
                kp.s_rows_N = s_rows_N;
                kp.s_cols_N = s_cols_N;
                kp.x_N = x_N;
                kp.y_N = y_N;

                // Copy small arrays into the fixed-size fields (max 8 dims each)
                std::copy_n(s_from_rows, s_rows_N, kp.s_from_rows);
                std::copy_n(s_size_rows, s_rows_N, kp.s_size_rows);
                std::copy_n(s_dim_rows, s_rows_N, kp.s_dim_rows);
                std::copy_n(perm_index_rows.data(), s_rows_N, kp.perm_index_rows);
                std::copy_n(perm_stride_rows.data(), s_rows_N, kp.perm_stride_rows);
                std::copy_n(s_from_cols, s_cols_N, kp.s_from_cols);
                std::copy_n(s_size_cols, s_cols_N, kp.s_size_cols);
                std::copy_n(s_dim_cols, s_cols_N, kp.s_dim_cols);
                std::copy_n(x_from, x_N, kp.x_from);
                std::copy_n(x_size, x_N, kp.x_size);
                std::copy_n(x_dim, x_N, kp.x_dim);
                std::copy_n(perm_index_x.data(), x_N, kp.perm_index_x);
                std::copy_n(perm_from_x.data(), x_N, kp.perm_from_x);
                std::copy_n(perm_stride_x.data(), x_N, kp.perm_stride_x);
                std::copy_n(y_from, y_N, kp.y_from);
                std::copy_n(y_size, y_N, kp.y_size);
                std::copy_n(y_dim, y_N, kp.y_dim);
                std::copy_n(perm_index_y.data(), y_N, kp.perm_index_y);
                std::copy_n(perm_from_y.data(), y_N, kp.perm_from_y);
                std::copy_n(perm_stride_y.data(), y_N, kp.perm_stride_y);

                // Launch kernel
                const IndexType total_threads = vol_T_rows * vol_A_rows * vol_C_val;
                const int block_size = 256;
                const int grid_size = (int)((total_threads + block_size - 1) / block_size);

                using Tc = typename superbblas::detail::ccomplex<T>::type;
                setDevice(xpu);
                sptensor_tensor_product_kernel<N, Tc, IndexType>
                    <<<grid_size, block_size, 0, getStream(xpu)>>>(
                        kp, s_i, s_j, (const Tc *)s_values, (const Tc *)x_values, (Tc *)y_values);
                gpuCheck(SUPERBBLAS_GPU_SYMBOL(GetLastError)());
            })
#endif // SUPERBBLAS_USE_GPU
        }

        template <std::size_t Ni, std::size_t Nd, std::size_t Nx, std::size_t Ny, typename T,
                  typename XPU>
        void sptensor_tensor_product(const Coor<Ni> &dimi, const Order<Ni> &oim, std::size_t ni,
                                     const Coor<Nd> &dimd, const Order<Nd> &odm, std::size_t nd,
                                     const vector<IndexType, XPU> &vi,
                                     const vector<IndexType, XPU> &vj, const vector<T, XPU> &va,
                                     const Coor<Nx> &dimx, const Order<Nx> &ox, std::size_t nx,
                                     const vector<T, XPU> &vx, const Coor<Ny> &dimy,
                                     const Order<Ny> &oy, std::size_t ny, vector<T, XPU> &vy) {
            const T *x = vx.data();
            T *y = vy.data();
            const T beta{0};
            xscal(volume(dimy), beta, y, 1, vy.ctx());

            // Find the permutations
            Coor<Ni + Nd> p_ABT;
            Coor<Nx> p_CBT;
            Coor<Ny> p_ACT;
            Order<Ni + Nd> os;
            std::copy_n(oim.begin(), ni, os.begin());
            std::copy_n(odm.begin(), nd, os.begin() + ni);
            const auto ns = ni + nd;
            int na = 0, nb = 0, nc = 0, nt = 0;
            auto is_in = [](auto o, std::size_t n, char c) {
                return std::find(o.begin(), o.begin() + n, c) != o.begin() + n;
            };
            auto index_in = [](auto o, std::size_t n, char c) {
                return std::find(o.begin(), o.begin() + n, c) - o.begin();
            };
            for (std::size_t i = 0; i < ny; ++i) {
                const auto c = oy.at(i);
                if (is_in(os, ns, c) && !is_in(ox, nx, c)) {
                    p_ACT[na] = i;
                    p_ABT[na] = index_in(os, ns, c);
                    na++;
                }
            }
            for (std::size_t i = 0; i < nx; ++i) {
                const auto c = ox.at(i);
                if (!is_in(os, ns, c) && is_in(oy, ny, c)) {
                    p_ACT[na + nc] = index_in(oy, ny, c);
                    p_CBT[nc] = i;
                    nc++;
                }
            }
            for (std::size_t i = 0; i < nx; ++i) {
                const auto c = ox.at(i);
                if (is_in(os, ns, c) && !is_in(oy, ny, c)) {
                    p_ABT[na + nb] = index_in(os, ns, c);
                    p_CBT[nc + nb] = i;
                    nb++;
                }
            }
            for (std::size_t i = 0; i < nx; ++i) {
                const auto c = ox.at(i);
                if (is_in(os, ns, c) && is_in(oy, ny, c)) {
                    p_ABT[na + nb + nt] = index_in(os, ns, c);
                    p_CBT[nc + nb + nt] = i;
                    p_ACT[na + nc + nt] = is_in(oy, ny, c);
                    nt++;
                }
            }

            Coor<Ni> s_from_rows{{}};
            Coor<Nd> s_from_cols{{}};
            Coor<Nx> x_from{{}};
            Coor<Ny> y_from{{}};
            if constexpr (std::is_same<XPU, Cpu>::value) {
                aux_sptensor_tensor_product::sptensor_tensor_product(
                    ni, dimi.data(), nd, dimd.data(), vi.data(), vj.data(), va.data(),
                    s_from_rows.data(), s_from_cols.data(), dimi.data(), dimd.data(), nx,
                    dimx.data(), x, x_from.data(), dimx.data(), ny, dimy.data(), y, y_from.data(),
                    dimy.data(), p_ABT.data(), p_CBT.data(), p_ACT.data(), na, nb, nc, nt,
                    va.ctx());
            } else {
                constexpr auto N = multiple_of(std::max(std::max(Ni, Nd), std::max(Nx, Ny)), 4ul);
                aux_sptensor_tensor_product::sptensor_tensor_product<N>(
                    ni, dimi.data(), nd, dimd.data(), vi.data(), vj.data(), va.data(),
                    s_from_rows.data(), s_from_cols.data(), dimi.data(), dimd.data(), nx,
                    dimx.data(), x, x_from.data(), dimx.data(), ny, dimy.data(), y, y_from.data(),
                    dimy.data(), p_ABT.data(), p_CBT.data(), p_ACT.data(), na, nb, nc, nt,
                    va.ctx());
            }
        }
    }
}

#endif // __SUPERBBLAS_TENDUCKS__
