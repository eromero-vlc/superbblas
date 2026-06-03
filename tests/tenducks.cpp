#include "superbblas.h"
#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <typename IndexType>
void index_to_coor(IndexType index, int N, const IndexType *dim, IndexType *coor) {
    IndexType step = 1;
    for (int i = 0; i < N; ++i) {
        coor[i] = (index / step) % dim[i];
        step *= dim[i];
    }
}

template <typename IndexType>
IndexType coor_to_index(int N, const IndexType *dim, const IndexType *coor) {
    IndexType index = 0;
    IndexType step = 1;
    for (int i = 0; i < N; ++i) {
        index += (coor[i] % dim[i]) * step;
        step *= dim[i];
    }
    return index;
}

template <typename T, typename IndexType>
void dense_tensor_product(T alpha, const T *a_values, const T *x_values, T beta, T *y_values,
                          IndexType vol_A, IndexType vol_B, IndexType vol_C, IndexType vol_T) {

    // Do the products
    // ABT x CBT->ACT
    for (IndexType ti = 0; ti < vol_T; ++ti) {
        for (IndexType ci = 0; ci < vol_C; ++ci) {
            for (IndexType ai = 0; ai < vol_A; ++ai) {
                const auto yi = ai + vol_A * ci + vol_A * vol_C * ti;
                auto acc = (std::norm(beta) == 0 ? 0 : beta * y_values[yi]);
                for (IndexType bi = 0; bi < vol_B; ++bi) {
                    const auto aai = ai + vol_A * bi + vol_A * vol_B * ti;
                    const auto xi = ci + vol_C * bi + vol_C * vol_B * ti;
                    acc += alpha * a_values[aai] * x_values[xi];
                }
                y_values[yi] = acc;
            }
        }
    }
}


template <typename T, typename XPU> void test(const XPU xpu) {
    using namespace aux_sptensor_tensor_product;

    // Make A, B and T to have two coordinates and C only one
    std::vector<IndexType> A_c = {2, 3}, B_c = {4}, C_c = {5}, T_c = {6, 7};

    for (int A_is_singleton = 0; A_is_singleton < 2; A_is_singleton++) {
        for (int B_is_singleton = 0; B_is_singleton < 2; B_is_singleton++) {
            for (int C_is_singleton = 0; C_is_singleton < 2; C_is_singleton++) {
                for (int T_is_singleton = 0; T_is_singleton < 2; T_is_singleton++) {
                    if ((A_is_singleton && B_is_singleton && T_is_singleton) ||
                        (C_is_singleton && B_is_singleton && T_is_singleton) ||
                        (A_is_singleton && C_is_singleton && T_is_singleton))
                        continue;
                    const auto NA = (A_is_singleton ? 0 : (int)A_c.size());
                    const auto NB = (B_is_singleton ? 0 : (int)B_c.size());
                    const auto NC = (C_is_singleton ? 0 : (int)C_c.size());
                    const auto NT = (T_is_singleton ? 0 : (int)T_c.size());
                    const auto vol_A = (NA == 0 ? 1 : volume(NA, A_c.data()));
                    const auto vol_B = (NB == 0 ? 1 : volume(NB, B_c.data()));
                    const auto vol_C = (NC == 0 ? 1 : volume(NC, C_c.data()));
                    const auto vol_T = (NT == 0 ? 1 : volume(NT, T_c.data()));
                    for (int A_rows_cols = 0; A_rows_cols < (A_is_singleton ? 1 : 2);
                         ++A_rows_cols) {
                        for (int T_rows_cols = 0; T_rows_cols < (T_is_singleton ? 1 : 2);
                             ++T_rows_cols) {
                            // Find coordinates for the sparse matrix
                            IndexType a_rows_N = 0, a_cols_N = 0;
                            std::array<std::array<std::vector<IndexType>, 3>, 2> p_ABT{
                                std::array<std::vector<IndexType>, 3>{{{}, {}, {}}},
                                std::array<std::vector<IndexType>, 3>{{{}, {}, {}}}};
                            if (!A_is_singleton) {
                                if (A_rows_cols == 0) {
                                    p_ABT.at(0).at(0).push_back(a_rows_N++);
                                    p_ABT.at(0).at(0).push_back(a_rows_N++);
                                } else if (A_rows_cols == 1) {
                                    p_ABT.at(0).at(0).push_back(a_rows_N++);
                                    p_ABT.at(1).at(0).push_back(a_cols_N++);
                                } else {
                                    p_ABT.at(1).at(0).push_back(a_cols_N++);
                                    p_ABT.at(1).at(0).push_back(a_cols_N++);
                                }
                            }
                            if (!B_is_singleton) p_ABT.at(1).at(1).push_back(a_cols_N++);
                            if (!T_is_singleton) {
                                if (T_rows_cols == 0) {
                                    p_ABT.at(0).at(2).push_back(a_rows_N++);
                                    p_ABT.at(0).at(2).push_back(a_rows_N++);
                                } else if (T_rows_cols == 1) {
                                    p_ABT.at(0).at(2).push_back(a_rows_N++);
                                    p_ABT.at(1).at(2).push_back(a_cols_N++);
                                } else {
                                    p_ABT.at(1).at(2).push_back(a_cols_N++);
                                    p_ABT.at(1).at(2).push_back(a_cols_N++);
                                }
                            }
                            std::vector<IndexType> a_dim(a_rows_N + a_cols_N);
                            std::vector<IndexType> p_ABT_c(a_rows_N + a_cols_N);
                            {
                                std::array<int, 3> n_abt{0, 0, 0};
                                const std::array<std::vector<IndexType>, 3> abt_c{A_c, B_c, T_c};
                                for (int rc = 0; rc < 2; ++rc) {
                                    for (int abt = 0; abt < 3; ++abt) {
                                        for (auto k : p_ABT.at(rc).at(abt)) {
                                            a_dim.at(rc == 0 ? k : a_rows_N + k) =
                                                abt_c.at(abt).at(n_abt.at(abt)++);
                                        }
                                    }
                                }
                                for (int abt = 0, i = 0; abt < 3; ++abt) {
                                    for (int rc = 0; rc < 2; ++rc) {
                                        for (auto k : p_ABT.at(rc).at(abt)) {
                                            p_ABT_c.at(i++) = (rc == 0 ? k : a_rows_N + k);
                                        }
                                    }
                                }
                            }

                            // Find the coordinates for the dense matrix x
                            std::vector<IndexType> x_dim;
                            if (!C_is_singleton) {
                                for (const auto &k : C_c) x_dim.push_back(k);
                            }
                            if (!B_is_singleton) {
                                for (const auto &k : B_c) x_dim.push_back(k);
                            }
                            if (!T_is_singleton) {
                                for (const auto &k : T_c) x_dim.push_back(k);
                            }
                            const int x_N = x_dim.size();
                            std::vector<IndexType> p_CBT(x_N);
                            std::iota(p_CBT.begin(), p_CBT.end(), 0);

                            // Find the coordinates for the dense matrix y
                            std::vector<IndexType> y_dim;
                            if (!A_is_singleton) {
                                for (const auto &k : A_c) y_dim.push_back(k);
                            }
                            if (!C_is_singleton) {
                                for (const auto &k : C_c) y_dim.push_back(k);
                            }
                            if (!T_is_singleton) {
                                for (const auto &k : T_c) y_dim.push_back(k);
                            }
                            const int y_N = y_dim.size();
                            std::vector<IndexType> p_ACT(y_N);
                            std::iota(p_ACT.begin(), p_ACT.end(), 0);

                            // Create the nonzeros of the sparse matrix
                            const auto a_num_rows = volume(a_rows_N, a_dim.data());
                            const auto a_num_cols = volume(a_cols_N, a_dim.data() + a_rows_N);
                            const auto a_nnz_per_row =
                                std::max(1, (int)(.3 * volume(a_cols_N, a_dim.data() + a_rows_N)));
                            std::vector<IndexType> a_i(a_num_rows + 1);
                            std::vector<IndexType> a_j(a_num_rows * a_nnz_per_row, -1);
                            std::vector<T> a_values(a_num_rows * a_nnz_per_row);
                            const auto a_dense_vol = volume(a_rows_N + a_cols_N, a_dim.data());
                            std::vector<T> a_dense_values(a_dense_vol);
                            std::vector<IndexType> a_c(a_rows_N + a_cols_N);
                            std::vector<IndexType> a_dense_c(a_rows_N + a_cols_N);
                            std::vector<IndexType> a_dense_dim(a_rows_N + a_cols_N);
                            copy_coor(a_rows_N + a_cols_N, a_dim.data(), p_ABT_c.data(),
                                      a_dense_dim.data());
                            for (int r = 0, nnz = 0; r < a_num_rows; ++r) {
                                index_to_coor(r, a_rows_N, a_dim.data(), a_c.data());
                                for (int ji = 0; ji < a_nnz_per_row; ++ji) {
                                    const auto col = nnz % a_num_cols;
                                    const auto val = nnz + 1;
                                    a_j.at(nnz) = col;
                                    a_values.at(nnz) = val;
                                    nnz++;
                                    index_to_coor(col, a_cols_N, a_dim.data() + a_rows_N,
                                                  a_c.data() + a_rows_N);
                                    copy_coor(a_rows_N + a_cols_N, a_c.data(), p_ABT_c.data(),
                                              a_dense_c.data());
                                    const auto a_densei = coor_to_index(
                                        a_rows_N + a_cols_N, a_dense_dim.data(), a_dense_c.data());
                                    a_dense_values.at(a_densei) = val;
                                }
                                a_i.at(r + 1) = nnz;
                            }

                            // Create the values for x and y
                            auto vol_x = volume(x_N, x_dim.data());
                            std::vector<T> x_values(vol_x);
                            std::iota(x_values.begin(), x_values.end(), 0);
                            auto vol_y = volume(y_N, y_dim.data());
                            std::vector<T> y_values(vol_y);

                            // Move all values to gpu if needed
                            const auto to_xpu = [&](auto &v) {
                                return makeSure(
                                    superbblas::detail::vector<
                                        typename std::remove_const<typename std::remove_reference<
                                            decltype(v)>::type>::type::value_type,
                                        Cpu>(v.size(), v.data(), Cpu{}),
                                    xpu);
                            };
                            const auto ai_xpu = to_xpu(a_i);
                            const auto aj_xpu = to_xpu(a_j);
                            const auto avalues_xpu = to_xpu(a_values);
                            const auto x_xpu = to_xpu(x_values);
                            auto y_xpu = to_xpu(y_values);

                            // Do the product
                            const auto a_from = std::vector<IndexType>(a_dim.size());
                            const auto a_size = a_dim;
                            const auto x_from = std::vector<IndexType>(x_dim.size());
                            const auto x_size = x_dim;
                            const auto y_from = std::vector<IndexType>(y_dim.size());
                            const auto y_size = y_dim;
                            const auto alpha = T{1};
                            const auto beta = T{0};
                            if constexpr (std::is_same<XPU, Cpu>::value) {
                                aux_sptensor_tensor_product::sptensor_tensor_product(
                                    a_rows_N, a_dim.data(), a_cols_N, a_dim.data() + a_rows_N,
                                    ai_xpu.data(), aj_xpu.data(), avalues_xpu.data(), a_from.data(),
                                    a_from.data() + a_rows_N, a_size.data(),
                                    a_size.data() + a_rows_N, x_N, x_dim.data(), x_xpu.data(),
                                    x_from.data(), x_size.data(), y_N, y_dim.data(), y_xpu.data(),
                                    y_from.data(), y_size.data(), p_ABT_c.data(), p_CBT.data(),
                                    p_ACT.data(), NA, NB, NC, NT, xpu);
                            } else {
                                aux_sptensor_tensor_product::sptensor_tensor_product<8>(
                                    a_rows_N, a_dim.data(), a_cols_N, a_dim.data() + a_rows_N,
                                    ai_xpu.data(), aj_xpu.data(), avalues_xpu.data(), a_from.data(),
                                    a_from.data() + a_rows_N, a_size.data(),
                                    a_size.data() + a_rows_N, x_N, x_dim.data(), x_xpu.data(),
                                    x_from.data(), x_size.data(), y_N, y_dim.data(), y_xpu.data(),
                                    y_from.data(), y_size.data(), p_ABT_c.data(), p_CBT.data(),
                                    p_ACT.data(), NA, NB, NC, NT, xpu);
                            }
                            copy_n(y_xpu.data(), xpu, y_xpu.size(), y_values.data(), Cpu{});

                            std::vector<T> y_true_values(vol_y);
                            dense_tensor_product(alpha, a_dense_values.data(), x_values.data(),
                                                 beta, y_true_values.data(), vol_A, vol_B, vol_C,
                                                 vol_T);

                            double frob_diff = 0, frob_y_true = 0;
                            for (IndexType i = 0; i < vol_y; ++i) {
                                frob_y_true += std::norm(y_true_values.at(i));
                                frob_diff += std::norm(y_values.at(i) - y_true_values.at(i));
                            }
                            if (frob_diff > frob_y_true * 1e-5)
                                throw std::runtime_error("test failed");
                        }
                    }
                }
            }
        }
    }
}

int main(int, char **) {
    {
        Context ctx = createCpuContext();
        test<float>(ctx.toCpu(0));
        test<std::complex<double>>(ctx.toCpu(0));
    }
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(0);
        test<float>(ctx.toGpu(0));
        test<std::complex<double>>(ctx.toGpu(0));
    }
#endif

    return 0;
}
