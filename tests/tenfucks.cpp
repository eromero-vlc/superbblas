#include "superbblas.h"
#include "superbblas/tenfucks_cpu.h"
#include <algorithm>
#include <ccomplex>
#include <chrono>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

using namespace superbblas;
using namespace superbblas::detail;

template <typename SCALAR> void test() {
    const std::unordered_map<std::type_index, std::string> type_to_string{
        {std::type_index(typeid(std::complex<float>)), "complex float"},
        {std::type_index(typeid(std::complex<double>)), "complex double"}};
    std::cout << "Testing " << type_to_string.at(std::type_index(typeid(SCALAR)))
#ifndef SUPERBBLAS_LIB
              << " with a specific implementation for a vectorization of "
              << superbblas::detail_xp::get_native_size<SCALAR>::size << " parts"
#endif
              << std::endl;
    std::vector<SCALAR> a(9 * 2);
    for (size_t i = 0; i < a.size(); ++i) a[i] = {1.f * i, .5f * i};

    for (int n = 1; n < 10; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        std::vector<SCALAR> b(3 * 4 * n * 2);
        for (size_t i = 0; i < b.size(); ++i) b[i] = {1.f * i, 1.f * i};

        {
            std::vector<SCALAR> c(3 * n);

            xgemm_alt_alpha1_beta1('n', 'n', 3, n, 3, a.data(), 3, b.data(), 3, c.data(), 3, Cpu{});

            std::vector<SCALAR> c0(3 * n);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < 3; ++k) c0[i + 3 * j] += a[i + 3 * k] * b[k + 3 * j];

            double r = 0;
            for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
        {
            std::vector<SCALAR> c(3 * n);

            xgemm_alt_alpha1_beta1('n', 'n', n, 3, 3, b.data(), n, a.data(), 3, c.data(), n, Cpu{});

            std::vector<SCALAR> c0(3 * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 3; ++k) c0[i + n * j] += b[i + n * k] * a[k + 3 * j];

            double r = 0;
            for (int i = 0; i < 3 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
        {
            std::vector<SCALAR> c(3 * 4 * n);
            std::vector<int> bj({0, 3 * 4});
            const int bjprod = n;
            std::vector<int> perm({0, 1, 2, 3, 3, 2, 1, 0});
            std::vector<int> perm_sign({1, -1, 1, 1, -1, 1, -1, 1});
            xgemm_alt_alpha1_beta0_perm(2, 3, 4 * n, 3, a.data(), 1, 3, b.data(), bj.data(), bjprod,
                                        1, 3, 4, perm.data(), perm_sign.data(), c.data(), 1, 3,
                                        Cpu{});

            std::vector<SCALAR> c0(3 * 4 * n);
            for (int m = 0; m < 2; ++m)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4 * n; j++)
                        for (int k = 0; k < 3; ++k)
                            c0[i + 3 * j] +=
                                a[9 * m + i + 3 * k] *
                                b[bj[m] * bjprod + k + 3 * (j / 4 * 4 + perm[4 * m + j % 4])] *
                                (SCALAR)perm_sign[4 * m + j % 4];

            double r = 0;
            for (int i = 0; i < 3 * 4 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
        {
            std::vector<SCALAR> c(3 * 4 * n);
            std::vector<int> bj({0, 3 * 4});
            const int bjprod = n;
            std::vector<int> perm({0, 1, 2, 3, 3, 2, 1, 0});
            std::vector<int> perm_sign({1, -1, 1, 1, -1, 1, -1, 1});
            xgemm_alt_alpha1_beta0_perm(2, 3, 4 * n, 3, a.data(), 3, 1, b.data(), bj.data(), bjprod,
                                        1, 3, 4, perm.data(), perm_sign.data(), c.data(), 1, 3,
                                        Cpu{});

            std::vector<SCALAR> c0(3 * 4 * n);
            for (int m = 0; m < 2; ++m)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4 * n; j++)
                        for (int k = 0; k < 3; ++k)
                            c0[i + 3 * j] +=
                                a[9 * m + 3 * i + k] *
                                b[bj[m] * bjprod + k + 3 * (j / 4 * 4 + perm[4 * m + j % 4])] *
                                (SCALAR)perm_sign[4 * m + j % 4];

            double r = 0;
            for (int i = 0; i < 3 * 4 * n; ++i) r += std::norm(c[i] - c0[i]);
            std::cout << "Error: " << std::sqrt(r) << std::endl;
        }
    }
}

int main(int, char **) {
#ifdef SUPERBBLAS_USE_FLOAT16
    test<std::complex<_Float16>>();
#endif
    test<std::complex<float>>();
    test<std::complex<double>>();
    return 0;
}
