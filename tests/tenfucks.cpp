#include "superbblas.h"
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
    std::vector<SCALAR> a(9);
    for (size_t i = 0; i < a.size(); ++i) a[i] = {1.f * i, .5f * i};

    for (int n = 1; n < 10; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        std::vector<SCALAR> b(3 * n);
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
    }
}

template <typename SCALAR> void test_gpu(const Gpu &xpu) {
    const std::unordered_map<std::type_index, std::string> type_to_string{
        {std::type_index(typeid(std::complex<float>)), "complex float"},
        {std::type_index(typeid(std::complex<double>)), "complex double"}};

    const bool specialized_kernel_support =
        superbblas::detail::available_bsr_kron_3x3_4x4perm<SCALAR>(xpu);
    if (!specialized_kernel_support) {
        std::cout << "Not doing testing for " << type_to_string.at(std::type_index(typeid(SCALAR)))
                  << std::endl;
        return;
    }
    std::cout << "Testing " << type_to_string.at(std::type_index(typeid(SCALAR)))
              << " with a specialized kernel support" << std::endl;

    vector<SCALAR, Cpu> a_cpu(9, Cpu{});
    for (size_t i = 0; i < a_cpu.size(); ++i) a_cpu[i] = {i * 1.f, .5f * i};
    auto a = makeSure(a_cpu, xpu);
    vector<SCALAR, Cpu> k_cpu(4, Cpu{});
    for (size_t i = 0; i < k_cpu.size(); ++i) k_cpu[i] = {.5f * (i + 1), 1.f};
    auto k = makeSure(k_cpu, xpu);
    vector<int, Cpu> k_perm_cpu(4, Cpu{});
    for (size_t i = 0; i < k_perm_cpu.size(); ++i) k_perm_cpu[i] = 3 - i;
    auto k_perm = makeSure(k_perm_cpu, xpu);
    vector<int, Cpu> jj_cpu(1, Cpu{});
    jj_cpu[0] = 0;
    auto jj = makeSure(jj_cpu, xpu);

    for (int n = 1; n <= 10; ++n) {
        std::cout << ".. for rhs= " << n << std::endl;
        vector<SCALAR, Cpu> x_cpu(12 * n, Cpu{});
        for (size_t i = 0; i < x_cpu.size(); ++i) x_cpu[i] = {i * 1.f, .1f * i};
        auto x = makeSure(x_cpu, xpu);

        {
            vector<SCALAR, Gpu> y(12 * n, xpu);
            bsr_kron_3x3_4x4perm(a.data(), 1, 3, jj.data(), 1, 1, k.data(), k_perm.data(), x.data(),
                                 12 * n, y.data(), 12 * n, n, xpu);
            auto y_cpu = makeSure(y, Cpu{});

            std::vector<SCALAR> y0(12 * n);
            for (int j = 0; j < n; j++)
                for (int k = 0; k < 4; ++k)
                    for (int i = 0; i < 3; ++i)
                        for (int s = 0; s < 3; ++s)
                            y0[k + 4 * j + 4 * n * i] += a_cpu[i + 3 * s] *
                                                         x_cpu[k_perm_cpu[k] + 4 * j + 4 * n * s] *
                                                         k_cpu[k];

            double r = 0;
            for (int i = 0; i < 12 * n; ++i) r += std::norm(y0[i] - y_cpu[i]);
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
#ifdef SUPERBBLAS_USE_GPU
    {
        Context ctx = createGpuContext(0);
        test_gpu<std::complex<double>>(ctx.toGpu(0));
    }
#endif

    return 0;
}
