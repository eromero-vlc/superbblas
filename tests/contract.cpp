#include "superbblas.h"
#include <iostream>
#include <vector>

using namespace superbblas;

template <std::size_t N, typename T> using Operator = std::tuple<Coor<N>, Order<N>, std::vector<T>>;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <typename T> T conj(T t) { return std::conj(t); }
template <> float conj<float>(float t) { return t; }
template <> double conj<double>(double t) { return t; }

template <typename T, typename T::value_type = 0>
T make_complex(typename T::value_type a, typename T::value_type b) {
    return T{a, b};
}
template <typename T> T make_complex(T a, T) { return a; }

template <std::size_t N> Order<N + 1> toStr(Order<N> o) {
    Order<N + 1> r{};
    std::copy(o.begin(), o.end(), r.begin());
    return r;
}

template <std::size_t NA, std::size_t NB, std::size_t NC, typename T>
Operator<NA + NB + NC, T> generate_tensor(char a, char b, char c) {
    // Build the operator with A,B,C
    constexpr std::size_t N = NA + NB + NC;
    Coor<N> dim{};
    for (IndexType &c : dim) c = 2;
    std::size_t vol = detail::volume(dim);
    std::vector<T> v(vol);
    for (std::size_t i = 0; i < vol; ++i) v[i] = make_complex<T>(i, i);
    Order<N> o{};
    for (std::size_t i = 0; i < NA; ++i) o[i] = a + i;
    for (std::size_t i = 0; i < NB; ++i) o[i + NA] = b + i;
    for (std::size_t i = 0; i < NC; ++i) o[i + NA + NB] = c + i;
    return {dim, o, v};
}

const char sT = 'A', sA = sT + 8, sB = sA + 8, sC = sB + 8;

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T>
void test_contraction(Operator<N0, T> op0, Operator<N1, T> op1, Operator<N2, T> op2, bool conj0,
                      bool conj1, char dist_dir) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    nprocs = 1;
    rank = 0;
#endif

    const Coor<N0> dim0 = std::get<0>(op0);
    const Coor<N1> dim1 = std::get<0>(op1);
    const Coor<N2> dim2 = std::get<0>(op2);
    const auto o0 = toStr(std::get<1>(op0));
    const auto o1 = toStr(std::get<1>(op1));
    const auto o2 = toStr(std::get<1>(op2));
    const std::vector<T> v0_ = std::get<2>(op0);
    const std::vector<T> v1_ = std::get<2>(op1);
    const std::vector<T> v2_ = std::get<2>(op2);

    Context ctx = createCpuContext();

    // Distribute op0, op1, and a zeroed op2 along the `dist_dir` direction

    Coor<N0> procs0;
    for (std::size_t i = 0; i < N0; ++i) procs0[i] = o0[i] == dist_dir ? nprocs : 1;
    PartitionStored<N0> p0 = basic_partitioning(dim0, procs0, nprocs, true);
    std::vector<T> v0(detail::volume(p0[rank][1]));
    PartitionStored<N0> p0_(nprocs, {Coor<N0>{}, dim0}); // tensor replicated partitioning
    T const *ptrv0_ = v0_.data();
    T *ptrv0 = v0.data();
    copy(1.0, p0_.data(), 1, &o0[0], {}, dim0, (const T **)&ptrv0_, &ctx, p0.data(), 1, &o0[0], {},
         &ptrv0, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    Coor<N1> procs1;
    for (std::size_t i = 0; i < N1; ++i) procs1[i] = o1[i] == dist_dir ? nprocs : 1;
    PartitionStored<N1> p1 = basic_partitioning(dim1, procs1, nprocs, true);
    std::vector<T> v1(detail::volume(p1[rank][1]));
    PartitionStored<N1> p1_(nprocs, {Coor<N1>{}, dim1}); // tensor replicated partitioning
    T const *ptrv1_ = v1_.data();
    T *ptrv1 = v1.data();
    copy(1.0, p1_.data(), 1, &o1[0], {}, dim1, (const T **)&ptrv1_, &ctx, p1.data(), 1, &o1[0], {},
         &ptrv1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    Coor<N2> procs2;
    for (std::size_t i = 0; i < N2; ++i) procs2[i] = o2[i] == dist_dir ? nprocs : 1;
    PartitionStored<N2> p2 = basic_partitioning(dim2, procs2, nprocs, true);
    std::vector<T> v2(detail::volume(p2[rank][1]));
    T *ptrv2 = v2.data();

    // Contract the distributed matrices

    contraction(T{1}, p0.data(), 1, &o0[0], conj0, (const T **)&ptrv0, &ctx, p1.data(), 1, &o1[0],
                conj1, (const T **)&ptrv1, &ctx, T{0}, p2.data(), 1, &o2[0], &ptrv2, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                MPI_COMM_WORLD,
#endif
                SlowToFast);

    // Move the result to proc 0
    PartitionStored<N2> pr(nprocs, {Coor<N2>{}, Coor<N2>{}});
    pr[0][1] = dim2; // tensor only supported on proc 0
    std::vector<T> vr(detail::volume(pr[rank][1]));
    T *ptrvr = vr.data();
    copy(1, p2.data(), 1, &o2[0], {}, dim2, (const T **)&ptrv2, &ctx, pr.data(), 1, &o2[0], {},
         &ptrvr, &ctx,
#ifdef SUPERBBLAS_USE_MPI
         MPI_COMM_WORLD,
#endif
         SlowToFast, Copy);

    // Test the resulting tensor

    if (rank == 0) {
        double diff_fn = 0, fn = 0; // Frob-norm of the difference and the correct tensor
        for (std::size_t i = 0; i < v2_.size(); ++i)
            diff_fn += std::norm(v2_[i] - vr[i]), fn += std::norm(v2_[i]);
        diff_fn = std::sqrt(diff_fn);
        fn = std::sqrt(fn);
        if (diff_fn > fn * 1e-4)
            throw std::runtime_error(
                "Result of contraction does not match with the correct answer");
    }
}

template <std::size_t N0, std::size_t N1, std::size_t N2, typename T>
void test_contraction(Operator<N0, T> p0, Operator<N1, T> p1, Operator<N2, T> p2) {
    // Compute correct result of the contraction of p0 and p1
    const Coor<N0> dim0 = std::get<0>(p0);
    const Coor<N1> dim1 = std::get<0>(p1);
    const Coor<N2> dim2 = std::get<0>(p2);
    const Order<N0> o0 = std::get<1>(p0);
    const Order<N1> o1 = std::get<1>(p1);
    const Order<N2> o2 = std::get<1>(p2);
    const std::vector<T> v0 = std::get<2>(p0);
    const std::vector<T> v1 = std::get<2>(p1);
    std::vector<T> r0(detail::volume(dim2)); // p0 not conj, and p1 not conj
    std::vector<T> r1(detail::volume(dim2)); // p0 conj, and p1 not conj
    std::vector<T> r2(detail::volume(dim2)); // p0 not conj, and p1 conj
    std::vector<T> r3(detail::volume(dim2)); // p0 conj, and p1 conj
    Coor<N0> strides0 = detail::get_strides(dim0, SlowToFast);
    Coor<N1> strides1 = detail::get_strides(dim1, SlowToFast);
    Coor<N2> strides2 = detail::get_strides(dim2, SlowToFast);
    for (std::size_t i = 0, m = detail::volume(dim0); i < m; ++i) {
        std::vector<int> dim(128, -1);
        Coor<N0> c0 = detail::index2coor(i, dim0, strides0);
        for (std::size_t d = 0; d < N0; ++d) dim[o0[d]] = c0[d];
        for (std::size_t j = 0, n = detail::volume(dim1); j < n; ++j) {
            std::vector<int> dim_ = dim;
            Coor<N1> c1 = detail::index2coor(j, dim1, strides1);
            bool get_out = false;
            for (std::size_t d = 0; d < N1; ++d) {
                if (dim_[o1[d]] == -1)
                    dim_[o1[d]] = c1[d];
                else if (dim_[o1[d]] != c1[d]) {
                    get_out = true;
                    break;
                }
            }
            if (get_out) continue;
            Coor<N2> c2{};
            for (std::size_t d = 0; d < N2; ++d) c2[d] = dim_[o2[d]];
            IndexType k = detail::coor2index(c2, dim2, strides2);
            r0[k] += v0[i] * v1[j];
            r1[k] += conj(v0[i]) * v1[j];
            r2[k] += v0[i] * conj(v1[j]);
            r3[k] += conj(v0[i]) * conj(v1[j]);
        }
    }

    std::vector<char> labels({sT, sA, sB, sC});
    for (char c : labels) {
        // Test first operator no conj and second operator no conj
        std::get<2>(p2) = r0;
        test_contraction(p0, p1, p2, false, false, c);
        // Test first operator conj and second operator no conj
        std::get<2>(p2) = r1;
        test_contraction(p0, p1, p2, true, false, c);
        // Test first operator no conj and second operator conj
        std::get<2>(p2) = r2;
        test_contraction(p0, p1, p2, false, true, c);
        // Test first operator conj and second operator conj
        std::get<2>(p2) = r3;
        test_contraction(p0, p1, p2, true, true, c);
    }
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_third_operator(Operator<NT + NA + NB, T> p0, Operator<NT + NA + NC, T> p1) {
    test_contraction(p0, p1, generate_tensor<NT, NB, NC, T>(sT, sB, sC));
    test_contraction(p0, p1, generate_tensor<NT, NC, NB, T>(sT, sC, sB));
    test_contraction(p0, p1, generate_tensor<NB, NC, NT, T>(sB, sC, sT));
    test_contraction(p0, p1, generate_tensor<NB, NT, NC, T>(sB, sT, sC));
    test_contraction(p0, p1, generate_tensor<NC, NB, NT, T>(sC, sB, sT));
    test_contraction(p0, p1, generate_tensor<NC, NT, NB, T>(sC, sT, sB));
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_second_operator(Operator<NT + NA + NB, T> p0) {
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NT, NA, NC, T>(sT, sA, sC));
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NT, NC, NA, T>(sT, sC, sA));
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NA, NC, NT, T>(sA, sC, sT));
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NA, NT, NC, T>(sA, sT, sC));
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NC, NA, NT, T>(sC, sA, sT));
    test_third_operator<NT, NA, NB, NC, T>(p0, generate_tensor<NC, NT, NA, T>(sC, sT, sA));
}

template <std::size_t NT, std::size_t NA, std::size_t NB, std::size_t NC, typename T>
void test_first_operator() {
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NA, NB, T>(sT, sA, sB));
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NT, NB, NA, T>(sT, sB, sA));
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NB, NT, T>(sA, sB, sT));
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NA, NT, NB, T>(sA, sT, sB));
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NA, NT, T>(sB, sA, sT));
    test_second_operator<NT, NA, NB, NC, T>(generate_tensor<NB, NT, NA, T>(sB, sT, sA));
}

template <std::size_t NT, std::size_t NA, std::size_t NB, typename T> void test_for_C() {
    test_first_operator<NT, NA, NB, 0, T>();
    test_first_operator<NT, NA, NB, 1, T>();
    test_first_operator<NT, NA, NB, 2, T>();
}

template <std::size_t NT, std::size_t NA, typename T> void test_for_B() {
    test_for_C<NT, NA, 0, T>();
    test_for_C<NT, NA, 1, T>();
    test_for_C<NT, NA, 2, T>();
}

template <std::size_t NT, typename T> void test_for_A() {
    test_for_B<NT, 0, T>();
    test_for_B<NT, 1, T>();
    test_for_B<NT, 2, T>();
}

template <typename T> void test() {
    test_for_A<1, T>();
    test_for_A<2, T>();
}

int main(int argc, char **argv) {
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc;
    (void)argv;
#endif

    test<double>();
    test<std::complex<double>>();

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
