#include "superbblas.h"
#include <vector>
#include <iostream>
#ifdef _OPENMP
#    include <omp.h>
#endif

#ifdef SUPERBBLAS_USE_CUDA
#    include <thrust/device_vector.h>
#endif

using namespace superbblas;
using namespace superbblas::detail;

template <std::size_t Nd> using PartitionStored = std::vector<PartitionItem<Nd>>;

template <std::size_t Nd> PartitionStored<Nd> dist_tensor_on_root(Coor<Nd> dim, int nprocs) {
    PartitionStored<Nd> fs(nprocs);
    if (1 <= nprocs) fs[0][1] = dim;
    return fs;
}

int main(int argc, char **argv) {
    int nprocs, rank;
#ifdef SUPERBBLAS_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc;
    (void)argv;
    nprocs = 1;
    rank = 0;
#endif

    constexpr std::size_t Nd = 7; // xyztscn
    constexpr unsigned int nS = 4, nC = 3; // length of dimension spin and color dimensions
    constexpr unsigned int X = 0, Y = 1, Z = 2, T = 3, S = 4, C = 5, N = 6;
    Coor<Nd> dim = {16, 16, 16, 32, nS, nC, 64}; // xyztscn
    Coor<Nd> procs = {1, 1, 1, 1, 1, 1, 1};
    const unsigned int nrep = 10;

    // Get options
    bool procs_was_set = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp("--dim=", argv[i], 6) == 0) {
            if (sscanf(argv[i] + 6, "%d %d %d %d %d", &dim[X], &dim[Y], &dim[Z], &dim[T],
                       &dim[N]) != 5) {
                std::cerr << "--dim= should follow 5 numbers, for instance -dim='2 2 2 2 2'"
                          << std::endl;
                return -1;
            }
        } else if (std::strncmp("--procs=", argv[i], 8) == 0) {
            if (sscanf(argv[i] + 8, "%d %d %d %d", &procs[X], &procs[Y], &procs[Z], &procs[T]) !=
                4) {
                std::cerr << "--procs= should follow 4 numbers, for instance --procs='2 2 2 2'"
                          << std::endl;
                return -1;
            }
            if (detail::volume(procs) != (std::size_t)nprocs) {
                std::cerr << "The total number of processes set by the option `--procs=` should "
                             "match the number of processes"
                          << std::endl;
                return -1;
            }
            procs_was_set = true;
         } else if(std::strncmp("--help", argv[i], 6) == 0) {
             std::cout << "Commandline option:\n  " << argv[0]
                       << " [--dim='x y z t n'] [--procs='x y z t n'] [--help]" << std::endl;
             return 0;
        } else {
            std::cerr << "Not sure what is this: `" << argv[i] << "`" << std::endl;
            return -1;
        }
    }

    // If --procs isn't set, put all processes on the first dimension
    if (!procs_was_set) procs[X] = nprocs;

    // Show lattice dimensions and processes arrangement
    if (rank == 0) {
        std::cout << "Testing lattice dimensions xyzt= " << dim[X] << " " << dim[Y] << " " << dim[Z]
                  << " " << dim[T] << " spin-color= " << dim[S] << " " << dim[C]
                  << "  num_vecs= " << dim[N] << std::endl;
        std::cout << "Processes arrangement xyzt= " << procs[X] << " " << procs[Y] << " "
                  << procs[Z] << " " << procs[T] << std::endl;
    }

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    // using Scalar = float;
    using Scalar = std::complex<float>;
    using ScalarD = std::complex<double>;
    {
        using Tensor = std::vector<Scalar>;
        using TensorD = std::vector<ScalarD>;

        // Create tensor t0 of Nd-1 dims: a lattice color vector
        const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
        const Coor<Nd - 1> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
        PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
        const Coor<Nd - 1> local_size0 = p0[rank][1];
        std::size_t vol0 = detail::volume(local_size0);
        Tensor t0(vol0);

        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        const Coor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]}; // tnsxyzc
        const Coor<Nd> procs1 = {procs[T], procs[N], 1, procs[X], procs[Y], procs[Z], 1}; // tnsxyzc
        PartitionStored<Nd> p1 = basic_partitioning(dim1, procs1);
        const Coor<Nd> local_size1 = p1[rank][1];
        std::size_t vol1 = detail::volume(local_size1);
        Tensor t1(vol1);

        // Dummy initialization of t0
        for (unsigned int i = 0; i < vol0; i++) t0[i] = i;

        // Create a context in which the vectors live
        Context ctx = createCpuContext();

        if (rank == 0)
            std::cout << ">>> CPU tests with " << num_threads << " threads" << std::endl;

        if (rank == 0)
            std::cout << "Maximum number of elements in a tested tensor per process: " << vol1
                      << " ( " << vol1 * 1.0 * sizeof(Scalar) / 1024 / 1024 << " MiB)" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
#ifdef _OPENMP
#    pragma omp parallel for
#endif
                    for (unsigned int i = 0; i < (unsigned int)vol0; ++i)
                        t1[i + n * (unsigned int)vol0] = t0[i];
                }
            }
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep
                          << std::endl;
            tref = t / nrep; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    Scalar *ptr0 = t0.data(), *ptr1 = t1.data();
                    copy(1.0, p0.data(), 1, "xyztsc", from0, dim0, (const Scalar **)&ptr0, &ctx,
                         p1.data(), 1, "tnsxyzc", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                         MPI_COMM_WORLD,
#endif
                         SlowToFast, Copy);
                }
            }
            t = w_time() - t;
            if (rank == 0)
              std::cout << "Time in copying/permuting from xyztsc to tnsxyzc "
                        << t / nrep << " (overhead " << t / nrep / tref << " )"
                        << std::endl;
        }

        // Copy tensor t0 into each of the n components of tensor 1 (fast)
//         {
//             // Create tensor t0 of Nd-1 dims: a lattice color vector
//             const Coor<Nd - 2> dim0a = {dim[X], dim[Y], dim[Z], dim[T], dim[S]}; // xyzts
//             const Coor<Nd - 2> procs0a = {procs[X], procs[Y], procs[Z], procs[T], 1}; // xyzts
//             PartitionStored<Nd - 2> p0a = basic_partitioning(dim0a, procs0a);
//             const Coor<Nd - 2> local_size0a = p0a[rank][1];
//             assert(vol0 == detail::volume(local_size0a) * nC);
//             (void)local_size0a;
// 
//             // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
//             const Coor<Nd - 1> dim1a = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z]}; // tnsxyz
//             const Coor<Nd - 1> procs1a = {procs[T], procs[N], 1, procs[X], procs[Y], procs[Z]}; // tnsxyz
//             PartitionStored<Nd - 1> p1a = basic_partitioning(dim1a, procs1a);
//             const Coor<Nd - 1> local_size1a = p1a[rank][1];
//             assert(vol1 == detail::volume(local_size1a) * nC);
//             (void)local_size1a;
// 
//             double t = w_time();
//             for (unsigned int rep = 0; rep < nrep; ++rep) {
//                 for (int n = 0; n < dim[N]; ++n) {
//                     const Coor<Nd - 2> from0a = {0};
//                     const Coor<Nd - 1> from1a = {0, n, 0};
//                     Coor<Nd - 2> dim0a;
//                     std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
//                     Coor<Nd - 1> dim1a;
//                     std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
//                     std::array<Scalar, nC> *ptr0 = (std::array<Scalar, nC> *)t0.data(),
//                                       *ptr1 = (std::array<Scalar, nC> *)t1.data();
//                     copy(1.0, p0a.data(), 1, "xyzts", from0a, dim0a,
//                          (const std::array<Scalar, nC> **)&ptr0, &ctx, p1a.data(), 1, "tnsxyz",
//                          from1a, (std::array<Scalar, nC> **)&ptr1, &ctx,
// #ifdef SUPERBBLAS_USE_MPI
//                          MPI_COMM_WORLD,
// #endif
//                          SlowToFast, Copy);
//                  }
//             }
//             t = w_time() - t;
//             if (rank == 0)
//                 std::cout << "Time in copying/permuting from xyzts to tnsxyzs (fast) " << t / nrep
//                           << " (overhead " << t / nrep / tref << " )" << std::endl;
//         }

        // Copy tensor t0 into each of the c components of tensor 1 in double
        {
            TensorD t1d(vol1);
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    Scalar *ptr0 = t0.data(); ScalarD *ptr1 = t1d.data();
                    copy(1.0, p0.data(), 1, "xyztsc", from0, dim0, (const Scalar **)&ptr0, &ctx,
                         p1.data(), 1, "tnsxyzc", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                         MPI_COMM_WORLD,
#endif
                         SlowToFast, Copy);
                }
            }
            t = w_time() - t;
            if (rank == 0)
              std::cout << "Time in copying/permuting from xyztsc (single) to tnsxyzc (double) "
                        << t / nrep << " (overhead " << t / nrep / tref << " )"
                        << std::endl;
        }


        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                Scalar *ptr0 = t1.data(), *ptr1 = t2.data();
                copy(1.0, p1.data(), 1, "tnsxyzc", from0, dim1, (const Scalar **)&ptr0, &ctx,
                     p1.data(), 1, "tnsxyzc", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                     MPI_COMM_WORLD,
#endif
                     SlowToFast, Copy);
            }
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in shifting " << t / nrep << std::endl;
        }

        // Create tensor t3 of 5 dims
        const Coor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        PartitionStored<5> pc = dist_tensor_on_root<5>(dimc, nprocs);
        const Coor<5> local_sizec = pc[rank][1];
        std::size_t volc = detail::volume(local_sizec);
        Tensor tc(volc);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Scalar *ptr0 = t1.data(), *ptr1 = t2.data(), *ptrc = tc.data();
                contraction((Scalar)1.0, p1.data(), 1, "tnsxyzc", false, (const Scalar **)&ptr0,
                            &ctx, p1.data(), 1, "tNSxyzc", false, (const Scalar **)&ptr1, &ctx,
                            (Scalar)0.0, pc.data(), 1, "tNSns", &ptrc, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                            MPI_COMM_WORLD,
#endif
                            SlowToFast);
            }
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in contracting xyzs " << t / nrep << std::endl;
        }

        if (rank == 0) reportTimings(std::cout);
    }
#ifdef SUPERBBLAS_USE_CUDA
    {
	resetTimings();

        using Tensor = thrust::device_vector<Scalar>;

        // Create tensor t0 of Nd-1 dims: a lattice color vector
        const Coor<Nd - 1> dim0 = {dim[X], dim[Y], dim[Z], dim[T], dim[S], dim[C]}; // xyztsc
        const Coor<Nd - 1> procs0 = {procs[X], procs[Y], procs[Z], procs[T], 1, 1}; // xyztsc
        PartitionStored<Nd - 1> p0 = basic_partitioning(dim0, procs0);
        const Coor<Nd - 1> local_size0 = p0[rank][1];
        std::size_t vol0 = detail::volume(local_size0);
        Tensor t0(vol0);

        // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
        const Coor<Nd> dim1 = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z], dim[C]}; // tnsxyzc
        const Coor<Nd> procs1 = {procs[T], procs[N], 1, procs[X], procs[Y], procs[Z], 1}; // tnsxyzc
        PartitionStored<Nd> p1 = basic_partitioning(dim1, procs1);
        const Coor<Nd> local_size1 = p1[rank][1];
        std::size_t vol1 = detail::volume(local_size1);
        Tensor t1(vol1);

        // Dummy initialization of t0
        std::vector<Scalar> t0_host(vol0);
        for (unsigned int i = 0; i < vol0; i++) t0_host[i] = i;
        t0 = t0_host;

        // Create a context in which the vectors live
        Context ctx = createCudaContext();

        if (rank == 0)
            std::cout << ">>> GPU tests with " << num_threads << " threads" << std::endl;

        // Copy tensor t0 into tensor 1 (for reference)
        double tref = 0.0;
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    thrust::copy_n(t0.begin(), vol0, t1.begin() + n * vol0);
                }
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            if (rank == 0)
                std::cout << "Time in dummy copying from xyzts to tnsxyzc " << t / nrep
                          << std::endl;
            tref = t / nrep; // time in copying a whole tensor with size dim1
        }


        // Copy tensor t0 into each of the c components of tensor 1
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                for (int n = 0; n < dim[N]; ++n) {
                    const Coor<Nd - 1> from0 = {0};
                    const Coor<Nd> from1 = {0, n, 0};
                    Scalar *ptr0 = t0.data().get(), *ptr1 = t1.data().get();
                    copy(1.0, p0.data(), 1, "xyztsc", from0, dim0, (const Scalar **)&ptr0, &ctx,
                         p1.data(), 1, "tnsxyzc", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                         MPI_COMM_WORLD,
#endif
                         SlowToFast, Copy);
                }
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            if (rank == 0)
              std::cout << "Time in copying/permuting from xyztsc to tnsxyzc "
                        << t / nrep << " (overhead " << t / nrep / tref << " )"
                        << std::endl;
        }

        // Copy tensor t0 into each of the n components of tensor 1 (fast)
//         {
//             // Create tensor t0 of Nd-1 dims: a lattice color vector
//             const Coor<Nd - 2> dim0a = {dim[X], dim[Y], dim[Z], dim[T], dim[S]}; // xyzts
//             const Coor<Nd - 2> procs0a = {procs[X], procs[Y], procs[Z], procs[T], 1}; // xyzts
//             PartitionStored<Nd - 2> p0a = basic_partitioning(dim0a, procs0a);
//             const Coor<Nd - 2> local_size0a = p0a[rank][1];
//             assert(vol0 == detail::volume(local_size0a) * nC);
//             (void)local_size0a;
// 
//             // Create tensor t1 of Nd dims: several lattice color vectors forming a matrix
//             const Coor<Nd - 1> dim1a = {dim[T], dim[N], dim[S], dim[X], dim[Y], dim[Z]}; // tnsxyz
//             const Coor<Nd - 1> procs1a = {procs[T], procs[N], 1, procs[X], procs[Y], procs[Z]}; // tnsxyz
//             PartitionStored<Nd - 1> p1a = basic_partitioning(dim1a, procs1a);
//             const Coor<Nd - 1> local_size1a = p1a[rank][1];
//             assert(vol1 == detail::volume(local_size1a) * nC);
//             (void)local_size1a;
// 
// #    ifndef SUPERBBLAS_LIB
//             double t = w_time();
//             for (unsigned int rep = 0; rep < nrep; ++rep) {
//                 for (int n = 0; n < dim[N]; ++n) {
//                     const Coor<Nd - 2> from0a = {0};
//                     const Coor<Nd - 1> from1a = {0, n, 0};
//                     Coor<Nd - 2> dim0a;
//                     std::copy_n(dim0.begin(), Nd - 2, dim0a.begin());
//                     Coor<Nd - 1> dim1a;
//                     std::copy_n(dim1.begin(), Nd - 1, dim1a.begin());
//                     std::array<Scalar, nC> *ptr0 = (std::array<Scalar, nC> *)t0.data().get(),
//                                       *ptr1 = (std::array<Scalar, nC> *)t1.data().get();
//                     copy(1.0, p0a.data(), 1, "xyzts", from0a, dim0a,
//                          (const std::array<Scalar, nC> **)&ptr0, &ctx, p1a.data(), 1, "tnsxyz",
//                          from1a, (std::array<Scalar, nC> **)&ptr1, &ctx,
// #ifdef SUPERBBLAS_USE_MPI
//                          MPI_COMM_WORLD,
// #endif
//                          SlowToFast, Copy);
//                  }
//             }
//             cudaDeviceSynchronize();
//             t = w_time() - t;
//             if (rank == 0)
//                 std::cout << "Time in copying/permuting from xyzts to tnsxyzs (fast) " << t / nrep
//                           << " (overhead " << t / nrep / tref << " )" << std::endl;
//#    endif // SUPERBBLAS_LIB
//         }

        // Shift tensor 1 on the z-direction and store it on tensor 2
        Tensor t2(vol1);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                const Coor<Nd> from0 = {0};
                Coor<Nd> from1 = {0};
                from1[4] = 1; // Displace one on the z-direction
                Scalar *ptr0 = t1.data().get(), *ptr1 = t2.data().get();
                copy(1.0, p1.data(), 1, "tnsxyzc", from0, dim1, (const Scalar **)&ptr0, &ctx,
                     p1.data(), 1, "tnsxyzc", from1, &ptr1, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                     MPI_COMM_WORLD,
#endif
                     SlowToFast, Copy);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in shifting " << t / nrep << std::endl;
        }

        // Create tensor t3 of 5 dims
        const Coor<5> dimc = {dim[T], dim[N], dim[S], dim[N], dim[S]}; // tnsns
        PartitionStored<5> pc = dist_tensor_on_root(dimc, nprocs);
        const Coor<5> local_sizec = pc[rank][1];
        std::size_t volc = detail::volume(local_sizec);
        Tensor tc(volc);
        {
            double t = w_time();
            for (unsigned int rep = 0; rep < nrep; ++rep) {
                Scalar *ptr0 = t1.data().get(), *ptr1 = t2.data().get(), *ptrc = tc.data().get();
                contraction((Scalar)1.0, p1.data(), 1, "tnsxyzc", false, (const Scalar **)&ptr0,
                            &ctx, p1.data(), 1, "tNSxyzc", false, (const Scalar **)&ptr1, &ctx,
                            (Scalar)0.0, pc.data(), 1, "tNSns", &ptrc, &ctx,
#ifdef SUPERBBLAS_USE_MPI
                            MPI_COMM_WORLD,
#endif
                            SlowToFast);
            }
            cudaDeviceSynchronize();
            t = w_time() - t;
            if (rank == 0) std::cout << "Time in contracting xyzs " << t / nrep << std::endl;
        }

        if (rank == 0) reportTimings(std::cout);
    }
#endif

#ifdef SUPERBBLAS_USE_MPI
    MPI_Finalize();
#endif // SUPERBBLAS_USE_MPI

    return 0;
}
