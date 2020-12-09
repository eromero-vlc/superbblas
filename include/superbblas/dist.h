#ifndef __SUPERBBLAS_DIST__
#define __SUPERBBLAS_DIST__

#include "tensor.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef SUPERBBLAS_USE_MPI
#    include "mpi.h"
#endif // SUPERBBLAS_USE_MPI

namespace superbblas {

    /// First coordinate and size of a range
    template <unsigned int N> using From_size_item = std::array<Coor<N>, 2>;
    /// List of ranges
    template <unsigned int N> using From_size = std::vector<From_size_item<N>>;
    /// From_size iterator
    template <unsigned int N> using From_size_iterator = typename From_size<N>::iterator;
    /// From_size iterator
    template <unsigned int N>
    using From_size_const_iterator = typename From_size<N>::const_iterator;

    namespace detail {

        /// Type use in MPI calls to indicate cardinality and displacements
        using MpiInt = int;

        //
        // Auxiliary functions
        //

        /// Return the global dimensions of a tensor from its partitioning
        /// \param p: partitioning of the tensor in consecutive ranges

        template <unsigned long N> Coor<N> get_dim(From_size_const_iterator<N> p, std::size_t n) {
            Coor<N> r = fill_coor<N>(0);
            for (std::size_t j = 0; j < n; j++) r = max_each(r, p[j][0] + p[j][1]);
            return r;
        }

        /// Return the global dimensions of a tensor from its partitioning
        /// \param p: partitioning of the tensor in consecutive ranges

        template <unsigned long N> Coor<N> get_dim(const From_size<N> &p) {
            return get_dim<N>(p.begin(), p.size());
        }

        /// Return the total volume in a partition
        /// \param p: partitioning of the tensor in consecutive ranges

        template <unsigned long N>
        std::size_t get_volume(From_size_const_iterator<N> p, std::size_t n) {
            std::size_t r = 0;
            for (std::size_t j = 0; j < n; j++) r += volume<N>(p[j][1]);
            return r;
        }

        /// Return the total volume in a partition
        /// \param p: partitioning of the tensor in consecutive ranges

        template <unsigned long N> std::size_t get_volume(const From_size<N> &p) {
            return get_volume<N>(p.begin(), p.size());
        }

        /// Output of `send` and input of `wait`
        using Request = std::function<void(void)>;

        /// Wait until an operation started by `send` finishes
        /// \param request:

        inline void wait(Request request) { request(); }

#ifdef SUPERBBLAS_USE_MPI
        /// Communicator
        struct MpiComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
            MPI_Comm comm;       ///< MPI communicator
        };

        /// Return a communicator for a MPI_Comm
        MpiComm get_comm(MPI_Comm comm) {
            int nprocs, rank;
            MPI_Comm_size(comm, &nprocs);
            MPI_Comm_rank(comm, &rank);
            return MpiComm{(unsigned int)nprocs, (unsigned int)rank, comm};
        }

#endif // SUPERBBLAS_USE_MPI

        /// Communicator
        struct SelfComm {
            unsigned int nprocs; ///< Number of processes
            unsigned int rank;   ///< Process id
        };

        /// Return a communicator for a MPI_Comm
        SelfComm get_comm() { return SelfComm{1u, 0u}; }

        /// Return the native type of MPI_Datatype use for MPI communications
        template <typename T>
        using NativeMpiDatatype = typename std::conditional<
            sizeof(T) % sizeof(double) == 0, double,
            std::conditional<sizeof(T) % sizeof(float) == 0, float, char>>::type;

#ifdef SUPERBBLAS_USE_MPI
        /// Return the MPI_datatype for a type returned by `NativeMpiDatatype`
        template <typename T> MPI_Datatype get_mpi_datatype() {
            if (sizeof(T) % sizeof(double) == 0) return MPI_DOUBLE;
            if (sizeof(T) % sizeof(float) == 0) return MPI_FLOAT;
            return MPI_CHAR;
        }
#endif // SUPERBBLAS_USE_MPI

        /// Component of a tensor
        template <unsigned int Nd, typename T, typename XPU> struct Component {
            data<T, XPU> it;          ///< data
            Coor<Nd> dim;             ///< dimension of the tensor
            XPU xpu;                  ///< context of the data
            unsigned int componentId; ///< Component Id

            template <typename Q = T, typename = typename std::enable_if<
                          std::is_same<Q, typename std::remove_const<Q>::type>::value>::type>
            operator Component<Nd, const Q, XPU>() const {
                return {it, dim, xpu, componentId};
            }
         };

        /// A tensor composed of several components
        template <unsigned int Nd, typename T, typename XPU0, typename XPU1>
        using Components_tmpl =
            std::pair<std::vector<Component<Nd, T, XPU0>>, std::vector<Component<Nd, T, XPU1>>>;

#ifdef SUPERBBLAS_USE_CUDA
        /// A tensor composed of several CPU and CUDA elements
        template <unsigned int Nd, typename T> using Components = Components_tmpl<Nd, T, Cuda, Cpu>;
#else
        /// A tensor composed of only of CPU components
        template <unsigned int Nd, typename T> using Components = Components_tmpl<Nd, T, Cpu, Cpu>;
#endif // SUPERBBLAS_USE_CUDA

        template <unsigned int Nd, typename T>
        Components<Nd, T> get_components(T **v, Context *ctx, unsigned int ncomponents,
                                         From_size_const_iterator<Nd> dim) {
            Components<Nd, T> r;
            for (unsigned int i = 0; i < ncomponents; ++i) {
                switch (ctx[i].plat) {
#ifdef SUPERBBLAS_USE_CUDA
                case CPU:
                    r.second.push_back(Component<Nd, T, Cpu>{v[i], dim[i][1], ctx[i].toCpu(), i});
                    break;
                case CUDA:
                    r.first.push_back(Component<Nd, T, Cuda>{encapsulate_pointer(v[i]), dim[i][1],
                                                             ctx[i].toCuda(), i});
                    break;
#else // SUPERBBLAS_USE_CUDA
                case CPU:
                    r.first.push_back(Component<Nd, T, Cpu>{v[i], dim[i][1], ctx[i].toCpu(), i});
                    break;
#endif
                default: throw std::runtime_error("Unsupported platform");
                }
            }
            return r;
        }

        /// Print a message in the standard error
        /// \param comm: a communicator
        /// \param msg: thing to print

        template <typename Comm, typename Msg> void print(const Comm &comm, const Msg msg) {
            std::cerr << "[" << comm.rank << "] " << msg << std::endl;
        }

        /// Print a vector in the standard error
        /// \param comm: a communicator
        /// \param v: vector print
        /// \param name: name to prefix the print

        template <typename Comm, typename Vector>
        void print(const Comm &comm, const Vector &v, const char *name) {
            std::cerr << "[" << comm.rank << "] " << name << ":";
            for (const auto &i : v) std::cerr << " " << i;
            std::cerr << std::endl;
        }

        /// Vectors used in MPI communications
        template <typename T> struct PackedValues {
            std::vector<T> buf;         ///< pointer to data
            std::vector<MpiInt> counts; ///< number of items send/receive for rank i
            std::vector<MpiInt> displ;  ///< index of the first element to send/receive for rank i
        };

#ifdef SUPERBBLAS_USE_MPI
        /// Allocate buffers and prepare arrays from a list of ranges to be used in a MPI communication
        /// \param ranges: iterator over a list of tensor ranges to be packed
        /// \param nranges: number of elements in the list
        /// \param ncomponents: comm.nprocs * ncomponents == the length of each element in `ranges`
        /// \param comm: communicator

        template <unsigned int Nd, typename T, typename From_size_iterator_or_ptr>
        PackedValues<T> prepare_pack(From_size_iterator_or_ptr ranges, unsigned int nranges,
                                     unsigned int ncomponents, MpiComm comm) {

            // Allocate PackedValues
            using MpiT = NativeMpiDatatype<T>;
            PackedValues<T> r{std::vector<T>(), std::vector<int>(comm.nprocs),
                              std::vector<int>(comm.nprocs)};

            // Prepare counts and displ
            std::size_t n = 0; // accumulate total number of T elements
            int d = 0;         // accumulate total number of MpiT elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of T elements in rank
                if (rank != comm.rank) { // Skip the communications of the local rank
                    // Compute the total number of T elements for rank i
                    for (unsigned int irange = 0; irange < nranges; ++irange) {
                        for (unsigned int componentId = 0; componentId < ncomponents;
                             ++componentId) {
                            n_rank +=
                                volume<Nd>(ranges[irange][rank * ncomponents + componentId][1]);
                        }
                    }
                }
                n += n_rank;
                r.counts[rank] = n_rank * MpiInt(sizeof(T) / sizeof(MpiT));
                r.displ[rank] = d;
                d += r.counts[rank];
            }
            r.buf.resize(n);

            return r;
        }

        /// Pack a list of subtensors contiguously in memory
        /// \param o0: dimension labels for the origin tensor
        /// \param fs: a From_size iterator
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param xpu0: device context for v0
        /// \param o1: dimension labels for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param xpu1: device context for v1
        /// \param ncomponents1: comm.nprocs * ncomponents1 == fs.size()
        /// \param comm: communicator

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename XPU0>
        void pack(const Order<Nd0> &o0, const From_size<Nd0> &fs, const Coor<Nd0> &dim0,
                  data<const T, XPU0> v0, XPU0 xpu0, const Order<Nd1> &o1,
                  typename Indices<Cpu>::iterator disp1, data<T, Cpu> v1, unsigned int ncomponents1,
                  MpiComm comm) {

            assert(fs.size() == comm.nprocs * ncomponents1);

            // Get the volume of communicated data without the local part
            std::size_t vol = get_volume<Nd0>(fs) -
                              get_volume<Nd0>(fs.begin() + comm.rank * ncomponents1, ncomponents1);

            // Find indices on cache
            using pointer_perm = std::tuple<const From_size_item<Nd0> *, Coor<Nd0>, Coor<Nd1>, int>;
            using PairIndices = std::pair<Indices<XPU0>, Indices<Cpu>>;
            static std::unordered_map<pointer_perm, PairIndices, TupleHash<pointer_perm>> cache(16);
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            pointer_perm key{fs.data(), perm1, perm0, deviceId(xpu0)};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            if (it == cache.end()) {
                Indices<Cpu> indices0(vol), indices1(vol);
                Coor<Nd1> zeros = fill_coor<Nd1>(0);
                for (std::size_t i = 0, n = 0; i < fs.size(); ++i) {
                    // Skip the communications of the local rank
                    if (i / ncomponents1 == comm.rank) continue;

                    // Compute the permutation so that the subtensors are packed on the natural
                    // order on the destination; in other words, apply the permutation before
                    // doing the MPI call
                    Coor<Nd0> fromi = fs[i][0], sizei = fs[i][1];
                    Coor<Nd1> dim1 = reorder_coor<Nd0, Nd1>(sizei, perm0);
                    std::shared_ptr<Indices<Cpu>> indices;
                    IndexType disp;
                    get_permutation_origin_cache<Nd0, Nd1>(o0, fromi, sizei, dim0, o1, zeros, dim1,
                                                           Cpu{}, indices, disp);
                    assert(indices->size() + n <= vol);
                    std::transform(indices->begin(), indices->end(), indices0.begin() + n,
                                   [&](const IndexType &d) { return d + disp; });
                    get_permutation_destination_cache<Nd0, Nd1>(o0, fromi, sizei, dim0, o1, zeros,
                                                                dim1, Cpu{}, indices, disp);
                    assert(indices->size() + n <= vol);
                    std::transform(indices->begin(), indices->end(), indices1.begin() + n,
                                   [&](const IndexType &d) { return d + disp1[i] + disp; });

                    n += indices->size();
                    assert(n <= vol);
                    assert(i != fs.size() - 1 || n == vol);
                }
                Indices<XPU0> indices0_xpu = indices0;
                cache[key] = PairIndices{indices0_xpu, indices1};
                it = cache.find(key);
            }

            // Do the copy
            copy_n<IndexType, T>(v0, it->second.first.begin(), xpu0, vol, v1,
                                 it->second.second.begin(), Cpu{}, EWOp::Copy{});
        }

        /// Pack a list of ranges to be used in a MPI communication
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param ncomponents0: number of elements in toSend and v
        /// \param v: vector containing the values to send
        /// \param o0: dimension labels for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param ncomponents1: number of components on the receiving tensor
        /// \param comm: communicator

        template <unsigned int Nd0, unsigned int Nd1, typename T,
                  typename From_size_iterator_or_ptr, typename XPU0, typename XPU1>
        PackedValues<T> pack(From_size_iterator_or_ptr toSend, unsigned int ncomponents0,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v,
                             const Order<Nd0> &o0, const Order<Nd1> &o1, unsigned int ncomponents1,
                             MpiComm comm) {

            PackedValues<T> r = prepare_pack<Nd0, T>(toSend, ncomponents0, ncomponents1, comm);

            Indices<Cpu> buf_disp(comm.nprocs);
            std::size_t n = 0; // accumulate total number of T elements
            for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                std::size_t n_rank = 0;  // total number of T elements in rank
                if (rank != comm.rank) { // Skip the local communications
                    // Compute the total number of T elements for rank i
                    for (unsigned int irange = 0; irange < ncomponents0; ++irange) {
                        for (unsigned int componentId1 = 0; componentId1 < ncomponents1;
                             ++componentId1) {
                            n_rank +=
                                volume<Nd0>(toSend[irange][rank * ncomponents1 + componentId1][1]);
                        }
                    }
                }
                buf_disp[rank] = n;
                n += n_rank;
            }
            assert(r.buf.size() == n);

            for (unsigned int componentId0 = 0; componentId0 < ncomponents0; ++componentId0) {
                for (const Component<Nd0, const T, XPU0> &c : v.first) {
                    if (c.componentId == componentId0) {
                        pack<Nd0, Nd1, T>(o0, toSend[componentId0], c.dim, c.it, c.xpu, o1,
                                          buf_disp.begin(), r.buf.data(), ncomponents1, comm);
                        for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                            if (rank != comm.rank)
                                buf_disp[rank] += volume<Nd0>(toSend[componentId0][rank][1]);
                        }
                    }
                }
                for (const Component<Nd0, const T, XPU1> &c : v.second) {
                    if (c.componentId == componentId0) {
                        pack<Nd0, Nd1, T>(o0, toSend[componentId0], c.dim, c.it, c.xpu, o1,
                                          buf_disp.begin(), r.buf.data(), ncomponents1, comm);
                        for (unsigned int rank = 0; rank < comm.nprocs; ++rank) {
                            if (rank != comm.rank)
                                buf_disp[rank] += volume<Nd0>(toSend[componentId0][rank][1]);
                        }
                    }
                }
            }
            return r;
        }

        /// Return an order with values 0, 1, 2, ..., N-1

        template <unsigned long N>
        Order<N> trivial_order() {
            Order<N> r;
            for (unsigned int i = 0; i < N; i++) r[i] = (char)i;
            return r;
        }

        /// Unpack and copy packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param xpu: device context
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication

        template <unsigned int Nd, typename T, typename XPU>
        void unpack(const PackedValues<T> &r, const From_size<Nd> &toReceive,
                    const Component<Nd, T, XPU> &v, unsigned int ncomponents0, MpiComm comm,
                    EWOp::Copy) {

            // Find indices on cache
            using pointer_dev = std::tuple<const From_size_item<Nd> *, int>;
            static std::unordered_map<pointer_dev, Indices<XPU>, TupleHash<pointer_dev>> cache(16);
            pointer_dev key{toReceive.data(), deviceId(v.xpu)};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            std::size_t vol = r.buf.size();
            if (it == cache.end()) {
                Indices<Cpu> indices1(vol);
                Coor<Nd> zeros = fill_coor<Nd>(0);
                Order<Nd> o = trivial_order<Nd>();
                for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                    if (i / ncomponents0 == comm.rank) continue;
                    Coor<Nd> fromi = toReceive[i][0], sizei = toReceive[i][1];
                    std::size_t voli = volume<Nd>(sizei);
                    Indices<Cpu> indices = get_permutation_destination<Nd, Nd>(
                        o, zeros, sizei, sizei, o, fromi, v.dim, Cpu{});
                    std::copy_n(indices.begin(), voli, indices1.begin() + n);
                    n += voli;
                    assert(n <= vol);
                }
                Indices<XPU> indices1_xpu = indices1;
                cache[key] = indices1_xpu;
                it = cache.find(key);
            }

            // Do the copy
            copy_n<IndexType, T>(r.buf.data(), Cpu{}, vol, v.it, it->second.begin(), v.xpu,
                                 EWOp::Copy{});
        }

        /// Unpack and sum-reduce packed tensors from a MPI communication
        /// \param r: packed subtensors
        /// \param toReceive: list of tensor ranges to receive
        /// \param v: data for the destination tensor
        /// \param xpu: device context
        /// \param ncomponents0: number of components on the origin tensor
        /// \param comm: communication

        template <unsigned int Nd, typename T, typename XPU>
        void unpack(const PackedValues<T> &r, const From_size<Nd> &toReceive,
                    const Component<Nd, T, XPU> &v, unsigned int ncomponents0, MpiComm comm,
                    EWOp::Add) {

            // Find indices on cache
            using pointer_dev = std::tuple<const From_size_item<Nd> *, int>;
            using PermPermreduceIndices = std::tuple<Indices<Cpu>, Indices<Cpu>, Indices<XPU>>;
            static std::unordered_map<pointer_dev, PermPermreduceIndices, TupleHash<pointer_dev>>
                cache(16);
            pointer_dev key{toReceive.data(), deviceId(v.xpu)};
            auto it = cache.find(key);

            // If they are not, compute the permutation vectors
            std::size_t vol = r.buf.size();
            if (it == cache.end()) {
                Indices<Cpu> indices1(vol);
                Coor<Nd> zeros = fill_coor<Nd>(0);
                Order<Nd> o = trivial_order<Nd>();
                for (std::size_t i = 0, n = 0; i < comm.nprocs * ncomponents0; ++i) {
                    if (i / ncomponents0 == comm.rank) continue;
                    Coor<Nd> fromi = toReceive[i][0], sizei = toReceive[i][1];
                    std::size_t voli = volume<Nd>(sizei);
                    Indices<Cpu> indices = get_permutation_destination<Nd, Nd>(
                        o, zeros, sizei, sizei, o, fromi, v.dim, Cpu{});
                    std::copy_n(indices.begin(), voli, indices1.begin() + n);
                    n += voli;
                    assert(n <= vol);
                }
                Indices<Cpu> perm;
                for (std::size_t i = 0; i < vol; ++i) perm[i] = i;
                std::sort(perm.begin(), perm.end(), [&](const IndexType &a, const IndexType &b) {
                    return indices1[a] < indices1[b];
                });
                Indices<Cpu> perm_distinct;
                perm_distinct.reserve(vol + 1);
                perm_distinct.push_back(0);
                for (std::size_t i = 1; i < vol; ++i) {
                    if (indices1[perm[i]] != indices1[perm[i - 1]]) perm_distinct.push_back(i);
                }
                perm_distinct.push_back(vol);
                Indices<XPU> indices1_xpu(perm_distinct.size() - 1);
                copy_n<IndexType, IndexType>(indices1.data(), perm_distinct.begin(), Cpu{},
                                             perm_distinct.size() - 1, indices1_xpu.data(), v.xpu,
                                             EWOp::Copy{});
                cache[key] = std::make_tuple(perm, perm_distinct, indices1_xpu);
                it = cache.find(key);
            }

            // Do the copy
            copy_reduce_n<IndexType, T>(r.buf.data(), Cpu{}, std::get<0>(it->second).begin(),
                                        std::get<1>(it->second).begin(),
                                        std::get<1>(it->second).size(), Cpu{}, v.it,
                                        std::get<2>(it->second).begin(), v.xpu);
        }

        /// Asynchronous sending and receiving
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param ncomponents1: number of components on the destination tensor
        /// \param comm: communication

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename XPU0, typename XPU1,
                  typename XPUr, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const std::vector<From_size<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const std::shared_ptr<From_size<Nd1>> toReceive,
                             const Component<Nd1, T, XPUr> &v1, unsigned int ncomponents1,
                             MpiComm comm, EWOp ewop) {

            if (comm.nprocs <= 1) return [] {};

            (void)v1;

            // Pack v0 and prepare for receiving data from other processes
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            std::shared_ptr<PackedValues<T>> v0ToSend = std::make_shared<PackedValues<T>>(
                pack<Nd0, Nd1, T>(toSend.begin(), toSend.size(), v0, o0, o1, ncomponents1, comm));
            std::shared_ptr<PackedValues<T>> v1ToReceive = std::make_shared<PackedValues<T>>(
                prepare_pack<Nd1, T>(&*toReceive, 1, ncomponents0, comm));

            // Do the MPI communication
            MPI_Datatype dtype = get_mpi_datatype<T>();
            MPI_Request r;
            assert(v0ToSend->counts.size() == comm.nprocs);
            assert(v0ToSend->displ.size() == comm.nprocs);
            assert(v1ToReceive->counts.size() == comm.nprocs);
            assert(v1ToReceive->displ.size() == comm.nprocs);
            MPI_Ialltoallv(v0ToSend->buf.data(), v0ToSend->counts.data(), v0ToSend->displ.data(),
                           dtype, v1ToReceive->buf.data(), v1ToReceive->counts.data(),
                           v1ToReceive->displ.data(), dtype, comm.comm, &r);

            // Do this later
            return [=] {
                // Wait for the MPI communication to finish
                MPI_Request r0 = r;
                MPI_Wait(&r0, MPI_STATUS_IGNORE);

                // Do this copy is unnecessary, but v0ToSend needs to be captured to avoid
                // being released until this point
                std::shared_ptr<PackedValues<T>> v0ToSend_dummy = v0ToSend;

                // Copy back to v1
                unpack<Nd1>(*v1ToReceive, *toReceive, v1, ncomponents0, comm, ewop);
            };
        }
#endif // SUPERBBLAS_USE_MPI

        /// Asynchronous sending and receiving; do nothing for `SelfComm` communicator
        /// \param o0: dimension labels for the origin tensor
        /// \param toSend: list of tensor ranges to be sent for each component
        /// \param v0: origin data to send
        /// \param o1: dimension labels for the destination tensor
        /// \param toReceive: list of tensor ranges to receive
        /// \param v1: destination data
        /// \param ncomponents1: number of components on the destination tensor
        /// \param comm: communication

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename XPU0, typename XPU1,
                  typename XPUr, typename EWOp>
        Request send_receive(const Order<Nd0> &o0, const std::vector<From_size<Nd0>> &toSend,
                             const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                             const Order<Nd1> &o1, const std::shared_ptr<From_size<Nd1>> toReceive,
                             const Component<Nd1, T, XPUr> &v1, unsigned int ncomponents1,
                             SelfComm comm, EWOp ewop) {
            (void)o0;
            (void)toSend;
            (void)v0;
            (void)o1;
            (void)toReceive;
            (void)v1;
            (void)ncomponents1;
            (void)ewop;
            if (comm.nprocs <= 1) return [] {};
            throw std::runtime_error("Unsupported SelfComm with nprocs > 1");
        }

        /// Return coor[i] % dim[i]
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions

        template <unsigned int Nd>
        Coor<Nd> normalize_coor(const Coor<Nd> &coor, const Coor<Nd> &dim) {
            Coor<Nd> r;
            for (unsigned int j = 0; j < Nd; j++) r[j] = coor[j] % dim[j];
            return r;
        }

        /// Return the intersection of two 1D ranges for a NOT toroidal lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param fromr: first coordinate of the resulting range
        /// \param sizer: size of the resulting range

        inline void intersection(IndexType from0, IndexType size0, IndexType from1, IndexType size1,
                                 IndexType dim, IndexType &fromr, IndexType &sizer) {
            fromr = from0 + std::min(std::max(from1 - from0, 0), size0);
            sizer = from0 + std::min(std::max(from1 + size1 - from0, 0), size0) - fromr;
            fromr = (fromr + dim) % dim;
        }

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param fromr0: first coordinate of the first resulting range
        /// \param sizer0: size of the first resulting range
        /// \param fromr1: first coordinate of the second resulting range
        /// \param sizer1: size of the second resulting range

        template <unsigned int Nd>
        From_size<Nd> intersection(const Coor<Nd> &from0, const Coor<Nd> &size0,
                                   const Coor<Nd> &from1, const Coor<Nd> &size1,
                                   const Coor<Nd> &dim) {

            From_size<Nd> r;
            r.push_back({fill_coor<Nd>(0), fill_coor<Nd>(0)});
            for (unsigned int i = 0; i < Nd; ++i) {
                //
                // Compute the subintervals for the dimension ith
                //
                IndexType fromr0 = 0, sizer0 = 0, fromr1 = 0, sizer1 = 0;

                // Proceed with easy cases: if one of the ranges in the whole lattice
                if (size0[i] == dim[i]) {
                    fromr0 = from1[i], sizer0 = size1[i];
                } else if (size1[i] == dim[i]) {
                    fromr0 = from0[i], sizer0 = size0[i];

                // Proceed with the general case
                } else {
                    intersection(from0[i], size0[i], from1[i], size1[i], dim[i], fromr0, sizer0);
                    intersection(from0[i], size0[i], from1[i] + dim[i], size1[i], dim[i], fromr1,
                                 sizer1);
                    From_size<Nd> q;
                    for(const auto &fs : r) {
                        if (sizer0 > 0) {
                            Coor<Nd> fromi = fs[0], sizei = fs[1];
                            fromi[i] = fromr0;
                            sizei[i] = sizer0;
                            q.push_back({fromi, sizei});
                        }
                        if (sizer1 > 0) {
                            Coor<Nd> fromi = fs[0], sizei = fs[1];
                            fromi[i] = fromr1;
                            sizei[i] = sizer1;
                            q.push_back({fromi, sizei});
                        }
                    }
                    r = q;
                }
            }

            return r;
        }

        /// Return the intersection between two ranges in a periodic lattice
        /// \param from0: first coordinate of the first range
        /// \param size0: size of the first range
        /// \param from1: first coordinate of the second range
        /// \param size1: size of the second range
        /// \param dim: size of lattice
        /// \param fromr: first coordinate of the first resulting range
        /// \param sizer: size of the first resulting range

        template <unsigned int Nd>
        void intersection(const Coor<Nd> &from0, const Coor<Nd> &size0, const Coor<Nd> &from1,
                          const Coor<Nd> &size1, const Coor<Nd> &dim, Coor<Nd> &fromr,
                          Coor<Nd> &sizer) {
            From_size<Nd> fs = intersection<Nd>(from0, size0, from1, size1, dim);
            if (fs.size() == 0) {
                fromr = fill_coor<Nd>(0);
                sizer = fill_coor<Nd>(0);
            } else if (fs.size() == 1) {
                fromr = fs[0][0];
                sizer = fs[0][1];
            } else {
                throw std::runtime_error("Not supported complex overlap of intervals");
            }
        }

        /// Translate a range from one coordinate lattice to another
        /// \param rfrom0: first coordinate of the range to translate
        /// \param rsize0: size of the range to translate
        /// \param from0: origin coordinate on the origin lattice
        /// \param dim0: dimensions of the origin lattice
        /// \param from1: origin coordinate on the destination lattice
        /// \param dim1: dimensions of the destination lattice
        /// \param perm: permutation of the coordinates
        /// \param fromr: first coordinate of input range into the destination lattice
        /// \param sizer: size of the input range on the destination lattice

        template <std::size_t Nd0, std::size_t Nd1>
        void translate_range(const Coor<Nd0> &rfrom0, const Coor<Nd0> &rsize0,
                             const Coor<Nd0> &from0, const Coor<Nd0> &dim0, const Coor<Nd1> &from1,
                             const Coor<Nd1> &dim1, const Coor<Nd1> perm, Coor<Nd1> &fromr,
                             Coor<Nd1> &sizer) {
            fromr = normalize_coor<Nd1>(
                reorder_coor<Nd0, Nd1>(normalize_coor<Nd0>(rfrom0 - from0 + dim0, dim0), perm) +
                    from1,
                dim1);
            sizer = reorder_coor<Nd0, Nd1>(rsize0, perm);
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param indices0: coordinate in origin tensor that are going to send to each process
        /// \param indices1: coordinate in destination tensor that are going to receive from each process
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <unsigned int Nd0, unsigned int Nd1>
        From_size<Nd0> get_indices_to_send(const From_size<Nd0> &p0, unsigned int from_rank,
                                           const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                           const Coor<Nd0> &size0, const From_size<Nd1> &p1,
                                           unsigned int componentId1, unsigned int ncomponents1,
                                           const Order<Nd1> &o1, const Coor<Nd1> &from1) {
            // Get the global dimensions of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Restrict the local range in v0 to the range from0, size0
            Coor<Nd0> local_from0 = p0[from_rank][0];
            Coor<Nd0> local_size0 = p0[from_rank][1];
            Coor<Nd0> rlocal_from0, rlocal_size0;
            intersection<Nd0>(from0, size0, local_from0, local_size0, dim0, rlocal_from0,
                              rlocal_size0);

            // Translate the restricted range to the destination lattice
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> rfrom1, rsize1;
            translate_range(rlocal_from0, rlocal_size0, from0, dim0, from1, dim1, perm0, rfrom1,
                            rsize1);

            // Compute the indices
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            unsigned int nprocs = p1.size() / ncomponents1;
            From_size<Nd0> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd1> &local_from1 = p1[i * ncomponents1 + componentId1][0];
                const Coor<Nd1> &local_size1 = p1[i * ncomponents1 + componentId1][1];
                Coor<Nd1> fromi, sizei;
                intersection<Nd1>(rfrom1, rsize1, local_from1, local_size1, dim1, fromi, sizei);
                translate_range(fromi, sizei, from1, dim1, from0, dim0, perm1, r[i][0], r[i][1]);
            }

            return r;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: dimension labels for the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param indices0: coordinate in origin tensor that are going to send to each process
        /// \param indices1: coordinate in destination tensor that are going to receive from each process
        /// \param rank: rank of the current process
        /// \param nprocs: total number of processes
        /// \param cpu: device context

        template <unsigned int Nd0, unsigned int Nd1>
        From_size<Nd1> get_indices_to_receive(const From_size<Nd0> &p0, const Order<Nd0> &o0,
                                              const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                                              const From_size<Nd1> &p1, unsigned int to_rank,
                                              const Order<Nd1> &o1, const Coor<Nd1> &from1) {
            // Get the global dimensions of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);

            // Check the compatibility of the tensors
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Restrict the local range in v1 to the range from1, size1
            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0); // size in the destination
            Coor<Nd1> local_from1 = p1[to_rank][0];
            Coor<Nd1> local_size1 = p1[to_rank][1];
            Coor<Nd1> rlocal_from1, rlocal_size1;
            intersection<Nd1>(from1, size1, local_from1, local_size1, dim1, rlocal_from1,
                              rlocal_size1);

            // Translate the restricted range to the origin lattice
            Coor<Nd0> perm1 = find_permutation<Nd1, Nd0>(o1, o0);
            Coor<Nd0> rfrom0, rsize0;
            translate_range(rlocal_from1, rlocal_size1, from1, dim1, from0, dim0, perm1, rfrom0,
                            rsize0);

            // Compute the indices
            unsigned int nprocs = p0.size();
            From_size<Nd1> r(nprocs);
            for (unsigned int i = 0; i < nprocs; ++i) {
                const Coor<Nd0> &local_from0 = p0[i][0];
                const Coor<Nd0> &local_size0 = p0[i][1];
                Coor<Nd0> fromi, sizei;
                intersection<Nd0>(rfrom0, rsize0, local_from0, local_size0, dim0, fromi, sizei);
                translate_range(fromi, sizei, from0, dim0, from1, dim1, perm0, r[i][0], r[i][1]);
            }

            return r;
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename Comm, typename XPU0,
                  typename XPU1, typename EWOp>
        void copy(const From_size<Nd0> &p0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                  const Order<Nd0> &o0, const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                  const From_size<Nd1> &p1, const Coor<Nd1> &from1, const Order<Nd1> &o1,
                  const Components_tmpl<Nd1, T, XPU0, XPU1> &v1, Comm comm, EWOp ewop) {

            // Check the dimensions of p0 and p1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            unsigned int ncomponents1 = v1.first.size() + v1.second.size();

            if (p0.size() != ncomponents0 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            if (p1.size() != ncomponents1 * comm.nprocs)
                throw std::runtime_error("Invalid number of elements in the tensor distribution");

            // Check the compatibility of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            if (!check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1))
                throw std::runtime_error("Invalid copy operation");

            // Split the work for each receiving component
            std::vector<Request> reqs;
            for (unsigned int i = 0; i < ncomponents1; ++i) {
                for (const Component<Nd1, T, XPU0> &c : v1.first) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T>(p0, from0, size0, o0, v0, p1, ncomponents1,
                                                         from1, o1, c, comm, ewop));
                }
                for (const Component<Nd1, T, XPU1> &c : v1.second) {
                    if (c.componentId == i)
                        reqs.push_back(copy<Nd0, Nd1, T>(p0, from0, size0, o0, v0, p1, ncomponents1,
                                                         from1, o1, c, comm, ewop));
                }
            }

            // Finish the request
            for (const Request &r : reqs) wait(r);
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        /// \param v0: data for the origin tensor
        /// \param p1: partitioning of the destination tensor in consecutive ranges
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param v1: data for the destination tensor
        /// \param comm: communicator context
        /// \param xpu: device context

        template <unsigned int Nd0, unsigned int Nd1, typename T, typename Comm, typename XPU0,
                  typename XPU1, typename XPU, typename EWOp>
        Request copy(const From_size<Nd0> &p0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                     const Order<Nd0> &o0, const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                     const From_size<Nd1> &p1, unsigned int ncomponents1, const Coor<Nd1> &from1,
                     const Order<Nd1> &o1, const Component<Nd1, T, XPU> &v1, Comm comm, EWOp ewop) {

            // Generate the list of subranges to send from each component from v0 to v1
            unsigned int ncomponents0 = v0.first.size() + v0.second.size();
            std::shared_ptr<std::vector<From_size<Nd0>>> toSend =
                std::make_shared<std::vector<From_size<Nd0>>>(ncomponents0);

            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                (*toSend)[v0.first[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                    p0, comm.rank * ncomponents0 + v0.first[i].componentId, o0, from0, size0, p1,
                    v1.componentId, ncomponents1, o1, from1);
            }
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                (*toSend)[v0.second[i].componentId] = get_indices_to_send<Nd0, Nd1>(
                    p0, comm.rank * ncomponents0 + v0.second[i].componentId, o0, from0, size0, p1,
                    v1.componentId, ncomponents1, o1, from1);
            }

            // Generate the list of subranges to receive from each component from v0 to v1
            std::shared_ptr<From_size<Nd1>> toReceive =
                std::make_shared<From_size<Nd1>>(get_indices_to_receive<Nd0, Nd1>(
                    p0, o0, from0, size0, p1, v1.componentId + comm.rank * ncomponents1, o1,
                    from1));

            // Do the sending and receiving
            Request mpi_req = send_receive<Nd0, Nd1>(o0, *toSend, v0, o1, toReceive, v1,
                                                     ncomponents1, comm, ewop);

            // Do the local copies
            Request local_req = [=] {
                unsigned int ncomponents0 = v0.first.size() + v0.second.size();
                for (const Component<Nd0, const T, XPU0> &c0 : v0.first) {
                    local_copy<Nd0, Nd1, T>(
                        o0, (*toSend)[c0.componentId][v1.componentId + comm.rank * ncomponents1][0],
                        (*toSend)[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        c0.dim, c0.it, c0.xpu, o1,
                        (*toReceive)[c0.componentId + comm.rank * ncomponents0][0], v1.dim, v1.it,
                        v1.xpu, ewop);
                }
                for (const Component<Nd0, const T, XPU1> &c0 : v0.second) {
                    local_copy<Nd0, Nd1, T, XPU1>(
                        o0, (*toSend)[c0.componentId][v1.componentId + comm.rank * ncomponents1][0],
                        (*toSend)[c0.componentId][v1.componentId + comm.rank * ncomponents1][1],
                        c0.dim, c0.it, c0.xpu, o1,
                        (*toReceive)[c0.componentId + comm.rank * ncomponents0][0], v1.dim, v1.it,
                        v1.xpu, ewop);
                }
            };

            return [=] {
                wait(local_req);
                wait(mpi_req);
            };
        }

        /// Copy the content of plural tensor v0 into v1
        /// \param p0: partitioning of the origin tensor in consecutive ranges
        /// \param mpicomm: MPI communicator context
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of elements to copy in each dimension
        template <unsigned int Nd, typename Comm>
        void check_from_size(const From_size<Nd> &p, int ncomponents, Comm comm,
                             const char *var_name = "") {
            if (p.size() != ncomponents * comm.nprocs) {
                std::stringstream s;
                s << "Invalid number of elements in tensor distribution " << var_name << "; it is "
                  << p.size() << " and it should be " << ncomponents * comm.nprocs;
                throw std::runtime_error(s.str());
            }
        }

        /// Return value for the dimensions in o_r matching the given for o0 and o1
        template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo>
        Coor<Ndo> get_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                                 const Coor<Nd1> &dim1, const Order<Ndo> &o_r) {
            std::map<char, IndexType> m;
            for (unsigned int i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (unsigned int i = 0; i < Nd1; ++i) m[o1[i]] = dim1[i];
            Coor<Ndo> r;
            for (unsigned int i = 0; i < Ndo; ++i) r[i] = m[o_r[i]];
            return r;
        }

        /// Get the output partition
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param o0: dimension labels for the first operator
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param o1: dimension labels for the second operator
        /// \param o_r: dimension labels for the output operator

        template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo>
        From_size<Ndo> get_output_partition(const From_size<Nd0> &p0, const Order<Nd0> &o0,
                                             const From_size<Nd1> &p1, const Order<Nd1> &o1,
                                             const Order<Ndo> &o_r) {
            assert(p0.size() == p1.size());
            From_size<Ndo> pr(p0.size());

            for (unsigned int i = 0; i < p0.size(); ++i) {
                pr[i][0] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][0], o1, p1[i][0], o_r);
                pr[i][1] = get_dimensions<Nd0, Nd1, Ndo>(o0, p0[i][1], o1, p1[i][1], o_r);
            }

            return pr;
        }

        /// Contract two tensors
        /// \param p0: partitioning of the first origin tensor in consecutive ranges
        /// \param ncomponents0: number of consecutive components in each MPI rank
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param ctx0: context for each data pointer in v0
        /// \param p1: partitioning of the second origin tensor in consecutive ranges
        /// \param ncomponents1: number of consecutive components in each MPI rank
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param ctx1: context for each data pointer in v1
        /// \param pr: partitioning of the resulting tensor in consecutive ranges
        /// \param ncomponentsr: number of consecutive components in each MPI rank
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param vr: data for the second operator
        /// \param ctxr: context for each data pointer in vr
        ///
        /// The order of the labels should be as following:
        ///
        /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
        /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
        /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
        /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

        template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename T, typename Comm,
                  typename XPU0, typename XPU1>
        void contraction(const From_size<Nd0> &p0, const Order<Nd0> &o0, bool conj0,
                         const Components_tmpl<Nd0, const T, XPU0, XPU1> &v0,
                         const From_size<Nd0> &p1, const Order<Nd1> &o1, bool conj1,
                         const Components_tmpl<Nd1, const T, XPU0, XPU1> &v1,
                         const From_size<Ndo> &pr, const Order<Ndo> &o_r,
                         const Components_tmpl<Ndo, T, XPU0, XPU1> &vr, Comm comm) {

            // Check the compatibility of the tensors
            Coor<Nd0> dim0 = get_dim<Nd0>(p0);
            Coor<Nd1> dim1 = get_dim<Nd1>(p1);
            Coor<Ndo> dimr = get_dim<Ndo>(pr);
            if (!check_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, dimr))
                throw std::runtime_error("some dimension does not match");

            // Check that v0 and v1 have the same components and on the same device
            if (v0.first.size() != v1.first.size() || v0.second.size() != v1.second.size())
                throw std::runtime_error(
                    "the two input tensors should have the same number of components");
            bool unmatch_dev = false;
            for (unsigned int i = 0; i < v0.first.size(); ++i)
                if (deviceId(v0.first[i].xpu) != deviceId(v1.first[i].xpu)) unmatch_dev = true;
            for (unsigned int i = 0; i < v0.second.size(); ++i)
                if (deviceId(v0.second[i].xpu) != deviceId(v1.second[i].xpu)) unmatch_dev = true;
            if (unmatch_dev)
                throw std::runtime_error(
                    "Each component of the input tensors should be on the same device");

            // Generate the partitioning and the storage for the output tensor
            unsigned int ncomponents = v0.first.size() + v1.second.size();
            From_size<Ndo> pr_ = get_output_partition<Nd0, Nd1, Ndo>(p0, o0, p1, o1, o_r);
            Components_tmpl<Ndo, const T, XPU0, XPU1> vr_;
            std::vector<vector<T, XPU0>> vr0(v0.first.size());
            for (unsigned int i = 0; i < v0.first.size(); ++i) {
                const unsigned int componentId = v0.first[i].componentId;
                const unsigned int pi = comm.rank * ncomponents + componentId;
                const Coor<Ndo> &dimi = pr_[pi][1];
                vr0[i].resize(volume<Ndo>(dimi));
                vr_.first.push_back(
                    Component<Ndo, T, XPU0>{vr0[i].data(), dimi, v0.first[i].xpu, componentId});
                local_contraction<Nd0, Nd1, Ndo>(o0, p0[pi][1], conj0, v0.first[i].it, o1,
                                                 p1[pi][1], conj1, v1.first[i].it, o_r, dimi,
                                                 vr0[i].data(), v0.first[i].xpu);
            }
            std::vector<vector<T, XPU1>> vr1(v0.second.size());
            for (unsigned int i = 0; i < v0.second.size(); ++i) {
                const unsigned int componentId = v0.second[i].componentId;
                const Coor<Ndo> &dimi = pr_[comm.rank * ncomponents + componentId][1];
                vr1[i].resize(volume<Ndo>(dimi));
                vr_.second.push_back(
                    Component<Ndo, T, XPU1>{vr1[i].data(), dimi, v0.second[i].xpu, componentId});
            }

            // Reduce all the subtensors to the final tensor
            const Coor<Ndo> zeros = fill_coor<Ndo>(0);
            copy<Ndo, Ndo, T>(pr_, zeros, dimr, o_r, vr_, pr, zeros, o_r, vr, comm, EWOp::Add{});
        }
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Copy the content of plural tensor v0 into v1
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param mpicomm: MPI communicator context
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void copy(const From_size<Nd0> &p0, int ncomponents0, const char *o0, const Coor<Nd0> &from0,
              const Coor<Nd0> &size0, const T **v0, Context *ctx0, const From_size<Nd1> &p1,
              int ncomponents1, const char *o1, const Coor<Nd1> &from1, T **v1, Context *ctx1,
              MPI_Comm mpicomm) {

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::check_from_size<Nd0>(p0, ncomponents0, comm, "p0");
        detail::check_from_size<Nd1>(p1, ncomponents1, comm, "p1");

        detail::copy<Nd0, Nd1>(p0, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                               detail::get_components<Nd0>(v0, ctx0, ncomponents0,
                                                           p0.begin() + comm.rank * ncomponents0),
                               p1, from1, detail::toArray<Nd1>(o1, "o1"),
                               detail::get_components<Nd1>(v1, ctx1, ncomponents1,
                                                           p1.begin() + comm.rank * ncomponents1),
                               comm, detail::EWOp::Copy{});
    }
#endif // SUPERBBLAS_USE_MPI

    /// Copy the content of plural tensor v0 into v1
    /// \param p0: partitioning of the origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of elements to copy in each dimension
    /// \param v0: vector of data pointers for the origin tensor
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the destination tensor in consecutive ranges
    /// \param o1: dimension labels for the destination tensor
    /// \param dim1: dimension size for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param v1: vector of data pointers for the origin tensor
    /// \param ctx1: context for each data pointer in v1

    template <unsigned int Nd0, unsigned int Nd1, typename T>
    void copy(const From_size<Nd0> &p0, int ncomponents0, const char *o0, const Coor<Nd0> from0,
              const Coor<Nd0> size0, const T **v0, Context *ctx0, const From_size<Nd1> &p1,
              int ncomponents1, const char *o1, const Coor<Nd1> from1, T **v1, Context *ctx1) {

        detail::SelfComm comm = detail::get_comm();
        detail::check_from_size<Nd0>(p0, ncomponents0, comm, "p0");
        detail::check_from_size<Nd1>(p1, ncomponents1, comm, "p1");

        detail::copy<Nd0, Nd1>(p0, from0, size0, detail::toArray<Nd0>(o0, "o0"),
                               detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0.begin()), p1,
                               from1, detail::toArray<Nd1>(o1, "o1"),
                               detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1.begin()),
                               comm, detail::EWOp::Copy{});
    }

#ifdef SUPERBBLAS_USE_MPI
    /// Contract two tensors
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    ///
    /// The order of the labels should be as following:
    ///
    /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename T>
    void contraction(const From_size<Nd0> &p0, int ncomponents0, const char *o0, bool conj0,
                     const T **v0, Context *ctx0, const From_size<Nd1> &p1, int ncomponents1,
                     const char *o1, bool conj1, const T **v1, Context *ctx1,
                     const From_size<Ndo> &pr, int ncomponentsr, const char *o_r, T **vr,
                     Context *ctxr, MPI_Comm mpicomm) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::MpiComm comm = detail::get_comm(mpicomm);

        detail::check_from_size<Nd0>(p0, ncomponents0, comm, "p0");
        detail::check_from_size<Nd1>(p1, ncomponents1, comm, "p1");
        detail::check_from_size<Ndo>(pr, ncomponentsr, comm, "pr");

        detail::contraction<Nd0, Nd1, Ndo>(
            p0, o0_, conj0, detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0.begin()), p1,
            o1_, conj1, detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1.begin()), pr, o_r_,
            detail::get_components<Ndo>(vr, ctxr, ncomponentsr, pr.begin()), comm);
    }
#endif // SUPERBBLAS_USE_MPI

    /// Contract two tensors
    /// \param p0: partitioning of the first origin tensor in consecutive ranges
    /// \param ncomponents0: number of consecutive components in each MPI rank
    /// \param o0: dimension labels for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param ctx0: context for each data pointer in v0
    /// \param p1: partitioning of the second origin tensor in consecutive ranges
    /// \param ncomponents1: number of consecutive components in each MPI rank
    /// \param o1: dimension labels for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param ctx1: context for each data pointer in v1
    /// \param pr: partitioning of the resulting tensor in consecutive ranges
    /// \param ncomponentsr: number of consecutive components in each MPI rank
    /// \param o_r: dimension labels for the output operator
    /// \param vr: data for the second operator
    /// \param ctxr: context for each data pointer in vr
    ///
    /// The order of the labels should be as following:
    ///
    /// - if !conj0 && !conj1, then (T,A,B) x (T,C,A) -> (T,C,B)
    /// - if conj0 && !conj1,  then (T,B,A) x (T,C,A) -> (T,C,B)
    /// - if !conj0 && conj1,  then (T,A,B) x (T,A,C) -> (T,C,B)
    /// - if conj0 && conj1,   then (T,B,A) x (T,A,C) -> (T,C,B)

    template <unsigned int Nd0, unsigned int Nd1, unsigned int Ndo, typename T>
    void contraction(const From_size<Nd0> &p0, int ncomponents0, const char *o0, bool conj0,
                     const T **v0, Context *ctx0, const From_size<Nd1> &p1, int ncomponents1,
                     const char *o1, bool conj1, const T **v1, Context *ctx1,
                     const From_size<Ndo> &pr, int ncomponentsr, const char *o_r, T **vr,
                     Context *ctxr) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        detail::SelfComm comm = detail::get_comm();

        detail::check_from_size<Nd0>(p0, ncomponents0, comm, "p0");
        detail::check_from_size<Nd1>(p1, ncomponents1, comm, "p1");
        detail::check_from_size<Ndo>(pr, ncomponentsr, comm, "pr");

        detail::contraction<Nd0, Nd1, Ndo>(
            p0, o0_, conj0, detail::get_components<Nd0>(v0, ctx0, ncomponents0, p0.begin()), p1,
            o1_, conj1, detail::get_components<Nd1>(v1, ctx1, ncomponents1, p1.begin()), pr, o_r_,
            detail::get_components<Ndo>(vr, ctxr, ncomponentsr, pr.begin()), comm);
    }
}

#endif //  __SUPERBBLAS_DIST__