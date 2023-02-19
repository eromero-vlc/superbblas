#ifndef __SUPERBBLAS_TENSOR__
#define __SUPERBBLAS_TENSOR__

#include "cache.h"
#include "copy_n.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

//////////////////////
// NOTE:
// Functions in this file that uses `thrust` should be instrumented to remove the dependency from
// `thrust` when the superbblas library is used not as header-only. Use the macro `IMPL` to hide
// the definition of functions using `thrust` and use DECL_... macros to generate template
// instantiations to be included in the library.

#ifdef SUPERBBLAS_CREATING_LIB

#    define COOR_DIMS                                                                              \
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36

#    ifdef SUPERBBLAS_USE_GPU
#        define XPUS detail::Cpu, detail::Gpu
#    else
#        define XPUS detail::Cpu
#    endif

/// Generate template instantiations for get_permutation function with template parameters IndexType and Nd

#    define DECL_PERM(...)                                                                         \
        EMIT REPLACE1(get_permutation, superbblas::detail::get_permutation<IndexType, Nd>)         \
            REPLACE_IndexType REPLACE(Nd, COOR_DIMS) template __VA_ARGS__;

/// Generate template instantiations for local_copy_normalize functions with template parameters T and Q

#    define DECL_LOCAL_COPY_NORMALIZE_T_Q(...)                                                     \
        EMIT REPLACE1(                                                                             \
            local_copy_normalize,                                                                  \
            superbblas::detail::local_copy_normalize<IndexType, Nd, T, Q, XPU0, XPU1, EWOP>)       \
            REPLACE_IndexType REPLACE(Nd, COOR_DIMS) REPLACE_T_Q REPLACE(XPU0, XPUS)               \
                REPLACE(XPU1, XPUS) REPLACE_EWOP template __VA_ARGS__;
#else
#    define DECL_PERM(...) __VA_ARGS__
#    define DECL_LOCAL_COPY_NORMALIZE_T_Q(...) __VA_ARGS__
#endif

namespace superbblas {

    /// Coordinate Index type
    using IndexType = int;
    /// Coordinate type
    template <std::size_t Nd, typename Idx = IndexType> using Coor = std::array<Idx, Nd>;
    /// Vector of dimension labels
    template <std::size_t Nd> using Order = std::array<char, Nd>;
    /// Mask/boolean element: use a type that work with BLAS
    using MaskType = float;

    /// How the coordinates are translates into positions in the tensor
    enum CoorOrder {
        SlowToFast, ///< The first coordinate runs the slowest and the last runs the fastest
        FastToSlow  ///< The first coordinate runs the fastest and the first runs the slowest
    };

    /// Action on the destination elements
    enum CopyAdd {
        Copy, ///< Copy the origin values into the destination tensor
        Add   ///< Add the origin values into the destination tensor
    };

    namespace detail {

#ifdef SUPERBBLAS_USE_THRUST

        /// Thrust does not support std::array container; here we implement a quick-and-dirty array container based on tuples

        template <typename T, std::size_t N> struct tarray;
        template <typename T, std::size_t N> struct tarray {
            static const std::size_t size_left = (N + 1) / 2;
            static const std::size_t size_right = N - size_left;
            tarray<T, size_left> left;
            tarray<T, size_right> right;
        };
        template <typename T> struct tarray<T, 0ul> {};
        template <typename T> struct tarray<T, 1ul> { T leaf; };

        /// Return the I-th element on a tarray
        /// \tparam I: index of the element to return
        /// \param t: input array

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I > 0 && I < N), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return (I < t.size_left ? tget<I>(t.left) : tget<I - t.size_left>(t.right));
        }

        template <std::size_t I, typename T, std::size_t N,
                  typename std::enable_if<(I == 0 && N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T &tget(tarray<T, N> &t) {
            return t.leaf;
        }

        /// Return the i-th element on a tarray
        /// \param i: index of the element to return
        /// \param t: input array

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N > 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i < Indx(t.size_left) ? tget(i, t.left) : tget(i - (Indx)t.size_left, t.right));
        }

        template <typename T, typename Indx, std::size_t N,
                  typename std::enable_if<(N == 1), bool>::type = true>
        inline __HOST__ __DEVICE__ T tget(Indx i, const tarray<T, N> &t) {
            return (i == 0 ? t.leaf : T{0});
        }

        /// Coordinate based on tarray
        /// \tparam Nd: number of dimensions

        template <std::size_t Nd, typename Idx = IndexType> using TCoor = tarray<Idx, Nd>;
#endif

        /// Vector of `IndexType`
        template <typename XPU> using Indices = vector<IndexType, XPU>;
        template <typename IndexType, typename XPU> using IndicesT = vector<IndexType, XPU>;

        /// Mask vector
        template <typename XPU> using Mask = vector<MaskType, XPU>;

        //
        // Auxiliary functions
        //

        template <typename T, std::size_t Na, std::size_t Nb,
                  typename std::enable_if<Na != Nb, bool>::type = true>
        bool operator==(const std::array<T, Na> &, const std::array<T, Nb> &) {
            return false;
        }

        template <typename T, std::size_t N>
        std::array<T, N> operator-(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = a[i] - b[i];
            return r;
        }

        template <typename T, std::size_t N>
        std::array<T, N> operator/(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = a[i] / b[i];
            return r;
        }

        template <typename T, std::size_t N>
        bool all_less_or_equal(const std::array<T, N> &a, const std::array<T, N> &b) {
            for (std::size_t i = 0; i < N; i++)
                if (a[i] > b[i]) return false;
            return true;
        }

        template <typename T, std::size_t N>
        std::array<T, N> min_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = std::min(a[i], b[i]);
            return r;
        }

        template <typename T, std::size_t N>
        std::array<T, N> max_each(const std::array<T, N> &a, const std::array<T, N> &b) {
            std::array<T, N> r;
            for (std::size_t i = 0; i < N; i++) r[i] = std::max(a[i], b[i]);
            return r;
        }

        template <typename T, std::size_t N> std::array<T, N> reverse(const std::array<T, N> v) {
            std::array<T, N> r = v;
            std::reverse(r.begin(), r.end());
            return r;
        }

        template <std::size_t N> Coor<N> ones() {
            Coor<N> r;
            for (auto &c : r) c = 1;
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_plus_aux {
            template <std::size_t Nd, typename std::enable_if<(Nd > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {plus_aux(a.left, b.left), plus_aux(a.right, b.right)};
            }

            template <std::size_t Nd, typename std::enable_if<(Nd == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd> plus_aux(const TCoor<Nd> &a,
                                                                 const TCoor<Nd> &b) {
                return {a.leaf + b.leaf};
            }
        };

        /// Add two arrays
        /// \param a: first array to add
        /// \param b: second array to add

        template <std::size_t Nd>
        __HOST__ __DEVICE__ inline TCoor<Nd> tplus(TCoor<Nd> a, TCoor<Nd> b) {
            return ns_plus_aux::plus_aux(a, b);
        }

        struct ns_toTCoor_aux {
            template <std::size_t I, std::size_t Nr, std::size_t N, typename IndexType,
                      typename std::enable_if<(I < N && 1 < Nr), bool>::type = true>
            static inline TCoor<Nr, IndexType> toTCoor_aux(const Coor<N, IndexType> &a) {
                const auto sl = TCoor<Nr, IndexType>::size_left;
                const auto sr = TCoor<Nr, IndexType>::size_right;
                return {toTCoor_aux<I, sl>(a), toTCoor_aux<I + sl, sr>(a)};
            }

            template <std::size_t I, std::size_t Nr, std::size_t N, typename IndexType,
                      typename std::enable_if<(I < N && 1 == Nr), bool>::type = true>
            static inline TCoor<Nr, IndexType> toTCoor_aux(const Coor<N, IndexType> &a) {
                return {a[I]};
            }
        };

        /// Convert from Coor to TCoor
        /// \param a: input coordinate

        template <std::size_t Nd, typename IndexType>
        inline TCoor<Nd, IndexType> toTCoor(const Coor<Nd, IndexType> &a) {
            return ns_toTCoor_aux::toTCoor_aux<0, Nd>(a);
        }
#endif

        /// Return whether the point is in the interval
        /// \param from: first coordinate in the interval
        /// \param size: number of consecutive elements in the interval in each direction
        /// \param dim: tensor dimensions
        /// \param coor: coordinate to evaluate whether it is in the interval

        template <std::size_t N, typename IndexType>
        bool is_in_interval(const Coor<N, IndexType> &from, const Coor<N, IndexType> &size,
                            const Coor<N, IndexType> &dim, const Coor<N, IndexType> &coor) {
            for (std::size_t i = 0; i < N; i++)
                if (!((from[i] <= coor[i] && coor[i] < from[i] + size[i]) ||
                      (from[i] <= coor[i] + dim[i] && coor[i] + dim[i] < from[i] + size[i])))
                    return false;
            return true;
        }

        /// Return an array from a string
        /// \param v: input string
        /// \param name: name of the variable

        template <std::size_t Nd, typename T>
        std::array<T, Nd> toArray(const T *v, const char *name) {
            if ((v == nullptr && Nd > 0) || std::strlen(v) != Nd) {
                std::stringstream ss;
                ss << "The length of the order should match the template argument; argument `"
                   << name << "` should have length " << Nd;
                throw std::runtime_error(ss.str());
            }
            std::array<T, Nd> r;
            std::copy_n(v, Nd, r.begin());
            return r;
        }

        /// Return the jumps to the next consecutive element in each dimension
        /// \param dim: lattice dimension
        /// \param co: coordinate linearization order

        template <typename SIdx, std::size_t Nd, typename CIdx>
        Coor<Nd, SIdx> get_strides(const Coor<Nd, CIdx> dim, CoorOrder co) {
            Coor<Nd, SIdx> p;
            if (Nd > 0) {
                if (co == SlowToFast) {
                    // p(i) = prod(dim(end:-1:i))
                    p.back() = 1;
                    for (std::size_t i = p.size() - 1; i >= 1; i--) p[i - 1] = p[i] * dim[i];
                } else {
                    // p(i) = prod(dim(1:i))
                    p[0] = 1;
                    for (std::size_t i = 1; i < Nd; ++i) p[i] = p[i - 1] * dim[i - 1];
                }
            }
            return p;
        }

        /// Return the index associated to a coordinate
        /// \param coors: input coordinate
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <std::size_t Nd, typename CIdx, typename SIdx>
        SIdx coor2index(const Coor<Nd, CIdx> &coor, const Coor<Nd, CIdx> &dim,
                        const Coor<Nd, SIdx> &stride) {
            IndexType r = 0;
            for (std::size_t j = 0; j < Nd; j++) r += (coor[j] % dim[j]) * stride[j];
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        template <std::size_t Nd, typename CIdx, typename SIdx,
                  typename std::enable_if<(Nd > 1), bool>::type = true>
        __HOST__ __DEVICE__ SIdx coor2index(const TCoor<Nd, CIdx> &coor, const TCoor<Nd, CIdx> &dim,
                                            const TCoor<Nd, SIdx> &stride) {
            return coor2index(coor.left, dim.left, stride.left) +
                   coor2index(coor.right, dim.right, stride.right);
        }

        template <std::size_t Nd, typename CIdx, typename SIdx,
                  typename std::enable_if<(Nd == 1), bool>::type = true>
        __HOST__ __DEVICE__ SIdx coor2index(const TCoor<Nd, CIdx> &coor, const TCoor<Nd, CIdx> &dim,
                                            const TCoor<Nd, SIdx> &stride) {
            return (coor.leaf % dim.leaf) * stride.leaf;
        }
#endif

        /// Return the coordinate associated to an index
        /// \param index: input vertex index
        /// \param dim: lattice dimensions
        /// \param stride: jump to get to the next coordinate in each dimension

        template <std::size_t Nd, typename CIdx, typename SIdx>
        inline Coor<Nd, CIdx> index2coor(const SIdx &index, const Coor<Nd, CIdx> &dim,
                                         const Coor<Nd, SIdx> &stride) {
            Coor<Nd, CIdx> r;
            for (std::size_t j = 0; j < Nd; j++) r[j] = (index / stride[j]) % (SIdx)dim[j];
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_index2coor_aux {
            template <std::size_t Nd, typename CIdx, typename SIdx,
                      typename std::enable_if<(Nd > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd, CIdx>
            index2coor(SIdx index, const TCoor<Nd, CIdx> &dim, const TCoor<Nd, SIdx> &stride) {
                return {index2coor(index, dim.left, stride.left),
                        index2coor(index, dim.right, stride.right)};
            }

            template <std::size_t Nd, typename CIdx, typename SIdx,
                      typename std::enable_if<(Nd == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd, CIdx>
            index2coor(SIdx index, const TCoor<Nd, CIdx> &dim, const TCoor<Nd, SIdx> &stride) {
                return {(CIdx)((index / stride.leaf) % (SIdx)dim.leaf)};
            }
        };

        template <std::size_t Nd, typename CIdx, typename SIdx>
        __HOST__ __DEVICE__ inline TCoor<Nd, CIdx>
        index2coor(SIdx index, const TCoor<Nd, CIdx> &dim, const TCoor<Nd, SIdx> &stride) {
            return ns_index2coor_aux::index2coor(index, dim, stride);
        }
#endif

        /// Check all dimension labels are distinct
        /// \param order: dimension labels
        ///
        /// Return whether all label dimension are distinct

        template <typename Vector> bool check_order(const Vector &order) {
            for (std::size_t i = 0; i < order.size(); ++i)
                if (std::find(order.begin() + i + 1, order.end(), order[i]) != order.end())
                    return false;
            return true;
        }

        /// Return the number of vertices in a lattice
        /// \param dim: lattice dimensions

        template <std::size_t Nd> std::size_t volume(const Coor<Nd> &dim) {
            if (dim.size() <= 0) return 0;

            std::size_t vol = dim[0];
            for (std::size_t i = 1; i < dim.size(); ++i) vol *= dim[i];
            return vol;
        }

        /// Return the number of vertices in a sublattice
        /// \param order: dimension labels
        /// \param dim: lattice dimensions
        /// \param starts_with: the first label of the sublattice
        /// \param size: number of consecutive dimension of the sublattice

        template <std::size_t Nd>
        std::size_t volume(typename Coor<Nd>::const_iterator begin,
                           typename Coor<Nd>::const_iterator end) {
            if (begin == end) return 0;

            std::size_t vol = 1;
            while (begin != end) {
                vol *= *begin;
                ++begin;
            }

            return vol;
        }

        /// Return a new array {coor[perm[0]], coor[perm[1]], ...}
        /// \param coor: input array
        /// \param perm: permutation
        /// \param black: value to set when perm[i] < 0
        ///
        /// NOTE: the output array will have zero on negative elements of `perm`.

        template <std::size_t Nd0, std::size_t Nd1>
        Coor<Nd1> reorder_coor(const Coor<Nd0> &coor, const Coor<Nd1> &perm, IndexType blanck = 0) {
            Coor<Nd1> r;
            for (std::size_t i = 0; i < Nd1; ++i) r[i] = perm[i] >= 0 ? coor[perm[i]] : blanck;
            return r;
        }

#ifdef SUPERBBLAS_USE_THRUST
        struct ns_reorder_coor_aux {
            template <std::size_t Nd0, std::size_t Nd1,
                      typename std::enable_if<(Nd1 > 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd1>
            reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck) {
                return {reorder_coor(coor, perm.left, blanck),
                        reorder_coor(coor, perm.right, blanck)};
            }

            template <std::size_t Nd0, std::size_t Nd1,
                      typename std::enable_if<(Nd1 == 1), bool>::type = true>
            static __HOST__ __DEVICE__ inline TCoor<Nd1>
            reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck) {
                return {(perm.leaf >= 0 ? tget(perm.leaf, coor) : blanck)};
            }
        };

        template <std::size_t Nd0, std::size_t Nd1>
        __HOST__ __DEVICE__ inline TCoor<Nd1>
        reorder_coor(const TCoor<Nd0> &coor, const TCoor<Nd1> &perm, IndexType blanck = 0) {
            return ns_reorder_coor_aux::reorder_coor(coor, perm, blanck);
        }
#endif

        /// Check that there exists a permutation from the first tensor to the second
        /// \param o0: dimension labels
        /// \param dim0: dimension size for o0
        /// \param o1: dimension labels
        ///
        /// Return whether all labels with dimension size greater than one in o0 are also in o1 and
        /// and the dimension of the first is smaller or equal than the second

        template <std::size_t Nd0, std::size_t Nd1>
        bool is_a_subset_of(Order<Nd0> o0, Coor<Nd0> dim0, Order<Nd1> o1) {
            for (std::size_t i = 0; i < o0.size(); ++i)
                if (dim0[i] > 1 && std::find(o1.begin(), o1.end(), o0[i]) == o1.end()) return false;
            return true;
        }

        /// Return a permutation that transform an o0 coordinate into an o1 coordinate
        /// \param o0: source dimension labels
        /// \param o1: destination dimension labels
        ///
        /// NOTE: the permutation can be used in function `reorder_coor`.

        template <std::size_t Nd0, std::size_t Nd1>
        Coor<Nd1> find_permutation(const Order<Nd0> &o0, const Order<Nd1> &o1) {
            Coor<Nd1> r;
            for (std::size_t i = 0; i < Nd1; ++i) {
                const auto j = std::find(o0.begin(), o0.end(), o1[i]);
                r[i] = (j != o0.end() ? j - o0.begin() : -1);
            }
            return r;
        }

        /// Check that all values are positive
        /// \param from: coordinates to check

        template <std::size_t Nd> bool check_positive(const Coor<Nd> &from) {
            return all_less_or_equal({}, from);
        }

        /// Check that the copy operation is possible
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: first coordinate not to copy from the origin tensor
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param dim1: dimension size for the destination tensor

        template <std::size_t Nd0, std::size_t Nd1>
        bool check_isomorphic(const Order<Nd0> &o0, const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                              const Order<Nd1> &o1, const Coor<Nd1> dim1) {

            if (!(check_order(o0) && check_order(o1) && check_positive<Nd0>(size0) &&
                  all_less_or_equal(size0, dim0) && is_a_subset_of<Nd0, Nd1>(o0, size0, o1)))
                return false;
            if (volume(size0) == 0) return true;

            Coor<Nd1> perm0 = find_permutation<Nd0, Nd1>(o0, o1);
            Coor<Nd1> size1 = reorder_coor<Nd0, Nd1>(size0, perm0, 1);
            return all_less_or_equal(size1, dim1);
        }

        //
        // Hash for tuples and arrays
        //

        template <typename T> struct Hash {
            template <typename U = T,
                      typename std::enable_if<!std::is_enum<U>::value, bool>::type = true>
            static std::size_t hash(U const &t) noexcept {
                return std::hash<T>{}(t);
            }
            template <typename U = T,
                      typename std::enable_if<std::is_enum<U>::value, bool>::type = true>
            static std::size_t hash(T const &t) noexcept {
                return std::size_t(t);
            }
        };

        template <typename T> struct Hash<const T> {
            static std::size_t hash(T const &t) noexcept { return Hash<T>::hash(t); }
        };

        /// Extend hash to std::array
        template <typename T, std::size_t N> struct Hash<std::array<T, N>> {
            static std::size_t hash(std::array<T, N> const &t) noexcept {
                std::size_t r = 12345;
                for (std::size_t i = 0; i < N; ++i) r = r ^ Hash<T>::hash(t[i]);
                return r;
            }
        };

        /// Extend hash to std::complex
        template <typename T> struct Hash<std::complex<T>> {
            static std::size_t hash(std::complex<T> const &t) noexcept {
                return Hash<std::array<T, 2>>::hash(std::array<T, 2>{std::real(t), std::imag(t)});
            }
        };

        template <class Tuple> struct TupleHash;

        /// Extend Hash for std::tuple

        template <typename... Ts> struct Hash<std::tuple<Ts...>> {
            static std::size_t hash(std::tuple<Ts...> const &t) noexcept {
                return TupleHash<std::tuple<Ts...>>{}(t);
            }
        };

        /// Extend Hash for vector<T, Cpu>

        template <typename T> struct Hash<vector<T, Cpu>> {
            static std::size_t hash(vector<T, Cpu> const &t) noexcept {
                std::size_t r = 12345;
                for (std::size_t i = 0; i < t.size(); ++i) r = r ^ Hash<T>::hash(t[i]);
                return r;
            }
        };

        /// Extend Hash for std::vector<T>

        template <typename T> struct Hash<std::vector<T>> {
            static std::size_t hash(std::vector<T> const &t) noexcept {
                std::size_t r = 12345;
                for (std::size_t i = 0; i < t.size(); ++i) r = r ^ Hash<T>::hash(t[i]);
                return r;
            }
        };

        template <class Tuple, std::size_t N> struct TupleHashHelp {
            static std::size_t hash(Tuple const &t) noexcept {
                return Hash<typename std::tuple_element<N, Tuple>::type>::hash(std::get<N>(t)) ^
                       TupleHashHelp<Tuple, N - 1>::hash(t);
            }
        };

        template <class Tuple> struct TupleHashHelp<Tuple, 0> {
            static std::size_t hash(Tuple const &t) noexcept {
                return Hash<typename std::tuple_element<0, Tuple>::type>::hash(std::get<0>(t));
            }
        };

        /// Hash for tuples

        template <class... TupleItems> struct TupleHash<typename std::tuple<TupleItems...>> {
            using Tuple = typename std::tuple<TupleItems...>;
            std::size_t operator()(Tuple const &t) const noexcept {
                return TupleHashHelp<Tuple, std::tuple_size<Tuple>::value - 1>::hash(t);
            }
        };

        template <typename T> struct TupleHash<vector<T, Cpu>> {
            using type = vector<T, Cpu>;
            std::size_t operator()(type const &t) const noexcept { return Hash<type>::hash(t); }
        };

        /// Return the memory footprint of an object
        /// \param v: input object

        template <typename T, typename XPU> std::size_t storageSize(const vector<T, XPU> &v) {
            return sizeof(T) * v.size();
        }

        /// Check that all dimensions with the same label has the same size
        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        bool check_dimensions(const Order<Nd0> &o0, const Coor<Nd0> &dim0, const Order<Nd1> &o1,
                              const Coor<Nd1> &dim1, const Order<Ndo> &o_r, const Coor<Ndo> &dimr) {
            std::map<char, IndexType> m;
            for (std::size_t i = 0; i < Nd0; ++i) m[o0[i]] = dim0[i];
            for (std::size_t i = 0; i < Nd1; ++i) {
                auto it = m.find(o1[i]);
                if (it != m.end()) {
                    if (it->second != dim1[i]) return false;
                } else {
                    m[o1[i]] = dim1[i];
                }
            }
            for (std::size_t i = 0; i < Ndo; ++i) {
                auto it = m.find(o_r[i]);
                if (it != m.end()) {
                    if (it->second != dimr[i]) return false;
                } else {
                    m[o_r[i]] = dimr[i];
                }
            }
            return true;
        }

#ifdef SUPERBBLAS_USE_GPU
        /// Return a copy of the given tensor with an allocation stream suitable to be stored
        /// in cache
        /// \param v: vector to store
        ///
        /// NOTE: the allocation streams are the ones that live forever, while the regular
        /// streams can come from coflow and be destroy anytime.

        template <typename T> vector<T, Gpu> archive(const vector<T, Gpu> &v) {
            if (getStream(v.ctx()) == getAllocStream(v.ctx())) return v;
            vector<T, Gpu> r(v.size(), v.ctx().withNewStream(getAllocStream(v.ctx())));
            copy_n(v.data(), v.ctx(), v.size(), r.data(), r.ctx());
            return r;
        }
#endif // SUPERBBLAS_USE_GPU

        template <typename T> vector<T, Cpu> archive(const vector<T, Cpu> &v) { return v; }

        /// Copy the content of tensor v0 into v1
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param co: coordinate linearization order
        /// \param new_disp0: (out) implicit index shift in the origin tensor
        /// \param new_from0: (out) first coordinate to copy after the implicit shift for the origin tensor
        /// \param new_size: (out) number of coordinates to copy in each dimension
        /// \param new_dim0: (out) origin tensor size
        /// \param new_strides0: (out) strides for the origin tensor
        /// \param new_disp1: (out) implicit index shift in the destination tensor
        /// \param new_from1: (out) first coordinate to copy after the implicit shift for the destination tensor
        /// \param new_dim1: (out) destination tensor size
        /// \param new_strides0: (out) strides for the destination tensor
        /// \param nblock: (out) the first `nblock` dimensions are equivalent to a trivial permutation
        ///
        /// This function translates the copy of a subtensor into another subtensor with possibly different
        /// ordering and number of dimensions into the copy of a subtensor into another one with the same
        /// number of dimensions. The origin and destination tensor dimensions are rearrange in order to
        /// coincided, and the `ordering` of each tensor is capture by taking a different element in the vector as
        /// the first tensor element (`new_disp0` and `new_disp`) and the `strides`. We only need to consider the
        /// common dimensions between the origin and the destination tensors in the strides. The other dimensions
        /// are captured by the initial displacements, `new_disp0` and `new_disp`.

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1,
                  std::size_t Nd = std::min(Nd0, Nd1)>
        void copy_normalize(const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                            const Coor<Nd0> &dim0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
                            const Coor<Nd1> &dim1, CoorOrder co,
                            // outputs
                            IndexType &new_disp0, Coor<Nd> &new_from0, Coor<Nd> &new_size,
                            Coor<Nd> &new_dim0, Coor<Nd, IndexType> &new_strides0,
                            IndexType &new_disp1, Coor<Nd> &new_from1, Coor<Nd> &new_dim1,
                            Coor<Nd, IndexType> &new_strides1, std::size_t &nblock) {

            // Normalize to FastToSlow
            if (co == SlowToFast) {
                copy_normalize(reverse(o0), reverse(from0), reverse(size0), reverse(dim0),
                               reverse(o1), reverse(from1), reverse(dim1), FastToSlow, new_disp0,
                               new_from0, new_size, new_dim0, new_strides0, new_disp1, new_from1,
                               new_dim1, new_strides1, nblock);
                return;
            }

            // Check the compatibility of the tensors
            assert((check_positive<Nd0>(from0) && check_positive<Nd1>(from1)));
            assert((check_isomorphic<Nd0, Nd1>(o0, size0, dim0, o1, dim1)));

            // Quick exit for zero volume
            if (volume(size0) == 0) {
                new_disp0 = new_disp1 = nblock = 0;
                new_from0 = new_size = new_dim0 = new_from1 = new_dim1 = Coor<Nd>{};
                new_strides0 = new_strides1 = Coor<Nd, IndexType>{};
                return;
            }

            Coor<Nd1> size1 = reorder_coor(size0, find_permutation(o0, o1), 1);
            IndexType stride1 = 1;
            new_disp1 = 0;
            std::size_t i = 0;
            for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                if (size1[i1] > 1) {
                    if (from1[i1] + size1[i1] <= dim1[i1]) {
                        new_from1[i] = 0;
                        new_disp1 += from1[i1] * stride1;
                    } else {
                        new_from1[i] = from1[i1];
                    }
                    new_size[i] = size1[i1];
                    new_dim1[i] = dim1[i1];
                    new_strides1[i] = stride1;
                    ++i;
                } else {
                    new_disp1 += from1[i1] * stride1;
                }
                stride1 *= dim1[i1];
            }
            for (; i < Nd; ++i) {
                new_from1[i] = 0;
                new_size[i] = 1;
                new_dim1[i] = 1;
                new_strides1[i] = (i > 0 ? new_strides1[i - 1] : 1);
            }
            assert(volume(size0) == volume(new_size));

            Coor<Nd1> perm0 = find_permutation(o0, o1);
            Coor<Nd0, IndexType> strides0 = get_strides<IndexType>(dim0, FastToSlow);
            i = 0;
            new_disp0 = 0;
            for (std::size_t i1 = 0; i1 < Nd1; ++i1) {
                if (perm0[i1] < 0) continue;
                superbblas::IndexType i0 = perm0[i1];
                if (size0[i0] > 1) {
                    if (from0[i0] + size0[i0] <= dim0[i0]) {
                        new_from0[i] = 0;
                        new_disp0 += from0[i0] * strides0[i0];
                    } else {
                        new_from0[i] = from0[i0];
                    }
                    new_dim0[i] = dim0[i0];
                    new_strides0[i] = strides0[i0];
                    ++i;
                }
            }
            for (; i < Nd; ++i) {
                new_from0[i] = 0;
                new_dim0[i] = 1;
                new_strides0[i] = (i > 0 ? new_strides0[i - 1] : 1);
            }

            for (std::size_t i0 = 0; i0 < Nd0; ++i0)
                if (size0[i0] == 1) new_disp0 += from0[i0] * strides0[i0];

            nblock = 0;
            IndexType strides = 1;
            for (std::size_t i = 0; i < Nd; ++i) {
                if (new_from0[i] != 0 || new_from1[i] != 0 || strides != new_strides0[i] ||
                    strides != new_strides1[i] || new_size[i] != new_dim0[i] ||
                    new_size[i] != new_dim1[i])
                    break;
                nblock++;
                strides *= new_size[i];
            }
        }

        /// Wether to allow returning a null pointer instead of the trivial permutation
        enum ImplicitPermutation {
            AllowImplicitPermutation,    ///< allow returning null pointers
            DontAllowImplicitPermutation ///< don't allow returning null pointer
        };

        /// Return the indices to copy
        /// \param from: first coordinate to copy
        /// \param size: number of coordinates to copy in each direction
        /// \param dim: dimension size
        /// \param strides: strides
        /// \param cpu: device context for the returned vector

        template <typename IndexType, std::size_t Nd>
        IndicesT<IndexType, Cpu> get_permutation(const Coor<Nd> &from, const Coor<Nd> &size,
                                                 const Coor<Nd> &dim,
                                                 const Coor<Nd, IndexType> &strides, Cpu cpu) {

            tracker<Cpu> _t("compute permutations", cpu);

            // Check inputs
            assert((check_positive<Nd>(from)));

            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(dim))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            // Quick exit
            IndexType vol = volume(size);
            if (volume(size) == 0) return IndicesT<IndexType, Cpu>();

            // Compute the permutation
            IndicesT<IndexType, Cpu> indices(vol, cpu);
            Coor<Nd, IndexType> size_strides = get_strides<IndexType>(size, FastToSlow);
#ifdef _OPENMP
#    pragma omp parallel for schedule(static)
#endif
            for (IndexType i = 0; i < vol; ++i)
                indices[i] = coor2index(index2coor(i, size, size_strides) + from, dim, strides);

            return indices;
        }

#ifdef SUPERBBLAS_USE_THRUST

        /// Class that compute the origin permutation

        template <typename IndexType, std::size_t Nd>
        struct perm_elem : public thrust::unary_function<IndexType, IndexType> {
            const TCoor<Nd> from, size, dim;
            const TCoor<Nd, IndexType> size_strides, strides;
            perm_elem(TCoor<Nd> from, TCoor<Nd> size, TCoor<Nd> dim,
                      TCoor<Nd, IndexType> size_strides, TCoor<Nd, IndexType> strides)
                : from(from), size(size), dim(dim), size_strides(size_strides), strides(strides) {}

            __HOST__ __DEVICE__ IndexType operator()(IndexType i) {
                return coor2index(tplus(index2coor(i, size, size_strides), from), dim, strides);
            }
        };

        template <typename IndexType, std::size_t Nd>
        IndicesT<IndexType, Gpu>
        get_permutation_thrust(const Coor<Nd> &from, const Coor<Nd> &size, const Coor<Nd> &dim,
                               const Coor<Nd, IndexType> &strides, Gpu gpu) {

            // Compute the permutation
            IndexType vol = volume(size);
            IndicesT<IndexType, Gpu> indices(vol, gpu);
            Coor<Nd, IndexType> size_strides = get_strides<IndexType>(size, FastToSlow);

            thrust::transform(thrust_par_on(gpu), thrust::make_counting_iterator(IndexType(0)),
                              thrust::make_counting_iterator(IndexType(vol)),
                              encapsulate_pointer(indices.data()),
                              perm_elem<IndexType, Nd>(toTCoor(from), toTCoor(size), toTCoor(dim),
                                                       toTCoor(size_strides), toTCoor(strides)));
            return indices;
        }
#endif

#ifdef SUPERBBLAS_USE_GPU
        template <typename IndexType, std::size_t Nd>
        DECL_PERM(IndicesT<IndexType, Gpu> get_permutation(
            const Coor<Nd> &from, const Coor<Nd> &size, const Coor<Nd> &dim,
            const Coor<Nd, IndexType> &strides, Gpu gpu))
        IMPL({
            tracker<Gpu> _t("compute permutations", gpu);

            // Check inputs
            assert((check_positive(from)));

            // Quick exit
            if (volume(size) == 0) return IndicesT<IndexType, Gpu>();

            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(dim))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            // Compute the permutation
            return get_permutation_thrust<IndexType, Nd>(from, size, dim, strides, gpu);
        })
#endif

        /// Return the indices to copy
        /// \param from: first coordinate to copy
        /// \param size: number of coordinates to copy in each direction
        /// \param dim: dimension size
        /// \param strides: strides
        /// \param implicitPermutation: whether to return a null pointer instead of the trivial permutation
        /// \param xpu: device context for the returned vector

        template <typename IndexType, std::size_t Nd, typename XPU>
        IndicesT<IndexType, XPU> get_permutation(const Coor<Nd> &from, const Coor<Nd> &size,
                                                 const Coor<Nd> &dim,
                                                 const Coor<Nd, IndexType> &strides,
                                                 ImplicitPermutation implicitPermutation, XPU xpu) {

            tracker<XPU> _t("get permutation", xpu);

            // Check inputs
            assert((check_positive<Nd>(from)));

            // Check that IndexType is big enough
            if ((std::size_t)std::numeric_limits<IndexType>::max() <= volume(dim))
                throw std::runtime_error("Ups! IndexType isn't big enough");

            // Quick exit
            IndexType vol = volume(size);
            if (volume(size) == 0) return IndicesT<IndexType, XPU>();
            Coor<Nd, IndexType> dim_strides = get_strides<IndexType>(dim, FastToSlow);
            if (implicitPermutation == AllowImplicitPermutation) {
                bool fail = true;
                for (std::size_t i = 0; i < Nd; ++i)
                    fail |= (from[i] != 0 || (size[i] > 1 && dim_strides[i] != strides[i]));
                if (!fail) return IndicesT<IndexType, XPU>(vol, nullptr, xpu);
            }

            // Check in the storage
            using Key = std::tuple<Coor<Nd>, Coor<Nd>, Coor<Nd>, Coor<Nd, IndexType>>;
            struct tag {};
            auto cache = getCache<Key, IndicesT<IndexType, XPU>, TupleHash<Key>, tag>(xpu);
            Key key{from, size, dim, strides};
            auto it = cache.find(key);
            if (it != cache.end()) return it->second.value;

            // Otherwise, compute the permutation
            IndicesT<IndexType, XPU> indices =
                get_permutation<IndexType>(from, size, dim, strides, xpu);

            // Store it in cache
            cache.insert(key, archive(indices), storageSize(indices));

            return indices;
        }

        /// Copy the content of tensor v0 into v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param mask0: mask for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param mask1: mask for the destination tensor (ignored)
        /// \param ewop: either to copy or to add the origin values into the destination values

        template <typename IndexType, std::size_t Nd, typename T, typename Q, typename XPU0,
                  typename XPU1, typename EWOP>
        DECL_LOCAL_COPY_NORMALIZE_T_Q(void local_copy_normalize(
            typename elem<T>::type alpha, IndexType disp0, const Coor<Nd> &from0,
            const Coor<Nd> &size, const Coor<Nd> &dim0, const Coor<Nd, IndexType> &strides0,
            vector<const T, XPU0> v0, Mask<XPU0> mask0, IndexType disp1, const Coor<Nd> &from1,
            const Coor<Nd> &dim1, const Coor<Nd, IndexType> &strides1, vector<Q, XPU1> v1,
            Mask<XPU1> mask1, std::size_t nblock, EWOP ewop))
        IMPL({
            // Get the permutation vectors
            Coor<Nd> sizeb = size;
            for (std::size_t i = 0; i < nblock; ++i) sizeb[i] = 1;

            // Shortcut for a trivial permutation
            if (volume(sizeb) == 1 && mask0.size() == 0) {
                IndexType extra_disp0 = coor2index(from0, dim0, strides0);
                IndexType extra_disp1 = coor2index(from1, dim1, strides1);
                copy_n<IndexType, T, Q>(alpha, v0.data() + disp0 + extra_disp0, v0.ctx(),
                                        volume(size), v1.data() + disp1 + extra_disp1, v1.ctx(),
                                        ewop);
                return;
            }

            // If using masks or there's no blocking, turn off blocking.
            // Also, performance reported by blas test shows that blocking in copy is worth it for
            // blocking at least 8
            std::size_t vol_sizeb = volume(sizeb);
            if (mask0.size() != 0 || vol_sizeb <= 1) {
                nblock = 0;
                sizeb = size;
            }
            IndicesT<IndexType, XPU0> indices0 = get_permutation(
                from0, sizeb, dim0, strides0,
                mask0.size() == 0 ? AllowImplicitPermutation : DontAllowImplicitPermutation,
                v0.ctx());
            IndicesT<IndexType, XPU1> indices1 = get_permutation(
                from1, sizeb, dim1, strides1,
                mask0.size() == 0 ? AllowImplicitPermutation : DontAllowImplicitPermutation,
                v1.ctx());
            IndexType blocking = 1;
            for (std::size_t i = 0; i < nblock; ++i) blocking *= size[i];

            // Do the copy
            if (blocking == 1) {
                if (mask0.size() > 0) {
                    indices0 = select(indices0, mask0.data() + disp0, indices0);
                    indices1 = select(indices1, mask1.data() + disp1, indices1);
                    if (indices0.size() != indices1.size())
                        throw std::runtime_error("copy: non-compatible masks");
                }
                copy_n<IndexType, T, Q>(alpha, v0.data() + disp0, v0.ctx(), indices0.begin(),
                                        indices0.ctx(), indices0.size(), v1.data() + disp1,
                                        v1.ctx(), indices1.begin(), indices1.ctx(), ewop);
            } else {
                copy_n_blocking<IndexType, T, Q>(alpha, v0.data() + disp0, v0.ctx(), blocking,
                                                 indices0.begin(), indices0.ctx(), indices0.size(),
                                                 v1.data() + disp1, v1.ctx(), indices1.begin(),
                                                 indices1.ctx(), ewop);
            }
        })

        /// Copy the content of tensor v0 into v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param mask0: mask for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param mask1: mask for the destination tensor (ignored)
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename T, typename Q,
                  typename XPU0, typename XPU1, typename EWOP>
        void local_copy(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        Mask<XPU0> mask0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
                        const Coor<Nd1> &dim1, vector<Q, XPU1> v1, Mask<XPU1> mask1, EWOP ewop,
                        CoorOrder co) {

            tracker<XPU1> _t(std::string("local copy ") + platformToStr(v0.ctx()) +
                                 std::string("-") + platformToStr(v1.ctx()),
                             v1.ctx());

            // Shortcut to scale or zero out a tensor
            if (std::is_same<T, Q>::value && (void *)v0.data() == (void *)v1.data() &&
                mask0.size() == 0 && o0 == o1 && from0 == Coor<Nd0>{{}} && from1 == Coor<Nd1>{{}} &&
                size0 == dim0 && dim0 == dim1 && std::is_same<EWOP, detail::EWOp::Copy>::value) {
                copy_n<IndexType, T, Q>(alpha, v0.data(), v0.ctx(), volume(size0), v1.data(),
                                        v1.ctx(), ewop);
                return;
            }

            // Canonize the copy operation
            constexpr std::size_t Nd = std::min(Nd0, Nd1);
            IndexType new_disp0, new_disp1;
            std::size_t nblock;
            Coor<Nd> new_from0, new_size, new_dim0, new_from1, new_dim1;
            Coor<Nd, IndexType> new_strides0, new_strides1;
            copy_normalize(o0, from0, size0, dim0, o1, from1, dim1, co, new_disp0, new_from0,
                           new_size, new_dim0, new_strides0, new_disp1, new_from1, new_dim1,
                           new_strides1, nblock);

            // Do the copy
            _t.cost = (double)(mask0.size() > 0 ? mask0.size() : volume(new_size)) *
                      (sizeof(T) + sizeof(Q));
            local_copy_normalize(alpha, new_disp0, new_from0, new_size, new_dim0, new_strides0, v0,
                                 mask0, new_disp1, new_from1, new_dim1, new_strides1, v1, mask1,
                                 nblock, ewop);
        }

        /// Copy the content of tensor v0 into v1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param mask0: mask for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param mask1: mask for the destination tensor (ignored)
        /// \param ewop: either to copy or to add the origin values into the destination values
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1, typename EWOP>
        void local_copy(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        Mask<XPU0> mask0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
                        const Coor<Nd1> &dim1, vector<Q, XPU1> v1, Mask<XPU1> mask1, EWOP,
                        CoorOrder co) {

            if (std::max(volume(dim0), volume(dim1)) >=
                (std::size_t)std::numeric_limits<IndexType>::max()) {
                local_copy<std::size_t>(alpha, o0, from0, size0, dim0, v0, mask0, o1, from1, dim1,
                                        v1, mask1, EWOP{}, co);
            } else {
                local_copy<IndexType>(alpha, o0, from0, size0, dim0, v0, mask0, o1, from1, dim1, v1,
                                      mask1, EWOP{}, co);
            }
        }

        /// Return the permutation on the origin to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param xpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename XPU>
        std::pair<IndicesT<IndexType, XPU>, IndexType>
        get_permutation_origin(const Order<Nd0> &o0, const Coor<Nd0> &from0, const Coor<Nd0> &size0,
                               const Coor<Nd0> &dim0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
                               const Coor<Nd1> &dim1, ImplicitPermutation implicitPermutation,
                               XPU xpu, CoorOrder co, std::size_t nblock = 0) {
            (void)from1;
            (void)dim1;

            tracker<XPU> _t("compute permutations (origin)", xpu);

            // Canonize the copy operation
            constexpr std::size_t Nd = std::min(Nd0, Nd1);
            std::size_t nblock1;
            IndexType new_disp0, new_disp1;
            Coor<Nd> new_from0, new_size, new_dim0, new_from1, new_dim1;
            Coor<Nd, IndexType> new_strides0, new_strides1;
            copy_normalize(o0, from0, size0, dim0, o1, from1, dim1, co, new_disp0, new_from0,
                           new_size, new_dim0, new_strides0, new_disp1, new_from1, new_dim1,
                           new_strides1, nblock1);

            for (std::size_t i = 0; i < nblock; ++i) new_size[i] = 1;

            // Compute the permutation
            return {get_permutation(new_from0, new_size, new_dim0, new_strides0,
                                    implicitPermutation, xpu),
                    new_disp0};
        }

        /// Return the permutation on the destination to copy from the origin tensor into the destination tensor
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param cpu: device context for the returned vector
        /// \param co: coordinate linearization order

        template <typename IndexType, std::size_t Nd0, std::size_t Nd1, typename XPU>
        std::pair<IndicesT<IndexType, XPU>, IndexType>
        get_permutation_destination(const Order<Nd0> &o0, const Coor<Nd0> &from0,
                                    const Coor<Nd0> &size0, const Coor<Nd0> &dim0,
                                    const Order<Nd1> &o1, const Coor<Nd1> &from1,
                                    const Coor<Nd1> &dim1, ImplicitPermutation implicitPermutation,
                                    XPU xpu, CoorOrder co, std::size_t nblock = 0) {

            (void)from0;
            (void)dim0;

            tracker<XPU> _t("compute permutations (destination)", xpu);

            // Canonize the copy operation
            constexpr std::size_t Nd = std::min(Nd0, Nd1);
            IndexType new_disp0, new_disp1;
            std::size_t nblock0;
            Coor<Nd> new_from0, new_size, new_dim0, new_from1, new_dim1;
            Coor<Nd, IndexType> new_strides0, new_strides1;
            copy_normalize(o0, from0, size0, dim0, o1, from1, dim1, co, new_disp0, new_from0,
                           new_size, new_dim0, new_strides0, new_disp1, new_from1, new_dim1,
                           new_strides1, nblock0);

            for (std::size_t i = 0; i < nblock; ++i) new_size[i] = 1;

            // Compute the permutation
            return {get_permutation(new_from1, new_size, new_dim1, new_strides1,
                                    implicitPermutation, xpu),
                    new_disp1};
        }

        /// Recommended orderings for contracting two tensors
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param sug_o0: (out) suggested dimension labels for the first operator
        /// \param sug_o1: (out) suggested dimension labels for the second operator
        /// \param sug_or: (out) suggested dimension labels for the output operator
        /// \param swap_operands: (out) suggest to swap the first and the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        void suggested_orders_for_contraction(
            const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0, const Order<Nd1> &o1,
            const Coor<Nd1> &dim1, bool conj1, const Order<Ndo> &o_r, const Coor<Ndo> &dimr,
            Order<Nd0> &sug_o0, Order<Nd1> &sug_o1, Order<Ndo> &sug_or, bool &swap_operands,
            unsigned int &nT, unsigned int &posT0, unsigned int &posT1, unsigned int &posTr,
            unsigned int &nA, unsigned int &posA0, unsigned int &posA1, unsigned int &nB,
            unsigned int &posB0, unsigned int &posBr, unsigned int &nC, unsigned int &posC1,
            unsigned int &posCr, CoorOrder co) {

            // TODO: not consider dimensions with a single element
            (void)dim0;
            (void)dim1;
            (void)dimr;

            // The rest of the code is for SlowToFast; so reverse if that is the case
            if (co == FastToSlow) {
                suggested_orders_for_contraction<Nd0, Nd1, Ndo>(
                    reverse(o0), reverse(dim0), conj0, reverse(o1), reverse(dim1), conj1,
                    reverse(o_r), reverse(dimr), sug_o0, sug_o1, sug_or, swap_operands, nT, posT0,
                    posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1, posCr, SlowToFast);
                sug_o0 = reverse(sug_o0);
                sug_o1 = reverse(sug_o1);
                sug_or = reverse(sug_or);
                posT0 = o0.size() - (posT0 + nT);
                posT1 = o1.size() - (posT1 + nT);
                posTr = o_r.size() - (posTr + nT);
                posA0 = o0.size() - (posA0 + nA);
                posA1 = o1.size() - (posA1 + nA);
                posB0 = o0.size() - (posB0 + nB);
                posBr = o_r.size() - (posBr + nB);
                posC1 = o1.size() - (posC1 + nC);
                posCr = o_r.size() - (posCr + nC);
                return;
            }

            // Find all common labels in o0, o1, and o_r
            Order<Nd0> oT;
            nT = 0;
            for (char c : o0)
                if (std::find(o1.begin(), o1.end(), c) != o1.end() &&
                    std::find(o_r.begin(), o_r.end(), c) != o_r.end())
                    oT[nT++] = c;

            // Find all common labels in o0 and o1 but not in oT
            Order<Nd0> oA;
            nA = 0;
            for (char c : o0)
                if (std::find(o1.begin(), o1.end(), c) != o1.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oA[nA++] = c;

            // Find all common labels in o0 and o_r but not in oT
            Order<Nd0> oB;
            nB = 0;
            for (char c : o0)
                if (std::find(o_r.begin(), o_r.end(), c) != o_r.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oB[nB++] = c;

            // Find all common labels in o1 and o_r but not in oT
            Order<Nd1> oC;
            nC = 0;
            for (char c : o1)
                if (std::find(o_r.begin(), o_r.end(), c) != o_r.end() &&
                    std::find(oT.begin(), oT.begin() + nT, c) == oT.begin() + nT)
                    oC[nC++] = c;

            // Check that o0 is made of the pieces T, A and B
            if (o0.size() != nT + nA + nB) throw std::runtime_error("o0 has unmatched dimensions");
            // Check that o1 is made of the pieces T, C and A
            if (o1.size() != nT + nA + nC) throw std::runtime_error("o1 has unmatched directions");
            // Check that o_r is made of the pieces T, C and B
            if (o_r.size() != nT + nB + nC)
                throw std::runtime_error("o_r has unmatched dimensions");

            // If oT, oA, or oB aren't found as either oT+oA+oB or oA+oT+oB or oT+oB+oA or oB+oT+oA for !conj,
            // and oT+oB+oA or oB+oT+oA for conj, then reorder the labels appropriately
            auto sT0 = std::search(o0.begin(), o0.end(), oT.begin(), oT.begin() + nT);
            auto sA0 = std::search(o0.begin(), o0.end(), oA.begin(), oA.begin() + nA);
            auto sB0 = std::search(o0.begin(), o0.end(), oB.begin(), oB.begin() + nB);
            if (sT0 == o0.end() || sA0 == o0.end() || sB0 == o0.end() ||
                (!conj0 && nT > 0 && nA > 0 && nB > 0 && sA0 < sT0 && sB0 < sT0) ||
                (conj0 && nA > 0 && ((nT > 0 && sA0 < sT0) || (nB > 0 && sA0 < sB0)))) {
                std::copy_n(oT.begin(), nT, sug_o0.begin());
                std::copy_n(oA.begin(), nA, sug_o0.begin() + nT + (!conj0 ? 0 : nB));
                std::copy_n(oB.begin(), nB, sug_o0.begin() + nT + (!conj0 ? nA : 0));
            } else
                sug_o0 = o0;

            // If oT, oA, or oC aren't found as either oT+oC+oA or oC+oT+oA or oT+oA+oC or oA+oT+oC for !conj,
            // and oT+oA+oC or oA+oT+oC for conj, then reorder the labels appropriately
            auto sT1 = std::search(o1.begin(), o1.end(), oT.begin(), oT.begin() + nT);
            auto sA1 = std::search(o1.begin(), o1.end(), oA.begin(), oA.begin() + nA);
            auto sC1 = std::search(o1.begin(), o1.end(), oC.begin(), oC.begin() + nC);
            if (sT1 == o1.end() || sA1 == o1.end() || sC1 == o1.end() ||
                (!conj1 && nT > 0 && nA > 0 && nC > 0 && sA1 < sT1 && sC1 < sT1) ||
                (conj1 && nC > 0 && ((nT > 0 && sC1 < sT1) || (nC > 0 && sC1 < sA1)))) {
                std::copy_n(oT.begin(), nT, sug_o1.begin());
                std::copy_n(oC.begin(), nC, sug_o1.begin() + nT + (!conj1 ? 0 : nA));
                std::copy_n(oA.begin(), nA, sug_o1.begin() + nT + (!conj1 ? nC : 0));
            } else
                sug_o1 = o1;

            // If oT, oB, or oC aren't found as either oT+oC+oB or oC+oT+oB, then reorder the labels appropriately
            auto sTr = std::search(o_r.begin(), o_r.end(), oT.begin(), oT.begin() + nT);
            auto sBr = std::search(o_r.begin(), o_r.end(), oB.begin(), oB.begin() + nB);
            auto sCr = std::search(o_r.begin(), o_r.end(), oC.begin(), oC.begin() + nC);
            swap_operands = false;
            if (sTr == o_r.end() || sBr == o_r.end() || sCr == o_r.end() ||
                (nT > 0 && nB > 0 && sBr < sTr) || (nB > 0 && nC > 0 && sBr < sCr)) {
                swap_operands = (nB > 0 && nC > 0 && sBr < sCr);
                std::copy_n(oT.begin(), nT, sug_or.begin());
                std::copy_n(oC.begin(), nC, sug_or.begin() + nT);
                std::copy_n(oB.begin(), nB, sug_or.begin() + nT + nC);
            } else
                sug_or = o_r;

            // Return positions
            posT0 = sT0 - o0.begin();
            posT1 = sT1 - o1.begin();
            posTr = sTr - o_r.begin();
            posA0 = sA0 - o0.begin();
            posA1 = sA1 - o1.begin();
            posB0 = sB0 - o0.begin();
            posBr = sBr - o_r.begin();
            posC1 = sC1 - o1.begin();
            posCr = sCr - o_r.begin();
        }

        /// Recommended orderings for contracting two tensors
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param sug_o0: (out) suggested dimension labels for the first operator
        /// \param sug_o1: (out) suggested dimension labels for the second operator
        /// \param sug_or: (out) suggested dimension labels for the output operator
        /// \param swap_operands: (out) suggest to swap the first and the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo>
        void
        suggested_orders_for_contraction(const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                                         const Order<Nd1> &o1, const Coor<Nd1> &dim1, bool conj1,
                                         const Order<Ndo> &o_r, const Coor<Ndo> &dimr,
                                         Order<Nd0> &sug_o0, Order<Nd1> &sug_o1, Order<Ndo> &sug_or,
                                         bool &swap_operands, CoorOrder co) {

            unsigned int nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1,
                posCr;
            suggested_orders_for_contraction(
                o0, dim0, conj0, o1, dim1, conj1, o_r, dimr, sug_o0, sug_o1, sug_or, swap_operands,
                nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1, posCr, co);
        }

        /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
        /// \param alpha: factor on the contraction
        /// \param o0: dimension labels for the first operator
        /// \param dim0: dimension size for the first operator
        /// \param conj0: whether element-wise conjugate the first operator
        /// \param v0: data for the first operator
        /// \param o1: dimension labels for the second operator
        /// \param dim1: dimension size for the second operator
        /// \param conj1: whether element-wise conjugate the second operator
        /// \param v1: data for the second operator
        /// \param beta: factor on the destination tensor
        /// \param o_r: dimension labels for the output operator
        /// \param dimr: dimension size for the output operator
        /// \param vr: data for the second operator
        /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order

        template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T, typename XPU>
        void local_contraction(T alpha, const Order<Nd0> &o0, const Coor<Nd0> &dim0, bool conj0,
                               vector<const T, XPU> v0, const Order<Nd1> &o1, const Coor<Nd1> &dim1,
                               bool conj1, vector<const T, XPU> v1, T beta, const Order<Ndo> &o_r,
                               const Coor<Ndo> &dimr, vector<T, XPU> vr, CoorOrder co) {

            tracker<XPU> _t(std::string("local contraction ") + platformToStr(vr.ctx()), vr.ctx());

            if (deviceId(v0.ctx()) != deviceId(v1.ctx()) ||
                deviceId(v1.ctx()) != deviceId(vr.ctx()))
                throw std::runtime_error("all arrays should be on the same device");

            // Check orders
            if (!check_order(o0)) throw std::runtime_error("o0 has repeated labels");
            if (!check_order(o1)) throw std::runtime_error("o1 has repeated labels");
            if (!check_order(o_r)) throw std::runtime_error("o_r has repeated labels");

            // Check dimensions
            if (!check_dimensions<Nd0, Nd1, Ndo>(o0, dim0, o1, dim1, o_r, dimr))
                throw std::runtime_error("some dimension does not match");

            // The rest of the code is for SlowToFast; so reverse if that is the case
            if (co == FastToSlow) {
                local_contraction<Nd0, Nd1, Ndo, T, XPU>(
                    alpha, reverse(o0), reverse(dim0), conj0, v0, reverse(o1), reverse(dim1), conj1,
                    v1, beta, reverse(o_r), reverse(dimr), vr, SlowToFast);
                return;
            }

            // If o0, o1, and o_r aren't appropriate, permute the input operands and the output
            unsigned int nT, posT0, posT1, posTr, nA, posA0, posA1, nB, posB0, posBr, nC, posC1,
                posCr;
            {
                Order<Nd0> sug_o0;
                Order<Nd1> sug_o1;
                Order<Ndo> sug_or;
                bool swap_operands;
                suggested_orders_for_contraction(o0, dim0, conj0, o1, dim1, conj1, o_r, dimr,
                                                 sug_o0, sug_o1, sug_or, swap_operands, nT, posT0,
                                                 posT1, posTr, nA, posA0, posA1, nB, posB0, posBr,
                                                 nC, posC1, posCr, co);
                if (swap_operands) {
                    local_contraction<Nd1, Nd0, Ndo, T, XPU>(alpha, o1, dim1, conj1, v1, o0, dim0,
                                                             conj0, v0, beta, o_r, dimr, vr,
                                                             SlowToFast);
                    return;
                }
                if (sug_o0 != o0 || sug_o1 != o1 || sug_or != o_r) {
                    Coor<Nd0> sug_dim0 = dim0;
                    vector<const T, XPU> sug_v0 = v0;
                    if (sug_o0 != o0) {
                        sug_dim0 = reorder_coor(dim0, find_permutation(o0, sug_o0));
                        vector<T, XPU> sug_v0_(v0.size(), v0.ctx());
                        local_copy<Nd0, Nd0>(T{1}, o0, {{}}, dim0, dim0, v0, Mask<XPU>{}, sug_o0,
                                             {{}}, sug_dim0, sug_v0_, Mask<XPU>{}, EWOp::Copy{},
                                             SlowToFast);
                        sug_v0 = sug_v0_;
                    }

                    Coor<Nd1> sug_dim1 = dim1;
                    vector<const T, XPU> sug_v1 = v1;
                    if (sug_o1 != o1) {
                        sug_dim1 = reorder_coor(dim1, find_permutation(o1, sug_o1));
                        vector<T, XPU> sug_v1_(v1.size(), v1.ctx());
                        local_copy<Nd1, Nd1>(T{1}, o1, {{}}, dim1, dim1, v1, Mask<XPU>{}, sug_o1,
                                             {{}}, sug_dim1, sug_v1_, Mask<XPU>{}, EWOp::Copy{},
                                             SlowToFast);
                        sug_v1 = sug_v1_;
                    }

                    Coor<Ndo> sug_dimr = dimr;
                    vector<T, XPU> sug_vr = vr;
                    if (sug_or != o_r) {
                        sug_dimr = reorder_coor(dimr, find_permutation(o_r, sug_or));
                        sug_vr = vector<T, XPU>(vr.size(), vr.ctx());
                        if (std::fabs(beta) != 0)
                            local_copy<Ndo, Ndo, T, T>(T{1}, o_r, {{}}, dimr, dimr,
                                                       vector<const T, XPU>(vr), Mask<XPU>{},
                                                       sug_or, {{}}, sug_dimr, sug_vr, Mask<XPU>{},
                                                       EWOp::Copy{}, SlowToFast);
                    }

                    local_contraction<Nd0, Nd1, Ndo, T, XPU>(alpha, sug_o0, sug_dim0, conj0, sug_v0,
                                                             sug_o1, sug_dim1, conj1, sug_v1, beta,
                                                             sug_or, sug_dimr, sug_vr, SlowToFast);

                    if (sug_or != o_r)
                        local_copy<Ndo, Ndo>(T{1}, sug_or, {{}}, sug_dimr, sug_dimr,
                                             vector<const T, XPU>(sug_vr), Mask<XPU>{}, o_r, {{}},
                                             dimr, vr, Mask<XPU>{}, EWOp::Copy{}, SlowToFast);
                    return;
                }
            }

            // Compute the volume for each piece
            std::size_t volT = volume<Nd0>(dim0.begin() + posT0, dim0.begin() + posT0 + nT);
            std::size_t volA = volume<Nd0>(dim0.begin() + posA0, dim0.begin() + posA0 + nA);
            std::size_t volB = volume<Nd0>(dim0.begin() + posB0, dim0.begin() + posB0 + nB);
            std::size_t volC = volume<Nd1>(dim1.begin() + posC1, dim1.begin() + posC1 + nC);
            std::size_t vol0 = volume<Nd0>(dim0);
            std::size_t vol1 = volume<Nd1>(dim1);
            std::size_t volr = volume<Ndo>(dimr);

            // Deal with zero dimensions and implicit dimensions
            if (volr == 0) return;
            if (volT == 0) volT = 1;
            if ((vol0 > 0 || vol1 > 0) && volA == 0) volA = 1;
            if ((vol0 > 0 || volr > 0) && volB == 0) volB = 1;
            if ((vol1 > 0 || volr > 0) && volC == 0) volC = 1;
            assert(volT * volA * volB == vol0);
            assert(volT * volA * volC == vol1);
            assert(volT * volB * volC == volr);

            // Avoid issues with uninitialized memory by zeroing out
            if (std::fabs(beta) == 0.0) zero_n<T>(vr.data(), volume<Ndo>(dimr), vr.ctx());

            // Quick exit
            if (volA == 0) return;

            // Check that no order ends with T
            if (nT > 0 && posT0 + nT == o0.size() && volA > 1 && volB > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the first tensor");
            if (nT > 0 && posT1 + nT == o1.size() && volA > 1 && volC > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the second tensor");
            if (nT > 0 && posTr + nT == o_r.size() && volB > 1 && volC > 1)
                throw std::runtime_error(
                    "Unsupported contraction: the common dimensions to the input and "
                    "output tensors cannot be packed at the end of the output tensor");

            // We don't support empty tensors
            if ((o0.size() == 0) xor (o1.size() == 0))
                throw std::runtime_error("Unsupported contraction: one of the input tensors is "
                                         "empty and the other is not");
            if (o_r.size() == 0 && o0.size() + o1.size() > 0)
                throw std::runtime_error("Unsupported contraction: the output tensor is empty but "
                                         "some of the input tensors is not");

            // Check whether each order starts with T
            bool o0_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Nd0; ++i) {
                if (i == posT0) o0_starts_with_T = true;
                if (dim0[i] > 1) break;
            }
            bool o1_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Nd1; ++i) {
                if (i == posT1) o1_starts_with_T = true;
                if (dim1[i] > 1) break;
            }
            bool or_starts_with_T = (volT <= 1);
            for (unsigned int i = 0; i < Ndo; ++i) {
                if (i == posTr) or_starts_with_T = true;
                if (dimr[i] > 1) break;
            }

            // Check if o0 and o1 need transpose TAB, TCA, TCB -> BA, AC, BC
            bool o0_trans = (volA > 1 && volB > 1 && posB0 < posA0) |              // BA
                            (volA == 1 && volB > 1 && volT > 1 && posB0 < posT0) | // BT
                            (conj0 && ((o0_starts_with_T && (volA == 1 || volB == 1)) ||
                                       (!o0_starts_with_T && volA == 1 && volB == 1)));
            bool o1_trans = (volC > 1 && volA > 1 && posA1 < posC1) |              // AC
                            (volC == 1 && volA > 1 && volT > 1 && posA1 < posT1) | // AT
                            (conj1 && ((o1_starts_with_T && (volC == 1 || volA == 1)) ||
                                       (!o1_starts_with_T && volC == 1 && volA == 1)));
            bool or_trans = (volC > 1 && volB > 1 && posBr < posCr) |             // BC
                            (volC == 1 && volB > 1 && volT > 1 && posBr < posTr); // BT
            if (!o0_trans && conj0)
                throw std::runtime_error("Unsupported contraction: reorder the labels on the first "
                                         "tensor to use conjugation");
            if (!o1_trans && conj1)
                throw std::runtime_error("Unsupported contraction: reorder the labels on the "
                                         "second tensor to use conjugation");
            if (or_trans)
                throw std::runtime_error("Unsupported contraction: on the output labels, put "
                                         "the labels from the second "
                                         "tensor before the labels from the first tensor.");

            // Let's do (A, B) x (C, A) -> (C, B)
            char transab = (o0_trans ? (conj0 ? 'C' : 'T') : 'N');
            char transca = (o1_trans ? (conj1 ? 'C' : 'T') : 'N');
            std::size_t ldab = (o0_starts_with_T ? 1u : volT) * (!o0_trans ? volB : volA);
            std::size_t strideab = (o0_starts_with_T ? volA * volB : (!o0_trans ? volB : volA));
            std::size_t ldca = (o1_starts_with_T ? 1u : volT) * (!o1_trans ? volA : volC);
            std::size_t strideca = (o1_starts_with_T ? volA * volC : (!o1_trans ? volA : volC));
            std::size_t ldcb = (or_starts_with_T ? 1u : volT) * volB;
            std::size_t stridecb = (or_starts_with_T ? volB * volC : volB);
            if (std::max(
                    volA,
                    std::max(
                        volB,
                        std::max(
                            volC,
                            std::max(
                                volT,
                                std::max(
                                    ldab,
                                    std::max(
                                        strideab,
                                        std::max(ldca, std::max(strideca,
                                                                std::max(ldcb, stridecb))))))))) >=
                (std::size_t)std::numeric_limits<int>::max()) {
                std::runtime_error("contraction: too large tensors to contract");
            }
            _t.cost = volA * volB * volC * volT * multiplication_cost<T>::value;
            xgemm_batch_strided(transab, transca, volB, volC, volA, alpha, v0.data(), ldab,
                                strideab, v1.data(), ldca, strideca, beta, vr.data(), ldcb,
                                stridecb, volT, vr.ctx());
        }

        /// Copy the content of tensor o0 into o1
        /// \param alpha: factor on the copy
        /// \param o0: dimension labels for the origin tensor
        /// \param from0: first coordinate to copy from the origin tensor
        /// \param size0: number of coordinates to copy in each direction
        /// \param dim0: dimension size for the origin tensor
        /// \param v0: data for the origin tensor
        /// \param mask0: mask for the origin tensor
        /// \param o1: dimension labels for the destination tensor
        /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
        /// \param dim1: dimension size for the destination tensor
        /// \param v1: data for the destination tensor
        /// \param mask1: mask for the destination tensor
        /// \param copyadd: either to copy or to add the origin values into the destination tensor
        /// \param co: coordinate linearization order

        template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q, typename XPU0,
                  typename XPU1>
        void local_copy(typename elem<T>::type alpha, const Order<Nd0> &o0, const Coor<Nd0> &from0,
                        const Coor<Nd0> &size0, const Coor<Nd0> &dim0, vector<const T, XPU0> v0,
                        Mask<XPU0> mask0, const Order<Nd1> &o1, const Coor<Nd1> &from1,
                        const Coor<Nd1> &dim1, vector<Q, XPU1> v1, Mask<XPU1> mask1,
                        CopyAdd copyadd, CoorOrder co) {
            switch (copyadd) {
            case Copy:
                local_copy<Nd0, Nd1>(alpha, o0, from0, size0, dim0, v0, mask0, o1, from1, dim1, v1,
                                     mask1, EWOp::Copy{}, co);
                break;
            case Add:
                local_copy<Nd0, Nd1>(alpha, o0, from0, size0, dim0, v0, mask0, o1, from1, dim1, v1,
                                     mask1, EWOp::Add{}, co);
                break;
            }
        }
    }

    /// Copy the content of tensor o0 into o1
    /// \param alpha: factor on the copy
    /// \param o0: dimension labels for the origin tensor
    /// \param from0: first coordinate to copy from the origin tensor
    /// \param size0: number of coordinates to copy in each direction
    /// \param dim0: dimension size for the origin tensor
    /// \param v0: data for the origin tensor
    /// \param mask0: mask for the origin tensor
    /// \param ctx0: device context for v0
    /// \param o1: dimension labels for the destination tensor
    /// \param from1: coordinate in destination tensor where first coordinate from origin tensor is copied
    /// \param dim1: dimension size for the destination tensor
    /// \param v1: data for the destination tensor
    /// \param mask1: mask for the destination tensor
    /// \param ctx1: device context for v1
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param copyadd: either copy or add the origin value to the destination values
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, typename T, typename Q>
    void local_copy(typename elem<T>::type alpha, const char *o0, const Coor<Nd0> &from0,
                    const Coor<Nd0> &size0, const Coor<Nd0> &dim0, const T *v0,
                    const MaskType *mask0, const Context ctx0, const char *o1,
                    const Coor<Nd1> &from1, const Coor<Nd1> &dim1, Q *v1, const MaskType *mask1,
                    const Context ctx1, CoorOrder co, CopyAdd copyadd, Session session = 0) {

        const Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        const Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");

        // Check the validity of the operation
        if (!detail::check_positive<Nd0>(from0))
            throw std::runtime_error("All values in `from0` should be non-negative");

        if (!detail::check_positive<Nd0>(size0))
            throw std::runtime_error("All values in `size0` should be non-negative");

        if (!detail::check_positive<Nd1>(from1))
            throw std::runtime_error("All values in `from1` should be non-negative");

        if (!detail::check_isomorphic<Nd0, Nd1>(o0_, size0, dim0, o1_, dim1))
            throw std::runtime_error("The orders and dimensions of the origin tensor are not "
                                     "compatible with the destination tensor");

        std::size_t vol0 = detail::volume(dim0);
        std::size_t vol1 = detail::volume(dim1);
        std::size_t volmask0 = mask0 ? vol0 : 0;
        std::size_t volmask1 = mask1 ? vol1 : 0;

        // Do the operation
        if (ctx0.plat == CPU && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0, detail::to_vector(v0, vol0, ctx0.toCpu(session)),
                detail::to_vector((MaskType *)mask0, volmask0, ctx0.toCpu(session)), o1_, from1,
                dim1, detail::to_vector(v1, detail::volume(dim1), ctx1.toCpu(session)),
                detail::to_vector((MaskType *)mask1, volmask1, ctx1.toCpu(session)), copyadd, co);
        }
#ifdef SUPERBBLAS_USE_GPU
        else if (ctx0.plat == CPU && ctx1.plat == GPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0, detail::to_vector(v0, vol0, ctx0.toCpu(session)),
                detail::to_vector((MaskType *)mask0, volmask0, ctx0.toCpu(session)), o1_, from1,
                dim1, detail::to_vector(v1, vol1, ctx1.toGpu(session)),
                detail::to_vector((MaskType *)mask1, volmask1, ctx1.toGpu(session)), copyadd, co);
        } else if (ctx0.plat == GPU && ctx1.plat == CPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0, detail::to_vector(v0, vol0, ctx0.toGpu(session)),
                detail::to_vector((MaskType *)mask0, volmask0, ctx0.toGpu(session)), o1_, from1,
                dim1, detail::to_vector(v1, vol1, ctx1.toCpu(session)),
                detail::to_vector((MaskType *)mask1, volmask1, ctx1.toCpu(session)), copyadd, co);
        } else if (ctx0.plat == GPU && ctx1.plat == GPU) {
            detail::local_copy<Nd0, Nd1, T, Q>(
                alpha, o0_, from0, size0, dim0, detail::to_vector(v0, vol0, ctx0.toGpu(session)),
                detail::to_vector((MaskType *)mask0, volmask0, ctx0.toGpu(session)), o1_, from1,
                dim1, detail::to_vector(v1, vol1, ctx1.toGpu(session)),
                detail::to_vector((MaskType *)mask1, volmask1, ctx1.toGpu(session)), copyadd, co);
        }
#endif
        else {
            throw std::runtime_error("Unsupported platform");
        }
    }

    /// Contract two tensors: vr = alpha * contraction(v0, v1) + beta * vr
    /// \param alpha: factor on the contraction
    /// \param o0: dimension labels for the first operator
    /// \param dim0: dimension size for the first operator
    /// \param conj0: whether element-wise conjugate the first operator
    /// \param v0: data for the first operator
    /// \param o1: dimension labels for the second operator
    /// \param dim1: dimension size for the second operator
    /// \param conj1: whether element-wise conjugate the second operator
    /// \param v1: data for the second operator
    /// \param beta: factor on the destination tensor
    /// \param o_r: dimension labels for the output operator
    /// \param dimr: dimension size for the output operator
    /// \param vr: data for the second operator
    /// \param co: coordinate linearization order; either `FastToSlow` for natural order or `SlowToFast` for lexicographic order
    /// \param session: concurrent calls should have different session

    template <std::size_t Nd0, std::size_t Nd1, std::size_t Ndo, typename T>
    void local_contraction(T alpha, const char *o0, const Coor<Nd0> &dim0, bool conj0, const T *v0,
                           const char *o1, const Coor<Nd1> &dim1, bool conj1, const T *v1, T beta,
                           const char *o_r, const Coor<Ndo> &dimr, T *vr, const Context ctx,
                           CoorOrder co, Session session = 0) {

        Order<Nd0> o0_ = detail::toArray<Nd0>(o0, "o0");
        Order<Nd1> o1_ = detail::toArray<Nd1>(o1, "o1");
        Order<Ndo> o_r_ = detail::toArray<Ndo>(o_r, "o_r");

        switch (ctx.plat) {
        case CPU:
            detail::local_contraction<Nd0, Nd1, Ndo, T>(
                alpha, o0_, dim0, conj0,
                detail::to_vector(v0, detail::volume(dim0), ctx.toCpu(session)), o1_, dim1, conj1,
                detail::to_vector(v1, detail::volume(dim1), ctx.toCpu(session)), beta, o_r_, dimr,
                detail::to_vector(vr, detail::volume(dimr), ctx.toCpu(session)), co);
            break;
#ifdef SUPERBBLAS_USE_GPU
        case GPU:
            detail::local_contraction<Nd0, Nd1, Ndo, T>(
                alpha, o0_, dim0, conj0,
                detail::to_vector(v0, detail::volume(dim0), ctx.toGpu(session)), o1_, dim1, conj1,
                detail::to_vector(v1, detail::volume(dim1), ctx.toGpu(session)), beta, o_r_, dimr,
                detail::to_vector(vr, detail::volume(dimr), ctx.toGpu(session)), co);
            break;
#endif
        default: throw std::runtime_error("Unsupported platform");
        }
    }
}
#endif // __SUPERBBLAS_TENSOR__
