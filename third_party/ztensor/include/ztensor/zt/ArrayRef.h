// ztensor/zt/ArrayRef.h
//
// Immutable reference to a contiguous span of `T`. A faithful, trimmed-down
// port of LLVM/PyTorch's ArrayRef. The main use is `IntArrayRef` below: the
// canonical "shape / strides / dims" argument type throughout the API.
//
// ArrayRef is a non-owning view; the caller must keep the backing storage
// alive for the duration of use. It is intended only for function arguments
// and short-lived locals.

#pragma once

// GCC's -Winit-list-lifetime fires on the initializer_list constructor below,
// but the backing temporary lives to the end of the full expression — exactly
// the lifetime an ArrayRef function argument needs (cf. PyTorch IntArrayRef).
// Silence the warning for this header only.
// -Winit-list-lifetime only exists in GCC >= 10. On older GCC (e.g. the 8.5
// shipped by the manylinux cu118 base) the pragma is an unknown option and
// trips -Wpragmas (-> -Werror). Gate it on the version that introduced it.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 10
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-list-lifetime"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <vector>

namespace zt {

template<typename T>
class ArrayRef {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = const T&;
    using const_reference = const T&;
    using pointer = const T*;
    using const_pointer = const T*;
    using iterator = const T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;

    constexpr ArrayRef() noexcept = default;

    // From C array / pointer + length.
    constexpr ArrayRef(const T* data, size_type length) noexcept
        : data_(data), length_(length) {}

    // From a range [begin, end).
    constexpr ArrayRef(const T* begin, const T* end) noexcept
        : data_(begin), length_(static_cast<size_type>(end - begin)) {}

    // From std::vector (only when `data()` returns a usable pointer).
    ArrayRef(const std::vector<T>& v) noexcept  // NOLINT(runtime/explicit)
        : data_(v.data()), length_(v.size()) {}

    // From std::array.
    template<std::size_t N>
    ArrayRef(const std::array<T, N>& arr) noexcept  // NOLINT(runtime/explicit)
        : data_(arr.data()), length_(N) {}

    // From initializer_list. NOTE: GCC's -Winit-list-lifetime fires here even
    // though the backing array lives to the end of the full expression, which
    // is sufficient for an ArrayRef used as a function argument. The warning
    // is therefore silenced at the top of this header (see file-level pragma).
    ArrayRef(std::initializer_list<T> il) noexcept  // NOLINT(runtime/explicit)
        : data_(il.begin() == il.end() ? nullptr : il.begin()),
          length_(il.size()) {}

    // From a single element (convenience for size-1 args).
    ArrayRef(const T& one) noexcept  // NOLINT(runtime/explicit)
        : data_(&one), length_(1) {}

    // ---- observers ----
    constexpr const T* data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return length_; }
    constexpr bool empty() const noexcept { return length_ == 0; }

    constexpr const T& operator[](size_type i) const noexcept {
        return data_[i];
    }
    const T& front() const noexcept { return data_[0]; }
    const T& back() const noexcept { return data_[length_ - 1]; }

    // ---- iteration ----
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + length_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + length_; }

    reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }

    // ---- comparison ----
    bool equals(ArrayRef<T> rhs) const noexcept {
        return length_ == rhs.length_ &&
               std::equal(data_, data_ + length_, rhs.data_);
    }

    // ---- slicing ----
    ArrayRef<T> slice(size_type start, size_type length) const noexcept {
        return ArrayRef<T>(data_ + start, length);
    }
    ArrayRef<T> drop_front(size_type n = 1) const noexcept {
        return ArrayRef<T>(data_ + n, length_ - n);
    }
    ArrayRef<T> drop_back(size_type n = 1) const noexcept {
        return ArrayRef<T>(data_, length_ - n);
    }

private:
    const T* data_ = nullptr;
    size_type length_ = 0;
};

template<typename T>
inline bool operator==(ArrayRef<T> a, ArrayRef<T> b) noexcept {
    return a.equals(b);
}
template<typename T>
inline bool operator!=(ArrayRef<T> a, ArrayRef<T> b) noexcept {
    return !a.equals(b);
}

// The canonical shape / strides / dims argument type.
using IntArrayRef = ArrayRef<int64_t>;

}  // namespace zt

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
