// ztensor/zt/SmallVector.h
//
// A minimal small-buffer-optimized vector. Trimmed-down port of LLVM's /
// PyTorch's SmallVector: stores up to `N` elements inline on the stack and
// falls back to the heap beyond that.
//
// ztensor uses SmallVector<int64_t, 4> for shape/strides, since the vast
// majority of tensors are <= 4-D; this avoids a heap allocation in the common
// case.
//
// This is intentionally small: it implements the subset of std::vector's API
// that ztensor's core relies on (random access, push_back, resize, size,
// data, begin/end). It is NOT a drop-in std::vector replacement.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <new>
#include <type_traits>
#include <utility>

namespace zt {

template<typename T, std::size_t N>
class SmallVector {
    static_assert(N > 0, "SmallVector requires N > 0");
    static_assert(std::is_trivially_copyable<T>::value,
                  "zt::SmallVector only supports trivially-copyable T "
                  "(sufficient for shape/stride int64_t).");

public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    SmallVector() noexcept = default;

    explicit SmallVector(size_type count, T value = T{}) {
        assign(count, value);
    }

    SmallVector(std::initializer_list<T> il) { assign(il.begin(), il.end()); }

    template<
        typename InputIt,
        typename = typename std::iterator_traits<InputIt>::iterator_category>
    SmallVector(InputIt first, InputIt last) {
        assign(first, last);
    }

    SmallVector(const SmallVector& other) {
        assign(other.begin(), other.end());
    }
    SmallVector(SmallVector&& other) noexcept { move_from(other); }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) assign(other.begin(), other.end());
        return *this;
    }
    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            clear();
            move_from(other);
        }
        return *this;
    }

    ~SmallVector() { destroy_storage(); }

    // ---- size / capacity ----
    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    // ---- element access ----
    reference operator[](size_type i) { return data()[i]; }
    const_reference operator[](size_type i) const { return data()[i]; }
    reference front() { return data()[0]; }
    const_reference front() const { return data()[0]; }
    reference back() { return data()[size_ - 1]; }
    const_reference back() const { return data()[size_ - 1]; }

    pointer data() noexcept { return is_inline() ? inline_data() : heap_.ptr; }
    const_pointer data() const noexcept {
        return is_inline() ? inline_data() : heap_.ptr;
    }

    // ---- iteration ----
    iterator begin() noexcept { return data(); }
    iterator end() noexcept { return data() + size_; }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end() const noexcept { return data() + size_; }
    const_iterator cbegin() const noexcept { return data(); }
    const_iterator cend() const noexcept { return data() + size_; }

    // ---- mutators ----
    void clear() noexcept {
        size_ = 0;
        // Keep capacity; do not free.
    }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity_) return;
        grow(new_cap);
    }

    void resize(size_type new_size, T value = T{}) {
        if (new_size < size_) {
            size_ = new_size;
            return;
        }
        if (new_size > capacity_) grow(new_size);
        std::fill(data() + size_, data() + new_size, value);
        size_ = new_size;
    }

    void push_back(T value) {
        if (size_ == capacity_) grow_for_push();
        data()[size_++] = value;
    }

    void pop_back() {
        if (size_ > 0) --size_;
    }

    template<
        typename InputIt,
        typename = typename std::iterator_traits<InputIt>::iterator_category>
    void assign(InputIt first, InputIt last) {
        clear();
        for (; first != last; ++first) push_back(*first);
    }

    void assign(size_type count, T value) {
        clear();
        reserve(count);
        std::fill(data(), data() + count, value);
        size_ = count;
    }

private:
    bool is_inline() const noexcept { return capacity_ == N; }

    T* inline_data() noexcept { return reinterpret_cast<T*>(inline_storage_); }
    const T* inline_data() const noexcept {
        return reinterpret_cast<const T*>(inline_storage_);
    }

    void destroy_storage() noexcept {
        if (!is_inline()) {
            ::operator delete(heap_.ptr);
            heap_.ptr = nullptr;
            heap_.cap = 0;
        }
        size_ = 0;
        capacity_ = N;
    }

    void move_from(SmallVector& other) noexcept {
        if (other.is_inline()) {
            std::copy(other.inline_data(),
                      other.inline_data() + other.size_,
                      inline_data());
            size_ = other.size_;
            capacity_ = N;
            other.size_ = 0;
        } else {
            heap_ = other.heap_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.heap_.ptr = nullptr;
            other.heap_.cap = 0;
            other.size_ = 0;
            other.capacity_ = N;
        }
    }

    void grow(size_type new_cap) {
        // Geometric growth, but at least to new_cap.
        const size_type double_cap = capacity_ * 2;
        const size_type target = std::max(new_cap, double_cap);
        T* new_buf = static_cast<T*>(::operator new(target * sizeof(T)));
        std::copy(data(), data() + size_, new_buf);
        if (!is_inline()) ::operator delete(heap_.ptr);
        heap_.ptr = new_buf;
        heap_.cap = target;
        capacity_ = target;
    }

    void grow_for_push() { grow(capacity_ * 2); }

    // Active union: either inline storage (when capacity_ == N) or heap.
    union {
        alignas(T) unsigned char inline_storage_[N * sizeof(T)];
        struct {
            T* ptr;
            std::size_t cap;
        } heap_;
    };
    std::size_t size_ = 0;
    std::size_t capacity_ = N;
};

template<typename T, std::size_t N>
inline bool operator==(const SmallVector<T, N>& a,
                       const SmallVector<T, N>& b) noexcept {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
template<typename T, std::size_t N>
inline bool operator!=(const SmallVector<T, N>& a,
                       const SmallVector<T, N>& b) noexcept {
    return !(a == b);
}

}  // namespace zt
