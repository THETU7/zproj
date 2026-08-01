// ztensor/zt/TensorKey.h
//
// Index-key descriptor for `Tensor::index()` / `Tensor::index_put_()`. A
// trimmed port of Open3D's open3d::core::TensorKey, expressed as a
// std::variant instead of Open3D's pimpl-with-virtual hierarchy (no
// per-key vtable, no shared_ptr<Impl>).
//
// Five primitive key kinds (mirroring NumPy/PyTorch/Open3D):
//   * Index    — a single integer along one axis (drops the dimension).
//   * Slice    — a (start, stop, step) triple, each optional (None).
//   * Tensor   — an int64 or bool index tensor (advanced indexing).
//   * Ellipsis — `...`, expands to enough full slices to cover the unlisted
//                trailing/interior dims (at most one per key list).
//   * NewAxis  — `None` as a standalone key, inserts a size-1 dim at its
//                position (equivalent to unsqueeze); does not consume a
//                source dim.
//
// Boolean masks are carried as Tensor keys and expanded to int64 via NonZero
// inside the advanced-indexing preprocessor.
//
// To avoid a header cycle with Tensor.h (which needs TensorKey for
// operator[]), the Tensor alternative is held via shared_ptr<Tensor>; only a
// forward declaration of zt::Tensor is required here.
//
// `None` (below) is the placeholder for an unspecified slice bound, matching
// the role of Python's `None` in a slice.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include "ztensor/zt/utility/Log.h"

namespace zt {

class Tensor;  // forward declaration — defined in Tensor.h

// Placeholder type for an unspecified slice bound (start/stop/step). Passing
// `None` (or {}) leaves that bound to be resolved against the dim size.
// Implicitly convertible to std::optional<int64_t> so that `Slice(None, 3)`
// reads naturally.
struct NoneTag {
    template<typename T>
    constexpr operator std::optional<T>()
        const noexcept {  // NOLINT(google-explicit-constructor)
        return std::nullopt;
    }
};
inline constexpr NoneTag None{};

// `...` (Ellipsis): a key that expands to enough full slices to cover every
// unlisted axis. At most one Ellipsis may appear in a key list (NumPy rule).
struct EllipsisTag {};
inline constexpr EllipsisTag Ellipsis{};

// `None` as a standalone key (a.k.a. np.newaxis): inserts a size-1 dimension
// at its position in the key list (equivalent to unsqueeze). Unlike the slice
// bound `None` (NoneTag above), this is a *key* — it does not consume a source
// dim. Written `zt::NewAxis` (or `zt::TensorKey::NewAxis()`) to read clearly
// alongside `zt::Slice` / `zt::Index`.
struct NewAxisTag {};
inline constexpr NewAxisTag NewAxis{};

// A raw slice triple before None-resolution. `InstantiateSlice(dim_size)`
// turns it into concrete bounds. Users normally build slices via the free
// `Slice(...)` helpers below, which accept None for any bound.
struct SliceSpec {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;
};

// A fully-resolved slice (no None): concrete (start, stop, step).
struct ResolvedSlice {
    int64_t start;
    int64_t stop;
    int64_t step;
};

// Resolve `s` against `dim_size`: defaults follow NumPy semantics.
//   step  defaults to 1; must be non-zero.
//   step > 0: start defaults to 0, stop defaults to dim_size.
//   step < 0: start defaults to dim_size - 1, stop defaults to a value that
//             includes index 0 (left as a sentinel negative; Tensor::slice's
//             clamp finalizes it).
// Explicit negative start is wrapped by adding dim_size (NumPy convention).
inline ResolvedSlice InstantiateSlice(const SliceSpec& s, int64_t dim_size) {
    ZT_CHECK(
        dim_size >= 0, "InstantiateSlice: dim_size {} must be >= 0", dim_size);
    const int64_t step = s.step.value_or(1);
    ZT_CHECK(step != 0, "InstantiateSlice: step must be non-zero");
    int64_t start, stop;
    if (step > 0) {
        start = s.start.value_or(0);
        stop = s.stop.value_or(dim_size);
    } else {
        start = s.start.value_or(dim_size - 1);
        stop = s.stop.value_or(-dim_size - 1);
    }
    if (s.start.has_value() && start < 0) start += dim_size;
    if (s.stop.has_value() && step > 0 && stop < 0) stop += dim_size;
    return {start, stop, step};
}

// A single index key. Cheap to copy for Index/Slice (a plain int64 / 24-byte
// struct); the Tensor alternative holds a shared_ptr (one heap node) so that
// this header needs only a forward declaration of zt::Tensor.
class TensorKey {
public:
    // Order matches the variant alternatives below so that mode() ==
    // static_cast<Mode>(impl_.index()).
    enum class Mode { Index, Slice, Tensor, Ellipsis, NewAxis };

    // ---- factories ----
    static TensorKey Index(int64_t i) { return TensorKey(i); }
    static TensorKey Slice(std::optional<int64_t> start,
                           std::optional<int64_t> stop,
                           std::optional<int64_t> step = std::nullopt) {
        return TensorKey(SliceSpec{start, stop, step});
    }
    static TensorKey Slice(const SliceSpec& s) { return TensorKey(s); }
    static TensorKey Tensor(const zt::Tensor& idx);
    static TensorKey Ellipsis() { return TensorKey(EllipsisTag{}); }
    static TensorKey NewAxis() { return TensorKey(NewAxisTag{}); }

    // Implicit converting constructors from the tag constants, so that
    // `zt::Ellipsis` / `zt::NewAxis` can be placed directly in a key list
    // (e.g. `{zt::Ellipsis, zt::NewAxis}`). Deliberately non-explicit.
    TensorKey(EllipsisTag) : impl_(EllipsisTag{}) {}  // NOLINT(google-explicit-constructor)
    TensorKey(NewAxisTag) : impl_(NewAxisTag{}) {}  // NOLINT(google-explicit-constructor)

    // ---- queries ----
    Mode mode() const { return static_cast<Mode>(impl_.index()); }
    bool IsIndex() const { return mode() == Mode::Index; }
    bool IsSlice() const { return mode() == Mode::Slice; }
    bool IsTensor() const { return mode() == Mode::Tensor; }
    bool IsEllipsis() const { return mode() == Mode::Ellipsis; }
    bool IsNewAxis() const { return mode() == Mode::NewAxis; }

    // Accessors (wrong type -> throws via ZT_LOG_ERROR). The Tensor accessor
    // returns a reference into the shared Tensor held by this key.
    int64_t GetIndex() const {
        ZT_CHECK(IsIndex(), "TensorKey::GetIndex: key is not an Index");
        return std::get<int64_t>(impl_);
    }
    const SliceSpec& GetSlice() const {
        ZT_CHECK(IsSlice(), "TensorKey::GetSlice: key is not a Slice");
        return std::get<SliceSpec>(impl_);
    }
    const zt::Tensor& GetTensor() const;

private:
    explicit TensorKey(int64_t i) : impl_(i) {}
    explicit TensorKey(const SliceSpec& s) : impl_(s) {}
    explicit TensorKey(const zt::Tensor& t);

    std::variant<int64_t,
                 SliceSpec,
                 std::shared_ptr<zt::Tensor>,
                 EllipsisTag,
                 NewAxisTag>
        impl_;
};

// ---- free convenience builders (so users write Slice(0, 3) / Ellipsis) ----
inline TensorKey Index(int64_t i) { return TensorKey::Index(i); }
inline TensorKey Slice(std::optional<int64_t> start,
                       std::optional<int64_t> stop,
                       std::optional<int64_t> step = std::nullopt) {
    return TensorKey::Slice(start, stop, step);
}
inline TensorKey Slice(const SliceSpec& s) { return TensorKey::Slice(s); }
// `zt::Ellipsis` / `zt::NewAxis` (the tag constants above) convert implicitly
// to TensorKey, so they can be used directly inside an index key list:
//   t.index({zt::Ellipsis, zt::NewAxis})   // t[..., None]

}  // namespace zt
