// ztensor/core/Generator.cpp
//
// Global random-number state backing Generator::global_seed() /
// manual_seed(). The global seed is a simple atomic so it is safe to call
// manual_seed() from any thread (the last write wins), and the read path
// (Generator::global_seed()) is lock-free.

#include "ztensor/zt/Generator.h"

#include <atomic>

namespace zt {

namespace {
std::atomic<uint64_t> g_global_seed{67280421310721ULL};
}  // namespace

uint64_t Generator::global_seed() noexcept {
    return g_global_seed.load(std::memory_order_relaxed);
}

void Generator::set_global_seed(uint64_t seed) noexcept {
    g_global_seed.store(seed, std::memory_order_relaxed);
}

void manual_seed(uint64_t seed) {
    Generator::set_global_seed(seed);
}

}  // namespace zt
