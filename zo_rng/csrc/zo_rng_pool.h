/*
 * zo_rng_pool.h — Dedicated std::thread pool for zo_rng, independent of OpenMP.
 *
 * This pool exists so that zo_rng's parallel RNG does NOT share the OpenMP
 * thread pool with PyTorch ATen.  When both run concurrently (shadow pipeline
 * producer vs consumer), they operate on completely separate threads.
 *
 * Supports multiple independent pool instances via ZoRngPoolRegistry:
 *   - pool_id=0: global default pool (backward compatible)
 *   - pool_id>=1: user-created pools via create_pool()
 *
 * Usage:
 *   // Default pool (backward compatible)
 *   ZoRngPoolRegistry::instance().set_num_threads(32);
 *   ZoRngPoolRegistry::instance().get(0).parallel_for(n, [&](...) { ... });
 *
 *   // Multi-instance: each producer gets its own pool
 *   int pid = ZoRngPoolRegistry::instance().create_pool(8);
 *   ZoRngPoolRegistry::instance().get(pid).parallel_for(n, [&](...) { ... });
 *   ZoRngPoolRegistry::instance().destroy_pool(pid);
 */
#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>

class ZoRngThreadPool {
public:
    ZoRngThreadPool() : num_threads_(1), shutdown_(false), generation_(0) {}

    explicit ZoRngThreadPool(int n)
        : num_threads_(n > 0 ? n : 1), shutdown_(false), generation_(0) {
        if (num_threads_ > 1) init_pool();
    }

    void set_num_threads(int n) {
        if (n == num_threads_ && !workers_.empty()) return;
        num_threads_ = (n > 0) ? n : 1;
        init_pool();  // explicit call → create workers immediately
    }

    int get_num_threads() const { return num_threads_; }

    void parallel_for(int64_t n,
                      const std::function<void(int64_t, int64_t)>& body) {
        if (num_threads_ <= 1) {
            body(0, n);
            return;
        }
        if (workers_.empty()) {
            init_pool();  // lazy: first parallel_for creates the pool
        }
        body_ = body;
        total_n_ = n;
        active_.store(num_threads_, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            generation_++;
        }
        start_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lk(mtx_);
            done_cv_.wait(lk, [&]{
                return active_.load(std::memory_order_acquire) == 0;
            });
        }
    }

    ~ZoRngThreadPool() { shutdown(); }

    // Non-copyable, non-movable (threads hold `this` pointer)
    ZoRngThreadPool(const ZoRngThreadPool&) = delete;
    ZoRngThreadPool& operator=(const ZoRngThreadPool&) = delete;
    ZoRngThreadPool(ZoRngThreadPool&&) = delete;
    ZoRngThreadPool& operator=(ZoRngThreadPool&&) = delete;

private:
    void init_pool() {
        shutdown();  // tear down old pool if any
        if (num_threads_ <= 1) return;  // single-thread: no pool needed
        shutdown_ = false;
        generation_ = 0;
        workers_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; i++) {
            workers_.emplace_back(&ZoRngThreadPool::worker_fn, this, i);
        }
    }

    void shutdown() {
        if (workers_.empty()) return;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            shutdown_ = true;
            generation_++;
        }
        start_cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
    }

    void worker_fn(int id) {
        int my_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mtx_);
                start_cv_.wait(lk, [&]{
                    return generation_ > my_gen || shutdown_;
                });
                if (shutdown_) return;
                my_gen = generation_;
            }
            // Static partitioning (same as OMP schedule(static))
            int64_t chunk = (total_n_ + num_threads_ - 1) / num_threads_;
            int64_t begin = static_cast<int64_t>(id) * chunk;
            int64_t end = std::min(begin + chunk, total_n_);
            if (begin < end) {
                body_(begin, end);
            }

            if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(mtx_);
                done_cv_.notify_one();
            }
        }
    }

    int num_threads_{0};
    std::vector<std::thread> workers_;

    std::mutex mtx_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::atomic<int> active_{0};
    bool shutdown_{false};
    int generation_{0};

    std::function<void(int64_t, int64_t)> body_;
    int64_t total_n_{0};
};


/* ---------------------------------------------------------------------------
 * ZoRngPoolRegistry — manages multiple independent ZoRngThreadPool instances.
 *
 * pool_id=0 is the default global pool (backward compatible with singleton).
 * pool_id>=1 are user-created pools via create_pool().
 * ------------------------------------------------------------------------- */
class ZoRngPoolRegistry {
public:
    static ZoRngPoolRegistry& instance() {
        static ZoRngPoolRegistry reg;
        return reg;
    }

    ZoRngThreadPool& get(int pool_id = 0) {
        if (pool_id == 0) return default_pool();
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = pools_.find(pool_id);
        if (it == pools_.end())
            throw std::runtime_error("zo_rng: invalid pool_id " + std::to_string(pool_id));
        return *it->second;
    }

    int create_pool(int num_threads) {
        std::lock_guard<std::mutex> lk(mtx_);
        int id = next_id_++;
        pools_[id] = std::make_unique<ZoRngThreadPool>(num_threads);
        return id;
    }

    void destroy_pool(int pool_id) {
        if (pool_id == 0)
            throw std::runtime_error("zo_rng: cannot destroy default pool (id=0)");
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = pools_.find(pool_id);
        if (it == pools_.end())
            throw std::runtime_error("zo_rng: invalid pool_id " + std::to_string(pool_id));
        pools_.erase(it);
    }

    // Backward compatible: operate on default pool
    void set_num_threads(int n) { default_pool().set_num_threads(n); }
    int get_num_threads() { return default_pool().get_num_threads(); }

    // Non-copyable
    ZoRngPoolRegistry(const ZoRngPoolRegistry&) = delete;
    ZoRngPoolRegistry& operator=(const ZoRngPoolRegistry&) = delete;

private:
    ZoRngPoolRegistry() = default;

    ZoRngThreadPool& default_pool() {
        std::call_once(default_init_, [this]{
            const char* env = std::getenv("ZO_RNG_NUM_THREADS");
            int n = 1;
            if (env) {
                n = std::atoi(env);
                if (n <= 0) n = 1;
            } else {
                int hw = static_cast<int>(std::thread::hardware_concurrency());
                n = (hw > 0) ? hw : 1;
            }
            default_pool_ = std::make_unique<ZoRngThreadPool>(n);
        });
        return *default_pool_;
    }

    std::unique_ptr<ZoRngThreadPool> default_pool_;
    std::once_flag default_init_;
    std::unordered_map<int, std::unique_ptr<ZoRngThreadPool>> pools_;
    std::mutex mtx_;
    int next_id_ = 1;  // 0 reserved for default
};


/* C interface (backward compatible) */
#ifdef __cplusplus
extern "C" {
#endif

static inline void zo_rng_set_num_threads(int n) {
    ZoRngPoolRegistry::instance().set_num_threads(n);
}

static inline int zo_rng_get_num_threads(void) {
    return ZoRngPoolRegistry::instance().get_num_threads();
}

#ifdef __cplusplus
}
#endif
