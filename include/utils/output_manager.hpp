#pragma once

#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <variant>
#include <typeindex>

#include "utils/h5_io.hpp"
#include "utils/device_fields.cuh"

namespace io {      

// Which index space this field belongs to
enum class IndexSpace { Particle, Vertex, System, None };
enum class Dimensionality { D1, D2 };

// Provider (no virtuals; store lambdas/struct functors) - templated for any type T
template<typename T>
struct Provider1D {
    std::function<void()> ensure_ready;                    // launch any prerequisites on the simulation stream
    std::function<df::DeviceField1D<T>*()> get_device;    // stable pointer to data (or staging)  
    IndexSpace index_space = IndexSpace::Particle;
    using value_type = T;  // for template metaprogramming
};

template<typename T>
struct Provider2D {
    std::function<void()> ensure_ready;
    std::function<df::DeviceField2D<T>*()> get_device;
    IndexSpace index_space = IndexSpace::Particle;
    using value_type = T;  // for template metaprogramming
};

// Variant types for heterogeneous Provider storage (matching DeviceField type instantiations)
using ProviderVariant1D = std::variant<
    Provider1D<float>, Provider1D<double>, Provider1D<int>,
    Provider1D<unsigned int>, Provider1D<long>, Provider1D<unsigned long long>,
    Provider1D<unsigned char>
>;

using ProviderVariant2D = std::variant<
    Provider2D<float>, Provider2D<double>, Provider2D<int>,
    Provider2D<unsigned int>, Provider2D<long>, Provider2D<unsigned long long>,
    Provider2D<unsigned char>
>;

struct FieldDesc {
    Dimensionality dim;
    IndexSpace index_space;
    std::type_index type_info;        // runtime type identification
    ProviderVariant1D p1d_variant;    // used if dim == D1
    ProviderVariant2D p2d_variant;    // used if dim == D2
    
    // Default constructor for std::map usage
    FieldDesc() : dim(Dimensionality::D1), index_space(IndexSpace::None), type_info(std::type_index(typeid(void))) {}
    
    // Constructor to initialize type_info properly
    template<typename T>
    FieldDesc(Dimensionality d, IndexSpace space, Provider1D<T> p1d) 
        : dim(d), index_space(space), type_info(std::type_index(typeid(T))), p1d_variant(std::move(p1d)) {}
    
    template<typename T>
    FieldDesc(Dimensionality d, IndexSpace space, Provider2D<T> p2d)
        : dim(d), index_space(space), type_info(std::type_index(typeid(T))), p2d_variant(std::move(p2d)) {}
};

struct OutputRegistry {
    std::map<std::string, FieldDesc> fields; // name -> descriptor
};

// Host buffers - templated for any type T
template<typename T>
struct Host1D { 
    std::vector<T> v; 
    int N = 0; 
    IndexSpace space = IndexSpace::None; 
    using value_type = T;  // for template metaprogramming
};

template<typename T>
struct Host2D { 
    std::vector<T> x, y; 
    int N = 0; 
    IndexSpace space = IndexSpace::None; 
    using value_type = T;  // for template metaprogramming
};

// Variant types for heterogeneous Host storage
using HostVariant1D = std::variant<
    Host1D<float>, Host1D<double>, Host1D<int>,
    Host1D<unsigned int>, Host1D<long>, Host1D<unsigned long long>,
    Host1D<unsigned char>
>;

using HostVariant2D = std::variant<
    Host2D<float>, Host2D<double>, Host2D<int>,
    Host2D<unsigned int>, Host2D<long>, Host2D<unsigned long long>,
    Host2D<unsigned char>
>;

// Cached dataset entry (one per field)
struct DsetCacheEntry {
    hid_t dset = -1;
    Dimensionality dim = Dimensionality::D1;
    hsize_t N = 0;           // size of entity dimension
    hsize_t reserved_T = 0;  // capacity in time dimension
    hsize_t cursor_T = 0;    // written timesteps
};

enum class TaskKind { Trajectory, Restart, FinalInit, FinalEnd };

struct SaveTask {
    TaskKind kind;
    int step = -1; // for trajectory/restart labeling
    std::map<std::string, HostVariant1D> one_d;  // now stores type variants
    std::map<std::string, HostVariant2D> two_d;  // now stores type variants
    // inverse maps for reindexing
    bool have_inv_particles = false;
    bool have_inv_vertices = false;
    std::vector<int> inv_particles; // new->orig
    std::vector<int> inv_vertices;  // new->orig
    // backpressure accounting
    std::size_t bytes = 0;
};

template<class ParticleType>
class OutputManager {
public:
    OutputManager(ParticleType& particles,
                  const std::string& path,
                  int max_workers = 1,
                  bool append_mode = false)
    : particles_(particles), path_(path),
      max_workers_(max_workers), append_mode_(append_mode) {}

    // Policy configuration
    void set_trajectory_fields(const std::vector<std::string>& names) { traj_names_ = names; }
    void set_extra_init_fields(const std::vector<std::string>& names) { extra_init_names_ = names; }
    void set_extra_final_fields(const std::vector<std::string>& names){ extra_final_names_ = names; }
    void set_extra_static_fields(const std::vector<std::string>& names){ extra_static_names_ = names; }
    void set_extra_restart_fields(const std::vector<std::string>& names){ extra_restart_names_ = names; }
    void set_trajectory_interval(int v) { traj_interval_ = v; }
    void set_restart_interval(int v)    { restart_interval_ = v; }
    void set_queue_limit(std::size_t tasks)  { queue_limit_ = tasks; }
    void set_bytes_limit(std::size_t bytes)  { bytes_limit_ = bytes; }
    void set_preextend_block(int K)          { preextend_block_ = K; }
    void set_append_mode(bool on)            { append_mode_ = on; }
    void set_resume_from_restart(bool on)    { resume_from_restart_ = on; }
    // Opt-outs
    void enable_meta(bool on)        { enable_meta_ = on; }
    void enable_trajectory(bool on)  { enable_traj_ = on; }
    void enable_console(bool on)     { enable_console_ = on; }
    void set_console_fields(const std::vector<std::string>& names) { console_names_ = names; }

    // Lifecycle
    void initialize() {
        meta_path_ = path_ + "/meta.h5";
        traj_path_ = path_ + "/trajectory.h5";
        // Create parent directories if needed
        if (enable_meta_) {
            std::filesystem::path parent_path = std::filesystem::path(meta_path_).parent_path();
            if (!std::filesystem::exists(parent_path)) {
                std::filesystem::create_directories(parent_path);
            }
        }
        if (enable_traj_) {
            std::filesystem::path parent_path = std::filesystem::path(traj_path_).parent_path();
            if (!std::filesystem::exists(parent_path)) {
                std::filesystem::create_directories(parent_path);
            }
        }

        // Handle file creation/opening based on append_mode
        bool meta_exists = false, traj_exists = false;

        if (enable_meta_) {
            if (!append_mode_ && std::filesystem::exists(meta_path_)) {
                std::filesystem::remove(meta_path_);
            }

            const bool want_append = append_mode_ && std::filesystem::exists(meta_path_);
            if (want_append) {
                hid_t f = H5Fopen(meta_path_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                if (f >= 0) {
                    meta_file_ = f;
                    meta_exists = true;
                } else {
                    // Fallback: create if open unexpectedly fails
                    meta_file_ = H5Fcreate(meta_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                    if (meta_file_ < 0) throw std::runtime_error("OutputManager::initialize: cannot create meta file");
                }
            } else {
                meta_file_ = H5Fcreate(meta_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (meta_file_ < 0) throw std::runtime_error("OutputManager::initialize: cannot create meta file");
            }
        }

        if (enable_traj_) {
            if (!append_mode_ && std::filesystem::exists(traj_path_)) {
                std::filesystem::remove(traj_path_);
            }

            const bool want_append = append_mode_ && std::filesystem::exists(traj_path_);
            if (want_append) {
                hid_t f = H5Fopen(traj_path_.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
                if (f >= 0) {
                    traj_file_ = f;
                    traj_exists = true;
                } else {
                    // Fallback: create if open unexpectedly fails
                    traj_file_ = H5Fcreate(traj_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                    if (traj_file_ < 0) throw std::runtime_error("OutputManager::initialize: cannot create trajectories file");
                }
            } else {
                traj_file_ = H5Fcreate(traj_path_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                if (traj_file_ < 0) throw std::runtime_error("OutputManager::initialize: cannot create trajectories file");
            }
        }

        OutputRegistry reg; build_registry_safe(reg);

        if (enable_meta_) {
            const bool do_write_static_init = !(append_mode_ && meta_exists);
            if (do_write_static_init) {
                // Resolve names from particle API
                std::vector<std::string> static_names = particles_.get_static_field_names();
                std::vector<std::string> state_names  = particles_.get_state_field_names();
                // Append extras
                if (!extra_static_names_.empty()) static_names.insert(static_names.end(), extra_static_names_.begin(), extra_static_names_.end());
                if (!extra_init_names_.empty())   state_names.insert(state_names.end(),   extra_init_names_.begin(),   extra_init_names_.end());
                // static group: open if exists, else create
                {
                    hid_t g = h5_link_exists(meta_file_, "/static")
                              ? H5Gopen(meta_file_, "static", H5P_DEFAULT)
                              : H5Gcreate2(meta_file_, "static", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    if (g >= 0) {
                        // Tag class name as dataset only (no attribute)
                        write_scalar<std::string>(g, "class_name", particles_.get_class_name());
                        // Record neighbor method used
                        write_scalar<std::string>(g, "neighbor_method", particles_.neighbor_method_to_string());
                        // Also record scalar counts
                        write_scalar<int>(g, "n_particles", particles_.n_particles());
                        write_scalar<int>(g, "n_systems",  particles_.n_systems());
                        write_scalar<int>(g, "n_vertices",  particles_.n_vertices());
                        // Snapshot and write static fields
                        SaveTask t; t.kind = TaskKind::FinalInit; capture_registry_to_host(reg, static_names, t);
                        write_host_maps_to_group(g, t);
                        H5Gclose(g);
                    }
                }
                // init group: open if exists, else create
                {
                    hid_t g = h5_link_exists(meta_file_, "/init")
                              ? H5Gopen(meta_file_, "init", H5P_DEFAULT)
                              : H5Gcreate2(meta_file_, "init", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                    if (g >= 0) {
                        // Snapshot and write non-static state fields (+extras already merged)
                        SaveTask t; t.kind = TaskKind::FinalInit; capture_registry_to_host(reg, state_names, t);
                        write_host_maps_to_group(g, t);
                        H5Gclose(g);
                    }
                }
                H5Fflush(meta_file_, H5F_SCOPE_GLOBAL);
            }
            // If appending (not rewriting static), still ensure class_name attribute exists/updated
            else {
                if (h5_link_exists(meta_file_, "/static")) {
                    hid_t g = H5Gopen(meta_file_, "static", H5P_DEFAULT);
                    if (g >= 0) {
                        // Ensure datasets exist/updated when appending
                        write_scalar<std::string>(g, "class_name", particles_.get_class_name());
                        write_scalar<std::string>(g, "neighbor_method", particles_.neighbor_method_to_string());
                        write_scalar<int>(g, "n_particles", particles_.n_particles());
                        write_scalar<int>(g, "n_systems",  particles_.n_systems());
                        write_scalar<int>(g, "n_vertices",  particles_.n_vertices());
                        H5Gclose(g);
                    }
                }
            }
            if (!h5_link_exists(meta_file_, "/restart")) { if (hid_t g = H5Gcreate2(meta_file_, "restart", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); g >= 0) H5Gclose(g); }
        }

        // If resuming, read step from /restart and trim trajectories beyond it
        if (resume_from_restart_ && enable_meta_ && meta_file_ >= 0) {
            int rstep = -1;
            if (h5_link_exists(meta_file_, "/restart/step")) {
                rstep = read_scalar<int>(meta_file_, "/restart/step");
                last_restart_step_ = rstep;
            }
            if (enable_traj_ && traj_file_ >= 0 && rstep >= 0) {
                trim_trajectories_to_restart(reg, rstep);
            }
        }

        start_workers();
    }

    void step(int step_num) {
        current_step_ = step_num;
        OutputRegistry reg; build_registry_safe(reg);

        // Console logging if enabled and fields provided
        if (enable_console_ && !console_names_.empty()) {
            SaveTask t; t.kind = TaskKind::FinalInit; capture_registry_to_host(reg, console_names_, t);
            log_console(step_num, t);
        }

        if (enable_traj_ && traj_interval_ > 0 && (step_num % traj_interval_) == 0 && !traj_names_.empty()) {
            SaveTask t; t.kind = TaskKind::Trajectory; t.step = step_num; capture_registry_to_host(reg, traj_names_, t);
            enqueue_blocking(std::move(t));
        }

        if (enable_meta_ && restart_interval_ > 0 && (step_num % restart_interval_) == 0) {
            // restart = static + state + extras
            std::vector<std::string> names = particles_.get_static_field_names();
            {
                std::vector<std::string> state = particles_.get_state_field_names();
                names.insert(names.end(), state.begin(), state.end());
            }
            if (!extra_static_names_.empty())  names.insert(names.end(), extra_static_names_.begin(),  extra_static_names_.end());
            if (!extra_restart_names_.empty()) names.insert(names.end(), extra_restart_names_.begin(), extra_restart_names_.end());
            SaveTask t; t.kind = TaskKind::Restart; t.step = step_num; capture_registry_to_host(reg, names, t);
            enqueue_blocking(std::move(t));
        }
    }

    void finalize() {
        if (enable_meta_) {
            OutputRegistry reg; build_registry_safe(reg);
            // final restart (static + state + extras)
            {
                std::vector<std::string> names = particles_.get_static_field_names();
                std::vector<std::string> state = particles_.get_state_field_names();
                names.insert(names.end(), state.begin(), state.end());
                if (!extra_static_names_.empty())  names.insert(names.end(), extra_static_names_.begin(),  extra_static_names_.end());
                if (!extra_restart_names_.empty()) names.insert(names.end(), extra_restart_names_.begin(), extra_restart_names_.end());
                SaveTask t; t.kind = TaskKind::Restart; t.step = current_step_; capture_registry_to_host(reg, names, t);
                enqueue_blocking(std::move(t));
            }
            // final state + extra_final to /final
            {
                std::vector<std::string> names = particles_.get_state_field_names();
                if (!extra_final_names_.empty()) names.insert(names.end(), extra_final_names_.begin(), extra_final_names_.end());
                SaveTask t; t.kind = TaskKind::FinalEnd; t.step = current_step_; capture_registry_to_host(reg, names, t);
                // synchronous write to /final using same path as write_final_extras
                std::lock_guard<std::mutex> lk(h5_mtx_);
                hid_t g = h5_link_exists(meta_file_, "/final")
                          ? H5Gopen(meta_file_, "final", H5P_DEFAULT)
                          : H5Gcreate2(meta_file_, "final", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (g >= 0) { write_host_maps_to_group(g, t); H5Gclose(g); H5Fflush(meta_file_, H5F_SCOPE_GLOBAL); }
            }
        }

        stop_workers();

        if (enable_traj_) {
            std::lock_guard<std::mutex> lk(h5_mtx_);
            for (auto& kv : traj_cache_) shrink_to_cursor_locked(kv.second);
            if (timestep_cache_.dset >= 0) shrink_to_cursor_locked(timestep_cache_);
            H5Fflush(traj_file_, H5F_SCOPE_GLOBAL);
        }

        if (traj_file_ >= 0) { H5Fflush(traj_file_, H5F_SCOPE_GLOBAL); H5Fclose(traj_file_); traj_file_ = -1; }
        if (meta_file_ >= 0) { H5Fflush(meta_file_, H5F_SCOPE_GLOBAL); H5Fclose(meta_file_); meta_file_ = -1; }
    }

private:
    // Particle and files
    ParticleType& particles_;
    std::string path_;
    std::string meta_path_;
    std::string traj_path_;
    hid_t meta_file_ = -1;
    hid_t traj_file_ = -1;

    // policy
    int traj_interval_ = 0;
    int restart_interval_ = 0;
    int preextend_block_ = 16;
    int current_step_ = 0;
    std::size_t queue_limit_ = 0;  // tasks
    std::size_t bytes_limit_ = 0;  // bytes in flight (0 => unlimited)
    bool append_mode_ = false;
    bool resume_from_restart_ = true;
    int  last_restart_step_ = -1;   // read from /restart/step if present

    // planned fields
    std::vector<std::string> traj_names_;
    std::vector<std::string> extra_init_names_;
    std::vector<std::string> extra_final_names_;
    std::vector<std::string> extra_static_names_;
    std::vector<std::string> extra_restart_names_;
    // opt-outs and console configuration
    bool enable_meta_ = true;
    bool enable_traj_ = true;
    bool enable_console_ = false;
    std::vector<std::string> console_names_;

    // HDF5 caches and lock
    std::mutex h5_mtx_;
    std::map<std::string, DsetCacheEntry> traj_cache_;
    DsetCacheEntry timestep_cache_{-1, Dimensionality::D1, 1, 0, 0};

    // executor: workers and backpressure
    int max_workers_ = 1;
    std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::queue<SaveTask> tasks_;
    std::vector<std::thread> workers_;
    std::atomic<bool> shutting_down_{false};
    std::size_t bytes_in_flight_ = 0;

    // ---- registry building (SFINAE to keep CRTP clean) ----
    template<class T=ParticleType>
    static auto has_build_registry_impl(int) -> decltype(std::declval<T&>().output_build_registry(std::declval<OutputRegistry&>()), std::true_type{});
    template<class>
    static std::false_type has_build_registry_impl(...);

    template<class T=ParticleType>
    static auto has_capture_inv_impl(int) -> decltype(std::declval<T&>().output_capture_inverse_orders(std::declval<std::vector<int>&>(), std::declval<std::vector<int>&>()), std::true_type{});
    template<class>
    static std::false_type has_capture_inv_impl(...);

    void build_registry_safe(OutputRegistry& reg) {
        if constexpr (decltype(has_build_registry_impl<ParticleType>(0))::value) {
            particles_.output_build_registry(reg);
        }
    }

    void capture_inverse_orders(SaveTask& t) {
        if constexpr (decltype(has_capture_inv_impl<ParticleType>(0))::value) {
            particles_.output_capture_inverse_orders(t.inv_particles, t.inv_vertices);
            t.have_inv_particles = !t.inv_particles.empty();
            t.have_inv_vertices  = !t.inv_vertices.empty();
        }
    }

    // ---- snapshot and reindex (same-stream D2H) ----
    void capture_registry_to_host(const OutputRegistry& reg, const std::vector<std::string>& names, SaveTask& out) {
        capture_inverse_orders(out);
        std::size_t bytes = 0;
        for (const auto& name : names) {
            auto it = reg.fields.find(name);
            if (it == reg.fields.end()) continue;
            const FieldDesc& fd = it->second;
            
            if (fd.dim == Dimensionality::D1) {
                // Use std::visit to handle different types at runtime
                std::visit([&](const auto& provider) {
                    using T = typename std::decay_t<decltype(provider)>::value_type;
                    if (provider.ensure_ready) provider.ensure_ready();
                    auto* dev = provider.get_device ? provider.get_device() : nullptr;
                    if (!dev || dev->size() == 0) return;
                    
                    Host1D<T> hb; 
                    hb.N = dev->size(); 
                    hb.space = fd.index_space; 
                    hb.v.resize(static_cast<std::size_t>(hb.N));
                    dev->to_host(hb.v); // blocking; sim-safe (same stream)
                    apply_inverse(hb, out);
                    bytes += hb.v.size() * sizeof(T);
                    out.one_d.emplace(name, HostVariant1D{std::move(hb)});
                }, fd.p1d_variant);
            } else {
                // Handle 2D fields similarly
                std::visit([&](const auto& provider) {
                    using T = typename std::decay_t<decltype(provider)>::value_type;
                    if (provider.ensure_ready) provider.ensure_ready();
                    auto* dev = provider.get_device ? provider.get_device() : nullptr;
                    if (!dev || dev->size() == 0) return;
                    
                    Host2D<T> hb; 
                    hb.N = dev->size(); 
                    hb.space = fd.index_space; 
                    dev->to_host(hb.x, hb.y);
                    apply_inverse(hb, out);
                    bytes += (hb.x.size() + hb.y.size()) * sizeof(T);
                    out.two_d.emplace(name, HostVariant2D{std::move(hb)});
                }, fd.p2d_variant);
            }
        }
        out.bytes = bytes;
    }

    template<typename T>
    static void apply_inverse(Host1D<T>& hb, const SaveTask& t) {
        if (hb.space == IndexSpace::Particle && t.have_inv_particles && static_cast<int>(t.inv_particles.size()) == hb.N) {
            std::vector<T> out(static_cast<std::size_t>(hb.N));
            for (int i=0;i<hb.N;++i) out[static_cast<std::size_t>(t.inv_particles[i])] = hb.v[static_cast<std::size_t>(i)];
            hb.v.swap(out);
        } else if (hb.space == IndexSpace::Vertex && t.have_inv_vertices && static_cast<int>(t.inv_vertices.size()) == hb.N) {
            std::vector<T> out(static_cast<std::size_t>(hb.N));
            for (int i=0;i<hb.N;++i) out[static_cast<std::size_t>(t.inv_vertices[i])] = hb.v[static_cast<std::size_t>(i)];
            hb.v.swap(out);
        }
    }

    template<typename T>
    static void apply_inverse(Host2D<T>& hb, const SaveTask& t) {
        if (hb.space == IndexSpace::Particle && t.have_inv_particles && static_cast<int>(t.inv_particles.size()) == hb.N) {
            std::vector<T> rx(static_cast<std::size_t>(hb.N)), ry(static_cast<std::size_t>(hb.N));
            for (int i=0;i<hb.N;++i) { int o=t.inv_particles[i]; rx[static_cast<std::size_t>(o)] = hb.x[static_cast<std::size_t>(i)]; ry[static_cast<std::size_t>(o)] = hb.y[static_cast<std::size_t>(i)]; }
            hb.x.swap(rx); hb.y.swap(ry);
        } else if (hb.space == IndexSpace::Vertex && t.have_inv_vertices && static_cast<int>(t.inv_vertices.size()) == hb.N) {
            std::vector<T> rx(static_cast<std::size_t>(hb.N)), ry(static_cast<std::size_t>(hb.N));
            for (int i=0;i<hb.N;++i) { int o=t.inv_vertices[i]; rx[static_cast<std::size_t>(o)] = hb.x[static_cast<std::size_t>(i)]; ry[static_cast<std::size_t>(o)] = hb.y[static_cast<std::size_t>(i)]; }
            hb.x.swap(rx); hb.y.swap(ry);
        }
    }

    // ---- worker management with blocking backpressure ----
    void start_workers() {
        if (max_workers_ < 1) max_workers_ = 1;
        if (queue_limit_ == 0) queue_limit_ = static_cast<std::size_t>(max_workers_);
        for (int i=0;i<max_workers_;++i) workers_.emplace_back([this]{ worker_loop(); });
    }

    void stop_workers() {
        {
            std::lock_guard<std::mutex> lk(q_mtx_);
            shutting_down_.store(true, std::memory_order_relaxed);
        }
        q_cv_.notify_all();
        for (auto& w : workers_) if (w.joinable()) w.join();
        workers_.clear();
    }

    void enqueue_blocking(SaveTask&& t) {
        if (max_workers_ == 1) { process_task(t); return; }
        std::unique_lock<std::mutex> lk(q_mtx_);
        q_cv_.wait(lk, [this,&t]{
            bool q_ok = tasks_.size() < queue_limit_;
            bool b_ok = (bytes_limit_ == 0) || (bytes_in_flight_ + t.bytes <= bytes_limit_);
            return shutting_down_.load(std::memory_order_relaxed) || (q_ok && b_ok);
        });
        bytes_in_flight_ += t.bytes;
        tasks_.push(std::move(t));
        lk.unlock();
        q_cv_.notify_one();
    }

    void worker_loop() {
        for (;;) {
            SaveTask t;
            {
                std::unique_lock<std::mutex> lk(q_mtx_);
                q_cv_.wait(lk, [this]{ return shutting_down_.load(std::memory_order_relaxed) || !tasks_.empty(); });
                if (shutting_down_.load(std::memory_order_relaxed) && tasks_.empty()) return;
                t = std::move(tasks_.front()); tasks_.pop();
            }
            process_task(t);
            {
                std::lock_guard<std::mutex> lk(q_mtx_);
                if (bytes_in_flight_ >= t.bytes) bytes_in_flight_ -= t.bytes; else bytes_in_flight_ = 0;
            }
            q_cv_.notify_all();
        }
    }

    void process_task(const SaveTask& t) {
        switch (t.kind) {
            case TaskKind::Trajectory: write_trajectory(t); break;
            case TaskKind::Restart:    write_restart_atomic(t); break;
            case TaskKind::FinalInit:  /* already handled in initialize */ break;
            case TaskKind::FinalEnd:   write_final_extras(t); break;
        }
    }

    // ---- HDF5 helpers: cached datasets + pre-extend ----
    void ensure_timestep_locked() {
        if (timestep_cache_.dset >= 0) return;
        if (h5_link_exists(traj_file_, "/timestep")) {
            hid_t d = H5Dopen2(traj_file_, "/timestep", H5P_DEFAULT);
            if (d < 0) throw std::runtime_error("ensure_timestep_locked: open failed");
            timestep_cache_.dset = d; timestep_cache_.dim = Dimensionality::D1; timestep_cache_.N = 1;
            auto dims = get_dims(d);
            hsize_t T = (dims.size()>=1)?dims[0]:0;
            timestep_cache_.reserved_T = T; timestep_cache_.cursor_T = T;
            return;
        }
        hsize_t dims[2] = {0, 1}, maxd[2] = {H5S_UNLIMITED, 1};
        hid_t space = H5Screate_simple(2, dims, maxd);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        hsize_t chunks[2] = {static_cast<hsize_t>(std::max(1, preextend_block_)), 1};
        H5Pset_chunk(dcpl, 2, chunks);
        hid_t d = H5Dcreate2(traj_file_, "/timestep", H5T_NATIVE_INT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space);
        if (d < 0) throw std::runtime_error("ensure_timestep_locked: create failed");
        timestep_cache_.dset = d; timestep_cache_.dim = Dimensionality::D1; timestep_cache_.N = 1;
        timestep_cache_.reserved_T = 0; timestep_cache_.cursor_T = 0;
    }

    void ensure_traj_dataset_locked(const std::string& name, Dimensionality dim, hsize_t N) {
        auto it = traj_cache_.find(name);
        if (it != traj_cache_.end()) return;
        DsetCacheEntry e; e.dim = dim; e.N = N; e.reserved_T = 0; e.cursor_T = 0;
        // If exists, open and initialize from current dims
        if (h5_link_exists(traj_file_, ("/"+name).c_str())) {
            hid_t d = H5Dopen2(traj_file_, ("/"+name).c_str(), H5P_DEFAULT);
            if (d < 0) throw std::runtime_error("ensure_traj_dataset_locked: open failed for "+name);
            auto dims = get_dims(d);
            if (dim == Dimensionality::D1) {
                if (!(dims.size()==2 && dims[1]==N)) { H5Dclose(d); throw std::runtime_error("ensure_traj_dataset_locked: dim mismatch for "+name); }
                e.dset = d; e.reserved_T = dims[0]; e.cursor_T = dims[0];
            } else {
                if (!(dims.size()==3 && dims[1]==N && dims[2]==2)) { H5Dclose(d); throw std::runtime_error("ensure_traj_dataset_locked: dim mismatch for "+name); }
                e.dset = d; e.reserved_T = dims[0]; e.cursor_T = dims[0];
            }
            traj_cache_.emplace(name, e);
            return;
        }
        // Otherwise create fresh
        hid_t space, dcpl;
        if (dim == Dimensionality::D1) {
            hsize_t dims2[2] = {0, N}, maxd[2] = {H5S_UNLIMITED, N};
            space = H5Screate_simple(2, dims2, maxd);
            dcpl  = H5Pcreate(H5P_DATASET_CREATE);
            hsize_t chunks[2] = {static_cast<hsize_t>(std::max(1, preextend_block_)), N};
            H5Pset_chunk(dcpl, 2, chunks);
        } else {
            hsize_t dims3[3] = {0, N, 2}, maxd3[3] = {H5S_UNLIMITED, N, 2};
            space = H5Screate_simple(3, dims3, maxd3);
            dcpl  = H5Pcreate(H5P_DATASET_CREATE);
            hsize_t chunks[3] = {static_cast<hsize_t>(std::max(1, preextend_block_)), N, 2};
            H5Pset_chunk(dcpl, 3, chunks);
        }
        hid_t dnew = H5Dcreate2(traj_file_, ("/"+name).c_str(), H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(space);
        if (dnew < 0) throw std::runtime_error("ensure_traj_dataset_locked: create failed for "+name);
        e.dset = dnew; traj_cache_.emplace(name, e);
    }

    static void extend_locked(DsetCacheEntry& e, hsize_t add_T) {
        if (add_T == 0) return;
        hsize_t newT = e.reserved_T + add_T;
        if (e.dim == Dimensionality::D1) { hsize_t dims[2] = {newT, e.N}; H5Dset_extent(e.dset, dims); }
        else { hsize_t dims[3] = {newT, e.N, 2}; H5Dset_extent(e.dset, dims); }
        e.reserved_T = newT;
    }

    static void shrink_to_cursor_locked(DsetCacheEntry& e) {
        if (e.dset < 0) return;
        if (e.dim == Dimensionality::D1) { hsize_t dims[2] = {e.cursor_T, e.N}; H5Dset_extent(e.dset, dims); }
        else { hsize_t dims[3] = {e.cursor_T, e.N, 2}; H5Dset_extent(e.dset, dims); }
    }

    void write_trajectory(const SaveTask& t) {
        std::lock_guard<std::mutex> lk(h5_mtx_);
        ensure_timestep_locked();
        if (timestep_cache_.cursor_T + 1 > timestep_cache_.reserved_T) extend_locked(timestep_cache_, static_cast<hsize_t>(preextend_block_));
        {
            hid_t fs = H5Dget_space(timestep_cache_.dset); hsize_t start[2] = {timestep_cache_.cursor_T, 0}; hsize_t cnt[2] = {1,1};
            H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
            hid_t ms = H5Screate_simple(2, cnt, nullptr);
            int step = t.step; H5Dwrite(timestep_cache_.dset, H5T_NATIVE_INT, ms, fs, H5P_DEFAULT, &step);
            H5Sclose(ms); H5Sclose(fs);
            timestep_cache_.cursor_T += 1;
        }

        for (const auto& kv : t.one_d) {
            const std::string& name = kv.first;
            std::visit([&](const auto& hb) {
                using T = typename std::decay_t<decltype(hb)>::value_type;
                hsize_t N = static_cast<hsize_t>(hb.N);
                ensure_traj_dataset_locked(name, Dimensionality::D1, N);
                auto& e = traj_cache_[name];
                // Align cursor with timestep
                if (e.cursor_T != timestep_cache_.cursor_T) e.cursor_T = timestep_cache_.cursor_T;
                if (e.cursor_T + 1 > e.reserved_T) extend_locked(e, static_cast<hsize_t>(preextend_block_));
                hid_t fs = H5Dget_space(e.dset); hsize_t start[2] = {e.cursor_T, 0}; hsize_t cnt[2] = {1, N};
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
                hid_t ms = H5Screate_simple(2, cnt, nullptr);
                H5Dwrite(e.dset, h5_native<T>(), ms, fs, H5P_DEFAULT, hb.v.data());
                H5Sclose(ms); H5Sclose(fs);
                e.cursor_T += 1;
            }, kv.second);
        }
        for (const auto& kv : t.two_d) {
            const std::string& name = kv.first;
            std::visit([&](const auto& hb) {
                using T = typename std::decay_t<decltype(hb)>::value_type;
                hsize_t N = static_cast<hsize_t>(hb.N);
                ensure_traj_dataset_locked(name, Dimensionality::D2, N);
                auto& e = traj_cache_[name];
                if (e.cursor_T != timestep_cache_.cursor_T) e.cursor_T = timestep_cache_.cursor_T;
                if (e.cursor_T + 1 > e.reserved_T) extend_locked(e, static_cast<hsize_t>(preextend_block_));
                hid_t fs = H5Dget_space(e.dset); hsize_t start[3] = {e.cursor_T, 0, 0}; hsize_t cnt[3] = {1, N, 2};
                H5Sselect_hyperslab(fs, H5S_SELECT_SET, start, nullptr, cnt, nullptr);
                hid_t ms = H5Screate_simple(3, cnt, nullptr);
                std::vector<T> inter(static_cast<std::size_t>(hb.N) * 2);
                for (int i=0;i<hb.N;++i) { inter[2*i+0]=hb.x[static_cast<std::size_t>(i)]; inter[2*i+1]=hb.y[static_cast<std::size_t>(i)]; }
                H5Dwrite(e.dset, h5_native<T>(), ms, fs, H5P_DEFAULT, inter.data());
                H5Sclose(ms); H5Sclose(fs);
                e.cursor_T += 1;
            }, kv.second);
        }
    }

    void trim_trajectories_to_restart(OutputRegistry& reg, int restart_step) {
        std::lock_guard<std::mutex> lk(h5_mtx_);
        // Require /timestep to exist and be int-compatible
        if (!h5_link_exists(traj_file_, "/timestep")) return;
        hid_t dset = H5Dopen2(traj_file_, "/timestep", H5P_DEFAULT);
        if (dset < 0) return;
        auto dims = get_dims(dset);
        hsize_t T = (dims.size()>=1)?dims[0]:0;
        std::vector<int> ts(static_cast<std::size_t>(T));
        {
            hid_t space = H5Dget_space(dset);
            if (dims.size()==2) {
                // stored as (T,1); read as a flat vector of T elements via hyperslab
                hsize_t start[2] = {0,0}, count[2] = {T,1};
                H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr);
                hid_t ms = H5Screate_simple(2, count, nullptr);
                H5Dread(dset, H5T_NATIVE_INT, ms, space, H5P_DEFAULT, ts.data());
                H5Sclose(ms);
            } else {
                // fallback: treat as (T)
                H5Dread(dset, H5T_NATIVE_INT, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, ts.data());
            }
            H5Sclose(space);
        }
        // Find first index with timestep > restart_step
        hsize_t keep_T = 0;
        for (hsize_t i=0;i<T;++i) { if (ts[static_cast<std::size_t>(i)] > restart_step) { keep_T = i; break; } }
        if (keep_T == 0) {
            // if none greater was found, keep all
            if (T>0 && ts.back() <= restart_step) keep_T = T; else keep_T = 0;
        }
        // Shrink /timestep
        if (keep_T < T) {
            if (dims.size()==2) { hsize_t nd[2] = {keep_T, 1}; H5Dset_extent(dset, nd); }
            else { hsize_t nd[1] = {keep_T}; H5Dset_extent(dset, nd); }
        }
        H5Dclose(dset);
        // Initialize caches for registered fields and shrink to keep_T
        for (auto& kv : reg.fields) {
            const std::string& name = kv.first; const FieldDesc& fd = kv.second;
            if (fd.dim == Dimensionality::D1) {
                std::visit([&](const auto& provider) {
                    if (provider.get_device) {
                        auto* dev = provider.get_device(); 
                        if (!dev) return; 
                        ensure_traj_dataset_locked(name, Dimensionality::D1, static_cast<hsize_t>(dev->size()));
                    }
                }, fd.p1d_variant);
            } else {
                std::visit([&](const auto& provider) {
                    if (provider.get_device) {
                        auto* dev = provider.get_device(); 
                        if (!dev) return; 
                        ensure_traj_dataset_locked(name, Dimensionality::D2, static_cast<hsize_t>(dev->size()));
                    }
                }, fd.p2d_variant);
            }
        }
        // Shrink all cached datasets
        for (auto& ckv : traj_cache_) {
            DsetCacheEntry& e = ckv.second;
            if (keep_T < e.reserved_T) {
                if (e.dim == Dimensionality::D1) { hsize_t nd[2] = {keep_T, e.N}; H5Dset_extent(e.dset, nd); }
                else { hsize_t nd[3] = {keep_T, e.N, 2}; H5Dset_extent(e.dset, nd); }
                e.reserved_T = keep_T; e.cursor_T = keep_T;
            } else {
                e.cursor_T = keep_T; // advance cursor to keep_T even if equal
            }
        }
        // Also update cached timestep
        if (timestep_cache_.dset >= 0) { timestep_cache_.reserved_T = keep_T; timestep_cache_.cursor_T = keep_T; }
    }

    void write_restart_atomic(const SaveTask& t) {
        std::lock_guard<std::mutex> lk(h5_mtx_);
        if (h5_link_exists(meta_file_, "/restart_tmp")) H5Ldelete(meta_file_, "/restart_tmp", H5P_DEFAULT);
        hid_t g = H5Gcreate2(meta_file_, "restart_tmp", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g < 0) return;
        write_scalar<int>(g, "step", t.step);
        write_host_maps_to_group(g, t);
        H5Gclose(g);
        if (h5_link_exists(meta_file_, "/restart")) H5Ldelete(meta_file_, "/restart", H5P_DEFAULT);
        H5Lmove(meta_file_, "/restart_tmp", meta_file_, "/restart", H5P_DEFAULT, H5P_DEFAULT);
        H5Fflush(meta_file_, H5F_SCOPE_GLOBAL);
    }

    void write_final_extras(const SaveTask& t) {
        std::lock_guard<std::mutex> lk(h5_mtx_);
        hid_t g = h5_link_exists(meta_file_, "/final")
                  ? H5Gopen(meta_file_, "final", H5P_DEFAULT)
                  : H5Gcreate2(meta_file_, "final", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        write_host_maps_to_group(g, t);
        H5Gclose(g);
        H5Fflush(meta_file_, H5F_SCOPE_GLOBAL);
    }

    static void write_host_maps_to_group(hid_t group, const SaveTask& t) {
        // Use std::visit to dispatch on the actual type stored in the variant
        for (const auto& kv : t.one_d) {
            std::visit([&](const auto& host_data) {
                using T = typename std::decay_t<decltype(host_data)>::value_type;
                write_vector<T>(group, kv.first, host_data.v);
            }, kv.second);
        }
        
        for (const auto& kv : t.two_d) {
            std::visit([&](const auto& host_data) {
                using T = typename std::decay_t<decltype(host_data)>::value_type;
                write_vector_2d<T>(group, kv.first, host_data.x, host_data.y);
            }, kv.second);
        }
    }

    // Optional particle save helpers (SFINAE-guarded)
    // TODO: do we need these anymore?
    // VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    template<class T=ParticleType>
    static auto has_save_static_impl(int) -> decltype(std::declval<T&>().save_static_to_hdf5_group(std::declval<hid_t>()), std::true_type{});
    template<class>
    static std::false_type has_save_static_impl(...);

    template<class T=ParticleType>
    static auto has_save_state_impl(int) -> decltype(std::declval<T&>().save_to_hdf5_group(std::declval<hid_t>()), std::true_type{});
    template<class>
    static std::false_type has_save_state_impl(...);

    void call_save_static_safe(hid_t g) {
        if constexpr (decltype(has_save_static_impl<ParticleType>(0))::value) particles_.save_static_to_hdf5_group(g);
    }
    void call_save_state_safe(hid_t g) {
        if constexpr (decltype(has_save_state_impl<ParticleType>(0))::value) particles_.save_to_hdf5_group(g);
    }
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // TODO: do we need these anymore?

    // Console logger
    void log_console(int step, const SaveTask& t) {
        // Build a single-row table: Step | name1(avg) | name2(avg) ... for system-level fields
        // Compute simple averages for D1 System fields. For D2 fields, omit unless later specified.
        struct Col { std::string name; std::string value; };
        std::vector<Col> cols;
        cols.push_back({"step", std::to_string(step)});
        for (const auto& kv : t.one_d) {
            const std::string& name = kv.first;
            std::visit([&](const auto& hb) {
                if (hb.space == IndexSpace::System && !hb.v.empty()) {
                    double s = 0.0; 
                    for (auto x : hb.v) s += static_cast<double>(x); 
                    double avg = s / static_cast<double>(hb.v.size());
                    char buf[64]; std::snprintf(buf, sizeof(buf), "%.6g", avg);
                    cols.push_back({name, std::string(buf)});
                }
            }, kv.second);
        }
        // Format columns with padding and separators
        std::vector<std::size_t> widths; widths.reserve(cols.size());
        for (auto& c : cols) widths.push_back(std::max(c.name.size(), c.value.size()));
        // header
        std::string line; line.reserve(128);
        for (std::size_t i=0;i<cols.size();++i) {
            if (i) line += " | ";
            line += pad(cols[i].name, widths[i]);
        }
        std::puts(line.c_str());
        // values
        line.clear();
        for (std::size_t i=0;i<cols.size();++i) {
            if (i) line += " | ";
            line += pad(cols[i].value, widths[i]);
        }
        std::puts(line.c_str());
    }

    static std::string pad(const std::string& s, std::size_t w) {
        if (s.size() >= w) return s;
        std::string out = s; out.append(w - s.size(), ' '); return out;
    }
};

} // namespace io


