#pragma once
// include/utils/h5_io.hpp

#include <hdf5.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstdint>
#include <type_traits>
#include <climits>  // for INT_MAX

// ===============================
// HDF5 helpers
// ===============================
inline std::vector<hsize_t> get_dims(hid_t dset) {
    if (dset < 0) throw std::runtime_error("get_dims: invalid dataset handle");
    hid_t space = H5Dget_space(dset);
    if (space < 0) throw std::runtime_error("get_dims: H5Dget_space failed");
    int nd = H5Sget_simple_extent_ndims(space);
    if (nd < 0) { H5Sclose(space); throw std::runtime_error("get_dims: H5Sget_simple_extent_ndims failed"); }
    std::vector<hsize_t> dims(static_cast<size_t>(nd));
    if (nd > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    return dims;
}

inline bool h5_link_exists(hid_t loc, const std::string& path) {
    htri_t ex = H5Lexists(loc, path.c_str(), H5P_DEFAULT);
    return ex > 0;
}

// Map C++ types to HDF5 native types
template<class T> inline hid_t h5_native(); // specializations
template<> inline hid_t h5_native<float>()                { return H5T_NATIVE_FLOAT; }
template<> inline hid_t h5_native<double>()               { return H5T_NATIVE_DOUBLE; }
template<> inline hid_t h5_native<int>()                  { return H5T_NATIVE_INT; }
template<> inline hid_t h5_native<unsigned int>()         { return H5T_NATIVE_UINT; }
template<> inline hid_t h5_native<long>()                 { return H5T_NATIVE_LONG; }
template<> inline hid_t h5_native<unsigned long>()        { return H5T_NATIVE_ULONG; }
template<> inline hid_t h5_native<long long>()            { return H5T_NATIVE_LLONG; }
template<> inline hid_t h5_native<unsigned long long>()   { return H5T_NATIVE_ULLONG; }
template<> inline hid_t h5_native<bool>()                 { return H5T_NATIVE_HBOOL; }
template<> inline hid_t h5_native<char>()                 { return H5T_NATIVE_CHAR; }
template<> inline hid_t h5_native<unsigned char>()        { return H5T_NATIVE_UCHAR; }
template<> inline hid_t h5_native<short>()                { return H5T_NATIVE_SHORT; }
template<> inline hid_t h5_native<unsigned short>()       { return H5T_NATIVE_USHORT; }

template <class T>
inline T read_scalar(hid_t loc, const std::string& path){
    hid_t dset = H5Dopen2(loc, path.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("read_scalar: missing dataset: " + path);
    hid_t space = H5Dget_space(dset);
    if (space < 0) { H5Dclose(dset); throw std::runtime_error("read_scalar: H5Dget_space failed: " + path); }
    if (H5Sget_simple_extent_ndims(space) != 0) {
        H5Sclose(space); H5Dclose(dset);
        throw std::runtime_error("read_scalar: expected scalar dataspace at " + path);
    }
    H5Sclose(space);
    T val{};
    if (H5Dread(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, &val) < 0){
        H5Dclose(dset);
        throw std::runtime_error("read_scalar: H5Dread failed for " + path);
    }
    H5Dclose(dset);
    return val;
}

template <class T>
inline void write_scalar(hid_t loc, const std::string& path, const T& value){
    if (h5_link_exists(loc, path)) {
        if (H5Ldelete(loc, path.c_str(), H5P_DEFAULT) < 0)
            throw std::runtime_error("write_scalar: failed to delete existing link: " + path);
    }
    hid_t space = H5Screate(H5S_SCALAR);
    if (space < 0) throw std::runtime_error("write_scalar: H5Screate failed");
    hid_t dset  = H5Dcreate2(loc, path.c_str(), h5_native<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); throw std::runtime_error("write_scalar: H5Dcreate2 failed for " + path); }
    if (H5Dwrite(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, &value) < 0){
        H5Dclose(dset); H5Sclose(space);
        throw std::runtime_error("write_scalar: H5Dwrite failed for " + path);
    }
    H5Dclose(dset); H5Sclose(space);
}

template<class T>
inline std::vector<T> read_vector(hid_t loc, const std::string& dset_path) {
    hid_t dset = H5Dopen2(loc, dset_path.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("read_vector: missing dataset: " + dset_path);
    auto dims = get_dims(dset);
    if (!(dims.size()==1)) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector: expected (N,) at " + dset_path);
    }
    if (dims[0] > static_cast<hsize_t>(INT_MAX)) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector: dimension exceeds INT_MAX at " + dset_path);
    }
    int N = static_cast<int>(dims[0]);
    std::vector<T> h(static_cast<size_t>(N));
    if (H5Dread(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, h.data()) < 0) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector: H5Dread failed for " + dset_path);
    }
    H5Dclose(dset);
    return h;
}

// HDF5: write (N,) exactly (overwrites if exists)
template<class T>
inline void write_vector(hid_t loc, const std::string& dset_path, const std::vector<T>& data) {
    if (h5_link_exists(loc, dset_path)) {
        if (H5Ldelete(loc, dset_path.c_str(), H5P_DEFAULT) < 0)
            throw std::runtime_error("write_vector: failed to delete existing link: " + dset_path);
    }
    hsize_t dims[1] = { static_cast<hsize_t>(data.size()) };
    hid_t space = H5Screate_simple(1, dims, nullptr);
    if (space < 0) throw std::runtime_error("write_vector: H5Screate_simple failed");
    hid_t dset  = H5Dcreate2(loc, dset_path.c_str(), h5_native<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); throw std::runtime_error("write_vector: H5Dcreate2 failed for " + dset_path); }
    if (H5Dwrite(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, data.data()) < 0) {
        H5Dclose(dset); H5Sclose(space);
        throw std::runtime_error("write_vector: H5Dwrite failed for " + dset_path);
    }
    H5Dclose(dset); H5Sclose(space);
}

// HDF5: require (N,2) exactly
template<class T>
inline std::pair<std::vector<T>, std::vector<T>> read_vector_2d(hid_t loc, const std::string& dset_path) {
    hid_t dset = H5Dopen2(loc, dset_path.c_str(), H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error("read_vector_2d: missing dataset: " + dset_path);
    auto dims = get_dims(dset);
    if (!(dims.size()==2 && dims[1]==2)) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector_2d: expected (N,2) at " + dset_path);
    }
    if (dims[0] > static_cast<hsize_t>(INT_MAX)) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector_2d: dimension exceeds INT_MAX at " + dset_path);
    }
    int N = static_cast<int>(dims[0]);
    std::vector<T> h(static_cast<size_t>(2*N));
    if (H5Dread(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, h.data()) < 0) {
        H5Dclose(dset);
        throw std::runtime_error("read_vector_2d: H5Dread failed for " + dset_path);
    }
    H5Dclose(dset);
    std::vector<T> hx(static_cast<size_t>(N)), hy(static_cast<size_t>(N));
    for (int i=0;i<N;++i) { hx[i]=h[2*i+0]; hy[i]=h[2*i+1]; }
    return {std::move(hx), std::move(hy)};
}

// HDF5: write (N,2) exactly (overwrites if exists)
template<class T>
inline void write_vector_2d(hid_t loc, const std::string& dset_path, const std::vector<T>& x, const std::vector<T>& y) {
    if (h5_link_exists(loc, dset_path)) {
        if (H5Ldelete(loc, dset_path.c_str(), H5P_DEFAULT) < 0)
            throw std::runtime_error("write_vector_2d: failed to delete existing link: " + dset_path);
    }
    if (x.size() != y.size())
        throw std::runtime_error("write_vector_2d: x and y size mismatch");
    int N = static_cast<int>(x.size());
    hsize_t dims[2] = { static_cast<hsize_t>(N), 2 };
    hid_t space = H5Screate_simple(2, dims, nullptr);
    if (space < 0) throw std::runtime_error("write_vector_2d: H5Screate_simple failed");
    hid_t dset  = H5Dcreate2(loc, dset_path.c_str(), h5_native<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) { H5Sclose(space); throw std::runtime_error("write_vector_2d: H5Dcreate2 failed for " + dset_path); }
    std::vector<T> h(static_cast<size_t>(2*N));
    for (int i=0;i<N;++i) { h[2*i+0]=x[i]; h[2*i+1]=y[i]; }
    if (H5Dwrite(dset, h5_native<T>(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT, h.data()) < 0) {
        H5Dclose(dset); H5Sclose(space);
        throw std::runtime_error("write_vector_2d: H5Dwrite failed for " + dset_path);
    }
    H5Dclose(dset); H5Sclose(space);
}