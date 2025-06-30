#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); }

#define CUSOLVER_CHECK(err) if (err != CUSOLVER_STATUS_SUCCESS) { \
    std::cerr << "cuSolver error: " << err << std::endl; exit(1); }

using BitState = uint64_t;

using namespace std;

// Generate all Fock states with exactly N particles on L sites (CPU)
void generate_fock_basis(int L, int N, vector<BitState>& basis) {
    BitState max_state = 1ULL << L;
    for (BitState s = 0; s < max_state; ++s) {
        if (__builtin_popcountll(s) == N) {
            basis.push_back(s);
        }
    }
    sort(basis.begin(), basis.end()); // important for binary search on GPU
}

// Device function to calculate fermionic sign
__device__ int fermionic_sign_device(BitState state, int i, int j) {
    if (i > j) { int temp = i; i = j; j = temp; }
    int count = 0;
    for (int k = i + 1; k < j; ++k) {
        if ((state >> k) & 1) count++;
    }
    return (count % 2 == 0) ? +1 : -1;
}

// Device binary search to find index of target state in basis array
__device__ int binary_search(const BitState* basis, int size, BitState target) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        BitState mid_val = basis[mid];
        if (mid_val == target) return mid;
        else if (mid_val < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1; // not found
}

// Custom atomicAdd for double 
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// CUDA kernel to build Hamiltonian matrix in parallel
__global__ void build_hamiltonian_kernel(
    int L, double t,
    const BitState* basis, int dim,
    double* H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;

    BitState s = basis[idx];

    for (int j = 0; j < L; ++j) {
        int jp = (j + 1) % L; // periodic boundary

        if (((s >> j) & 1) && !((s >> jp) & 1)) {
            BitState new_s = s ^ (1ULL << j) ^ (1ULL << jp);
            int new_idx = binary_search(basis, dim, new_s);
            if (new_idx != -1) {
                int sign = fermionic_sign_device(s, j, jp);
                double val = -t * sign;
                atomicAdd_double(&H[idx * dim + new_idx], val);
                atomicAdd_double(&H[new_idx * dim + idx], val);
            }
        }
    }
}

// Diagonalize symmetric matrix H using cuSolver (GPU)
void diagonalize_gpu(int dim, double* d_H, vector<double>& eigvals, vector<double>& eigvecs) {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(
        handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        dim, d_H, dim, nullptr, &work_size));

    double* d_W = nullptr;
    CUDA_CHECK(cudaMalloc(&d_W, dim * sizeof(double)));

    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(double)));

    int* dev_info = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_info, sizeof(int)));

    CUSOLVER_CHECK(cusolverDnDsyevd(
        handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        dim, d_H, dim, d_W, d_work, work_size, dev_info));

    int info_h = 0;
    CUDA_CHECK(cudaMemcpy(&info_h, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_h != 0) {
        std::cerr << "Eigen decomposition failed, info = " << info_h << std::endl;
        exit(1);
    }

    eigvals.resize(dim);
    eigvecs.resize(dim * dim);
    CUDA_CHECK(cudaMemcpy(eigvals.data(), d_W, dim * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eigvecs.data(), d_H, dim * dim * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(dev_info));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}

int main() {
    const int L = 6;            // Number of lattice sites
    const int N = L / 2;         // Half-filling
    const double t = 1.0;        // Hopping amplitude

    // 1. Generate basis
    vector<BitState> basis;
    generate_fock_basis(L, N, basis);
    int dim = basis.size();

    cout << "Half-filled system: L = " << L << ", N = " << N << ", Hilbert space dim = " << dim << endl;

    // 2. Allocate device memory
    BitState* d_basis = nullptr;
    CUDA_CHECK(cudaMalloc(&d_basis, dim * sizeof(BitState)));
    CUDA_CHECK(cudaMemcpy(d_basis, basis.data(), dim * sizeof(BitState), cudaMemcpyHostToDevice));

    double* d_H = nullptr;
    CUDA_CHECK(cudaMalloc(&d_H, dim * dim * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_H, 0, dim * dim * sizeof(double)));

    // 3. Launch kernel
    int threads_per_block = 256;
    int blocks = (dim + threads_per_block - 1) / threads_per_block;

    auto start_build = chrono::high_resolution_clock::now();

    build_hamiltonian_kernel<<<blocks, threads_per_block>>>(L, t, d_basis, dim, d_H);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_build = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_build = end_build - start_build;
    cout << "GPU Hamiltonian construction completed in " << elapsed_build.count() << " seconds." << endl;

    // 4. Diagonalize
    vector<double> eigvals, eigvecs;

    auto start_diag = chrono::high_resolution_clock::now();
    diagonalize_gpu(dim, d_H, eigvals, eigvecs);
    auto end_diag = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_diag = end_diag - start_diag;

    cout << "GPU diagonalization completed in " << elapsed_diag.count() << " seconds." << endl;

    // 5. Save eigenvalues
    ofstream fout("hamtest_eigenvalues.txt");
    for (double val : eigvals) fout << val << "\n";
    fout.close();
    cout << "Eigenvalues saved to '0test_ham_eigenvalues.txt'." << endl;

    // 6. Save eigenvectors
    ofstream vecout("hamtest_eigenvectors.txt");
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            vecout << eigvecs[col * dim + row];  // column-major
            if (row != dim - 1) vecout << ", ";
        }
        vecout << "\n";
    }
    vecout.close();
    cout << "Eigenvectors saved to '0test_ham_eigenvectors.txt'." << endl;

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_basis));
    CUDA_CHECK(cudaFree(d_H));

    return 0;
}
