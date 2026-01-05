using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using Plots

# ==============================================================================
# 1. Matrix Definitions
# ==============================================================================

function matrixbig(m::Int)
    M = [min(i, j) for i in 1:m, j in 1:m]
    # Standard Well-Conditioned Case
    return 0.5 * (inv(M) + 0.1 * I(m))
end

function mask(m::Int)
    # Use tolerance to identify non-zero elements
    return map(x -> abs(x) > 1e-10 ? 1.0 : 0.0, matrixbig(m))
end

const Omega_block = [0 1; -1 0]
hermitize(A) = (A + A') / 2

function mat_omega(n::Int)
    return kron(I(n), Omega_block)
end

# ==============================================================================
# 2. Covariance and Hamiltonian Formulas
# ==============================================================================

function covmeas(m::Int)
    dim = 2 * m
    M_big = matrixbig(dim)
    Omega_m = mat_omega(m)
    iOmega = im * Omega_m
    exponent_term = 2 * M_big * iOmega
    term_inv = inv((exp(exponent_term) - I(dim)) / 2)
    res = (iOmega * term_inv + iOmega) / 2 + I(dim) / 2
    return res
end

# ==============================================================================
# 3. Estimators (Exact - No Sampling)
# ==============================================================================

# Helper to estimate H from the intermediate matrix (2V - iOmega)^-1
function h_est(inv_cov_omega, m::Int)
    dim = 2 * m
    Omega_m = mat_omega(m)
    iOmega = im * Omega_m
    term = I(dim) + 2 * inv_cov_omega * iOmega
    return 0.5 * log(term) * iOmega
end

function h_est_jordan(inv_cov_omega, m::Int)
    H = h_est(inv_cov_omega, m)
    return real.(hermitize(H))
end

function h_est_zeros(H_est, m::Int)
    dim = 2 * m
    M_mask = mask(dim) 
    return M_mask .* H_est
end

# ==============================================================================
# 4. Exact Reconstruction Logic
# ==============================================================================

# Exact Local Inversion: Takes the full Cov matrix (Exact), not samples
function local_inv_cov_omega_exact(full_cov, m::Int, l::Int, x::Int)
    dim = 2 * m
    
    j_min = max(2*(x-1) - 2*l + 1, 1)
    j_max = min(2*(x-1) + 2*l + 2, dim)
    
    # Extract block directly from exact covariance
    block = full_cov[j_min:j_max, j_min:j_max]
    block_dim = size(block, 1)
    
    n_modes_local = div(block_dim, 2)
    iOmega_local = im * mat_omega(n_modes_local)
    
    term = 2 * (block - I(block_dim)/2) - iOmega_local
    return hermitize(inv(term))
end

# Exact Reconstruction Loop
function reconstruction_exact(full_cov, l::Int, m::Int)
    dim = 2 * m
    local_matrices = [local_inv_cov_omega_exact(full_cov, m, l, x) for x in 1:m]
    reconstr = spzeros(ComplexF64, dim, dim)
    
    for j in 1:m
        for k in 1:m
            if k >= j && k <= min(j + l, m)
                loc_mat = local_matrices[j]
                
                r_start = 2*(j-1) - max(2*(j-1)-2*l, 0) + 1
                r_end   = 2*(j-1) - max(2*(j-1)-2*l, 0) + 2
                c_start = 2*(k-1) - max(2*(j-1)-2*l, 0) + 1
                c_end   = 2*(k-1) - max(2*(j-1)-2*l, 0) + 2
                
                block_2x2 = loc_mat[r_start:r_end, c_start:c_end]
                reconstr[2j-1:2j, 2k-1:2k] = block_2x2
                
                if j != k
                    reconstr[2k-1:2k, 2j-1:2j] = block_2x2'
                end
            end
        end
    end
    return hermitize(reconstr)
end

# ==============================================================================
# 5. Exact Error Functions
# ==============================================================================

# 1. Exact Global Naive (With Log)
function error_naive_exact(full_cov, m::Int)
    target = matrixbig(2 * m)
    dim = 2 * m
    Omega_m = mat_omega(m)
    
    # Direct calculation on full covariance
    inv_cov_exact = inv(2 * (full_cov - I(dim)/2) - im * Omega_m)
    
    H_estimated = h_est_jordan(inv_cov_exact, m)
    H_masked = h_est_zeros(H_estimated, m)
    return maximum(abs.(target - H_masked))
end

# 2. Exact Linear (No Log)
function error_linear_exact(full_cov, m::Int)
    target = matrixbig(2 * m)
    dim = 2 * m
    Omega_m = mat_omega(m)
    
    # Direct calculation on full covariance
    inv_cov_exact = inv(2 * (full_cov - I(dim)/2) - im * Omega_m)
    
    # Linear approximation: H ~ Re(InverseCov)
    H_linear = real.(hermitize(inv_cov_exact))
    
    H_masked = h_est_zeros(H_linear, m)
    return maximum(abs.(target - H_masked))
end

# 3. Exact Local Reconstruction
function error_loc_exact(full_cov, l::Int, m::Int)
    target = matrixbig(2 * m)
    
    inv_cov_rec = reconstruction_exact(full_cov, l, m)
    H_estimated = h_est_jordan(inv_cov_rec, m)
    
    H_masked = h_est_zeros(H_estimated, m)
    return maximum(abs.(target - H_masked))
end

# ==============================================================================
# 6. Execution
# ==============================================================================

println("Running Exact L-dependence check (Theoretical Limit)...")
m_fixed = 100

# Step 1: Compute EXACT Theoretical Covariance
# (We take real part to simulate physical measurement, imaginary parts are ~1e-18 noise)
Cov_Exact = real.(covmeas(m_fixed))

# Range of l
l_values = 2:2:12

errors_local = Float64[]

# Compute Baselines using Exact Covariance
# Note: Naive error should be effectively 0 (machine precision) if we use exact covariance 
# because global inversion of exact covariance recovers H perfectly.
# The "Linear" error will be non-zero because the model is wrong (missing Log).
err_naive_base = error_naive_exact(Cov_Exact, m_fixed)
err_lin_base = error_linear_exact(Cov_Exact, m_fixed)

println("Baseline Global Errors (Exact):")
println("  Global (With Log, should be ~0): $err_naive_base")
println("  Classical / Linear (No Log):     $err_lin_base")

println("\nLocal Reconstruction Convergence:")
for l in l_values
    err = error_loc_exact(Cov_Exact, l, m_fixed)
    push!(errors_local, err)
    println("l = $l, Local Error = $err")
end


println("Done.")