using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using Plots

# Set a seed for reproducibility (optional)
# Random.seed!(1234)

# ==============================================================================
# 1. Matrix Definitions
# ==============================================================================

function matrixbig(m::Int)
    M = [min(i, j) for i in 1:m, j in 1:m]
    return 0.5 * (inv(M) + 0.1 * I(m))
end

function mask(m::Int)
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

function invcovomegam(m::Int)
    dim = 2 * m
    M_big = matrixbig(dim)
    Omega_m = mat_omega(m)
    inner = 2 * M_big * (im * Omega_m)
    num = exp(inner) - I(dim)
    den = 2
    return (num / den) * (im * Omega_m)
end

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
# 3. Estimators and Reconstruction
# ==============================================================================

function sample_data(m::Int, k::Int)
    Cov = covmeas(m)
    CovReal = real.(Cov)
    CovReal = Symmetric(CovReal)
    d = MvNormal(zeros(2*m), CovReal)
    return rand(d, k)' 
end

function covestnew(samp_matrix)
    k, dim = size(samp_matrix)
    return (samp_matrix' * samp_matrix) / k
end

function inv_cov_omega_est(samp_matrix, m::Int)
    cov_est = covestnew(samp_matrix)
    dim = 2 * m
    Omega_m = mat_omega(m)
    return inv(2 * (cov_est - I(dim)/2) - im * Omega_m)
end

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
# 4. Local Reconstruction Logic (Standard)
# ==============================================================================

function local_inv_cov_omega_est(samp_matrix, m::Int, l::Int, x::Int)
    full_cov_est = covestnew(samp_matrix)
    dim = 2 * m
    j_min = max(2*(x-1) - 2*l + 1, 1)
    j_max = min(2*(x-1) + 2*l + 2, dim)
    block = full_cov_est[j_min:j_max, j_min:j_max]
    block_dim = size(block, 1)
    n_modes_local = div(block_dim, 2)
    iOmega_local = im * mat_omega(n_modes_local)
    term = 2 * (block - I(block_dim)/2) - iOmega_local
    return hermitize(inv(term))
end

function reconstruction_inv_cov_omega_est(samp_matrix, l::Int, m::Int)
    dim = 2 * m
    local_matrices = [local_inv_cov_omega_est(samp_matrix, m, l, x) for x in 1:m]
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
# 5. Error Functions (Updated)
# ==============================================================================

# 1. Full Naive (With Logarithm)
function error_naive_est(samp_matrix, m::Int)
    target = matrixbig(2 * m)
    inv_cov = inv_cov_omega_est(samp_matrix, m)
    H_estimated = h_est_jordan(inv_cov, m)
    H_masked = h_est_zeros(H_estimated, m)
    return maximum(abs.(target - H_masked))
end


function error_linear_est(samp_matrix, m::Int)
    target = matrixbig(2 * m)
    
    # Get (2V - iOmega)^-1
    inv_cov = inv_cov_omega_est(samp_matrix, m)
    

    
    H_linear = real.(hermitize(inv_cov)) 
    
    # Apply mask and compare
    H_masked = h_est_zeros(H_linear, m)
    
    return maximum(abs.(target - H_masked))
end

# 3. Local
function error_loc_est(samp_matrix, l::Int, m::Int)
    target = matrixbig(2 * m)
    inv_cov_rec = reconstruction_inv_cov_omega_est(samp_matrix, l, m)
    H_estimated = h_est_jordan(inv_cov_rec, m)
    H_masked = h_est_zeros(H_estimated, m)
    return maximum(abs.(target - H_masked))
end


# ==============================================================================
# 6. Execution
# ==============================================================================

println("Running L-dependence check...")
m_fixed = 100
S_fixed = sample_data(m_fixed, 100000) 
l_values = 2:2:10

errors_local = Float64[]
errors_linear = Float64[] 

# Compute single value for Linear (Global) error as baseline
err_lin_base = error_linear_est(S_fixed, m_fixed)
err_naive_base = error_naive_est(S_fixed, m_fixed)

println("Baseline Global Errors:")
println("  Global (With Log): $err_naive_base")
println("  Classical and Global (No Log):  $err_lin_base")

for l in l_values
    err = error_loc_est(S_fixed, l, m_fixed)
    push!(errors_local, err)
    println("l = $l, Local Error = $err")
end
