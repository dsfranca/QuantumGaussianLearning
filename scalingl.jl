import Pkg; 
Pkg.add(["Plots", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "DelimitedFiles", "Dates", "Printf"])

using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using Plots
using DelimitedFiles
using Dates
using Printf

# ==============================================================================
# 1. Matrix Definitions
# ==============================================================================

function matrixbig(m::Int)
    M = [min(i, j) for i in 1:m, j in 1:m]
    return 0.5 * (inv(M) + 0.1 * I(m))
end

function mask(m::Int)
    return map(x -> abs(x) > 1e-9 ? 1.0 : 0.0, matrixbig(m))
end

# Check if defined to avoid "WARNING: redefinition"
if !isdefined(Main, :Omega_block)
    const Omega_block = [0 1; -1 0]
end
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

function inv_cov_omega_est(cov_matrix, m::Int)
    # Accepts either sampled or exact covariance
    dim = 2 * m
    Omega_m = mat_omega(m)
    return inv(2 * (cov_matrix - I(dim)/2) - im * Omega_m)
end

# --- NAIVE (LOG) ESTIMATOR ---
function h_est(inv_cov_omega, m::Int)
    dim = 2 * m
    Omega_m = mat_omega(m)
    iOmega = im * Omega_m
    term = I(dim) + 2 * inv_cov_omega * iOmega
    return 0.5 * log(term) * iOmega
end

# --- CLASSICAL (LINEAR) ESTIMATOR ---
function h_est_classical(inv_cov_omega, m::Int)
    return inv_cov_omega
end

function h_est_jordan(H_complex, m::Int)
    return real.(hermitize(H_complex))
end

function h_est_zeros(H_est, m::Int)
    dim = 2 * m
    M_mask = mask(dim) 
    return M_mask .* H_est
end

# ==============================================================================
# 4. Local Reconstruction Logic
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
# 5. Error Functions
# ==============================================================================

function error_naive_est(samp_matrix, m::Int)
    target = matrixbig(2 * m)
    cov_est = covestnew(samp_matrix)
    inv_cov = inv_cov_omega_est(cov_est, m)
    H_estimated = h_est(inv_cov, m) # Uses Log
    H_jordan = h_est_jordan(H_estimated, m)
    H_masked = h_est_zeros(H_jordan, m)
    return maximum(abs.(target - H_masked))
end

function error_loc_est(samp_matrix, l::Int, m::Int)
    target = matrixbig(2 * m)
    inv_cov_rec = reconstruction_inv_cov_omega_est(samp_matrix, l, m)
    H_estimated = h_est(inv_cov_rec, m) # Uses Log
    H_jordan = h_est_jordan(H_estimated, m)
    H_masked = h_est_zeros(H_jordan, m)
    return maximum(abs.(target - H_masked))
end

#Classical Exact Error Calculation ---
function error_classical_exact(m::Int)
    target = matrixbig(2 * m)
    
    # 1. Get Exact Covariance (No sampling)
    Cov_exact = covmeas(m) 
    
    # 2. Get Inverse Proxy
    inv_cov = inv_cov_omega_est(Cov_exact, m)
    
    # 3. Use Classical (Linear) Estimator
    H_class = h_est_classical(inv_cov, m) 
    
    # 4. Standard post-processing
    H_jordan = h_est_jordan(H_class, m)
    H_masked = h_est_zeros(H_jordan, m)
    
    return maximum(abs.(target - H_masked))
end

function format_sec(s)
    return Time(0) + Second(round(Int, s))
end

# ==============================================================================
# 6. Main Execution
# ==============================================================================

function run_simulation()
    println("Generating Data and Calculating Errors...")

    samples_count = 10000
    l_param = 3
    num_averaging_runs = 3
    filename = "simulation_results_classical.csv"

    # Range 
    m_values = 100:1:100

    # Initialize file
    open(filename, "w") do io
        writedlm(io, [["SystemSize_m" "Avg_Naive_Error" "Avg_Local_Error" "Classical_Exact_Error" "Avg_Naive_Time_sec" "Avg_Local_Time_sec"]], ',')
    end
    println("Initialized $filename.")
    println("---------------------------------------------------------------------------------------------------")
    println(" m     | Naive Err | Local Err | Class(Exact) | T_Naive(s) | T_Local(s) | Total Time | ETA ")
    println("---------------------------------------------------------------------------------------------------")

    # Storage for plotting
    m_values_mem = Int[]
    errors_local_mem = Float64[]
    errors_naive_mem = Float64[]
    errors_class_mem = Float64[] # Store classical exact errors

    t_start = time()
    total_work_units = sum(m^3 for m in m_values)
    completed_work_units = 0.0

    for m in m_values
        
        # --- 1. Classical Exact Calculation (Baseline) ---
        # This is fast, so we compute it once per m
        class_exact_err = error_classical_exact(m)

        # --- 2. Local Reconstruction (Sampled) ---
        temp_errors_local = Float64[]
        t_local_total = @elapsed begin
            for k in 1:num_averaging_runs
                S = sample_data(m, samples_count)
                push!(temp_errors_local, error_loc_est(S, l_param, m))
            end
        end
        avg_local_err = mean(temp_errors_local)
        avg_local_time = t_local_total / num_averaging_runs
        
        # --- 3. Naive Estimation (Sampled) ---
        temp_errors_naive = Float64[]
        t_naive_total = @elapsed begin
            for k in 1:num_averaging_runs
                S = sample_data(m, samples_count)
                push!(temp_errors_naive, error_naive_est(S, m))
            end
        end
        avg_naive_err = mean(temp_errors_naive)
        avg_naive_time = t_naive_total / num_averaging_runs
        
        # --- 4. Export Data ---
        open(filename, "a") do io
            writedlm(io, [m avg_naive_err avg_local_err class_exact_err avg_naive_time avg_local_time], ',')
        end
        
        # --- 5. Timing ---
        t_now = time()
        total_elapsed = t_now - t_start
        completed_work_units += m^3
        fraction_done = completed_work_units / total_work_units
        eta_str = fraction_done > 0 ? string(format_sec((total_elapsed/fraction_done) - total_elapsed)) : "Calc..."
        
        # Store
        push!(m_values_mem, m)
        push!(errors_local_mem, avg_local_err)
        push!(errors_naive_mem, avg_naive_err)
        push!(errors_class_mem, class_exact_err)
        
        Printf.@printf(" %-5d | %.2e  | %.2e  | %.2e     | %-10.2f | %-10.2f | %-10s | %s\n", 
            m, avg_naive_err, avg_local_err, class_exact_err, avg_naive_time, avg_local_time, format_sec(total_elapsed), eta_str)
    end

    println("\nSimulation finished. Data saved to $filename")
    
    return m_values_mem, errors_naive_mem, errors_local_mem, errors_class_mem
end

# Run the function
m_vals, err_naive, err_local, err_class_exact = run_simulation()

