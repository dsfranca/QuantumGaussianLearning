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

# Creates the base matrix based on the inverse of Min(i,j)
function matrixbig(m::Int)
    # Create the matrix M_ij = Min(i, j)
    M = [min(i, j) for i in 1:m, j in 1:m]
    
    # Calculate 0.5 * (Inverse(M) + 0.1 * Identity)
    # Note: We force it to be a float matrix
    return 0.5 * (inv(M) + 0.001 * I(m))
end

# Create a binary mask (Unitize in Mathematica)
function mask(m::Int)
    M = matrixbig(div(m, 2)) # The notebook calls mask(2m) based on matrixbig(2m) usually?
    # Actually, looking at the notebook logic, input is 'm', 
    # but typically used as mask(2m) where the underlying structure comes from matrixbig(2m).
    # Let's strictly follow the notebook definition: mask(m) := Unitize[matrixbig(m)]
    return map(x -> abs(x) > 1e-10 ? 1.0 : 0.0, matrixbig(m))
end

# The Omega block
const Omega_block = [0 1; -1 0]

# Helper to make Hermitian: (A + A')/2
hermitize(A) = (A + A') / 2

# Creates the block diagonal symplectic form Omega for size n (m in notebook)
# In Mathematica: mat[p, n] -> Diagonal with p blocks
function mat_omega(n::Int)
    # This creates a 2n x 2n matrix with n blocks of Omega on the diagonal
    return kron(I(n), Omega_block)
end

# ==============================================================================
# 2. Covariance and Hamiltonian Formulas
# ==============================================================================

# Formula for (2V - i*Omega)^-1
# Note: We use 'im' for the imaginary unit in Julia
function invcovomegam(m::Int)
    dim = 2 * m
    M_big = matrixbig(dim)
    Omega_m = mat_omega(m)
    
    # Term inside exponential: 2 * M_big . (i * Omega_m)
    inner = 2 * M_big * (im * Omega_m)
    
    # MatrixExp and subtraction
    num = exp(inner) - I(dim)
    den = 2
    
    # Result . (i * Omega_m)
    return (num / den) * (im * Omega_m)
end

# Formula for covariance matrix of Gaussian output (Covmeas)
function covmeas(m::Int)
    dim = 2 * m
    M_big = matrixbig(dim)
    Omega_m = mat_omega(m)
    iOmega = im * Omega_m
    
    # Calculate the complex term inside
    exponent_term = 2 * M_big * iOmega
    term_inv = inv((exp(exponent_term) - I(dim)) / 2)
    
    # Full formula: ((iOmega).Inverse[...] + iOmega)/2 + I/2
    res = (iOmega * term_inv + iOmega) / 2 + I(dim) / 2
    return res
end

# ==============================================================================
# 3. Estimators and Reconstruction
# ==============================================================================

# Sample from Multivariate distribution
function sample_data(m::Int, k::Int)
    # Get theoretical covariance
    Cov = covmeas(m)
    
    # In the notebook, they take Re[Covmeas]. 
    # The Covmeas should theoretically be real for physical states, 
    # but due to 'im' math, Julia might track it as Complex type.
    CovReal = real.(Cov)
    
    # Ensure symmetry for numerical stability
    CovReal = Symmetric(CovReal)
    
    d = MvNormal(zeros(2*m), CovReal)
    
    # Generate k samples. Julia returns a (2m x k) matrix. 
    # The Mathematica code treats samples as row vectors usually in Covariance estimators,
    # but let's look at covestnew: Transpose[samp].samp.
    # If samp is k x 2m (rows are samples), Transpose is 2m x k. Result 2m x 2m.
    # Julia's rand(d, k) produces columns as samples. 
    # So we transpose it to match Mathematica's "List of vectors" format if needed,
    # or just adapt the covariance calculation.
    return rand(d, k)' # Returns k x 2m matrix
end

# Covariance estimator
# In Mathematica: (1/Length) * Transpose[samp] . samp
# Julia equivalent for k x 2m input:
function covestnew(samp_matrix)
    k, dim = size(samp_matrix)
    return (samp_matrix' * samp_matrix) / k
end

# Naive estimate of (2V - i*Omega)^-1
function inv_cov_omega_est(samp_matrix, m::Int)
    cov_est = covestnew(samp_matrix)
    dim = 2 * m
    Omega_m = mat_omega(m)
    
    # Formula: Inverse[ 2(cov - I/2) - i*Omega ]
    return inv(2 * (cov_est - I(dim)/2) - im * Omega_m)
end

# Estimate of H using MatrixLog
function h_est(inv_cov_omega, m::Int)
    dim = 2 * m
    Omega_m = mat_omega(m)
    iOmega = im * Omega_m
    
    # 1/2 * Log[ I + 2 * inv_cov . iOmega ] . iOmega
    term = I(dim) + 2 * inv_cov_omega * iOmega
    return 0.5 * log(term) * iOmega
end

# Force Hermitian and Real (Jordan decomposition wrapper in NB, standard Log here)
function h_est_jordan(inv_cov_omega, m::Int)
    # Julia's standard matrix log is robust. 
    # We replicate the "Re[Hermitize[...]]" step.
    H = h_est(inv_cov_omega, m)
    return real.(hermitize(H))
end

# Hadamard product with adjacency mask (enforce sparsity)
function h_est_zeros(H_est, m::Int)
    dim = 2 * m
    M_mask = mask(dim) # mask creates Unitize[matrixbig[2m]]
    return M_mask .* H_est
end

# ==============================================================================
# 4. Local Reconstruction Logic
# ==============================================================================

# Exact local inversion
function local_inv_cov_omega(m::Int, l::Int, x::Int)
    # Get full theoretical covariance
    full_cov = covmeas(m) 
    dim = 2 * m
    
    # Define indices range logic from Mathematica notebook
    # Mathematica indices are 1-based (same as Julia)
    # Max[2(x-1) - 2l + 1, 1]
    j_min = max(2*(x-1) - 2*l + 1, 1)
    j_max = min(2*(x-1) + 2*l + 2, dim)
    
    # Extract block
    block = full_cov[j_min:j_max, j_min:j_max]
    
    # Matrix dimensions of the block
    block_dim = size(block, 1)
    
    # The term to subtract: i * Omega_local
    # Need to construct correct Omega block of specific size
    # The notebook generates `mat[p, ...]` with a dynamic size calculation
    # Size calc: (1 + Min[...] - Max[...]) / 2. This calculates number of modes.
    n_modes_local = div(block_dim, 2)
    iOmega_local = im * mat_omega(n_modes_local)
    
    # Formula: Hermitize[ Inverse[ 2(block - I/2) - iOmega ] ]
    term = 2 * (block - I(block_dim)/2) - iOmega_local
    return hermitize(inv(term))
end

# Estimated local inversion (using sample data)
function local_inv_cov_omega_est(samp_matrix, m::Int, l::Int, x::Int)
    # Calculate global covariance estimate first
    # (In efficient code we wouldn't recalc this every time, but this follows the NB logic)
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

# Reconstruction function
function reconstruction_inv_cov_omega_est(samp_matrix, l::Int, m::Int)
    dim = 2 * m
    
    # Pre-calculate local inverted matrices for all x
    local_matrices = [local_inv_cov_omega_est(samp_matrix, m, l, x) for x in 1:m]
    
    # We will build the result matrix. 
    # Initialize with zeros (dense or sparse)
    reconstr = spzeros(ComplexF64, dim, dim)
    
    # Loop over 2x2 blocks (j, k correspond to mode indices 1..m)
    for j in 1:m
        for k in 1:m
            if k >= j && k <= min(j + l, m)
                # Logic to extract 2x2 block from the specific LocalMatrix
                # LocalMatrix index: j
                
                # Calculate offsets within the local matrix
                # Mathematica: 2(j-1) - Max[2(j-1)-2l, 0] + 1
                # This calculates the relative start position of mode j in the window centered/based at j?
                # Actually, the notebook loops x from 1 to m. 
                # But here the loop is reconstruction.
                # In NB: If[k >= j && ..., Localmatrices[[j, range...]]]
                # So it uses the local matrix centered at 'j'.
                
                loc_mat = local_matrices[j]
                
                row_start_global = 2*(j-1)
                col_start_global = 2*(k-1)
                
                # Range in local matrix
                # Offset calculation based on notebook logic:
                # start_local = 2*(j-1) - max(2*(j-1)-2*l, 0) + 1
                r_start = 2*(j-1) - max(2*(j-1)-2*l, 0) + 1
                r_end   = 2*(j-1) - max(2*(j-1)-2*l, 0) + 2
                
                c_start = 2*(k-1) - max(2*(j-1)-2*l, 0) + 1
                c_end   = 2*(k-1) - max(2*(j-1)-2*l, 0) + 2
                
                block_2x2 = loc_mat[r_start:r_end, c_start:c_end]
                
                # Assign to reconstruction
                # Julia sparse assignment
                reconstr[2j-1:2j, 2k-1:2k] = block_2x2
                
                # Hermitian symmetry for the lower triangle
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

# Error for Naive Estimator
function error_naive_est(samp_matrix, m::Int)
    target = matrixbig(2 * m)
    
    # Estimate Inverse Covariance
    inv_cov = inv_cov_omega_est(samp_matrix, m)
    
    # Estimate H
    H_estimated = h_est_jordan(inv_cov, m)
    
    # Mask
    H_masked = h_est_zeros(H_estimated, m)
    
    # Max Abs difference
    return maximum(abs.(target - H_masked))
end

# Error for Local Reconstruction Estimator
function error_loc_est(samp_matrix, l::Int, m::Int)
    target = matrixbig(2 * m)
    
    # Reconstruct Inverse Covariance
    inv_cov_rec = reconstruction_inv_cov_omega_est(samp_matrix, l, m)
    
    # Estimate H
    H_estimated = h_est_jordan(inv_cov_rec, m)
    
    # Mask
    H_masked = h_est_zeros(H_estimated, m)
    
    return maximum(abs.(target - H_masked))
end


# ==============================================================================
# 6. Main Execution & Plotting
# ==============================================================================

println("Generating Data and Calculating Errors...")



# Also calculate the specific list plot from the end of the notebook
# which plotted specific list1 vs listtheo.
# Since we don't have the exact 'data' variable from the notebook's history which ran 
# for m=100, samples=10000, averaged over 10 trials for k=2,4,6,8...
# We will reproduce a single run of that specific plot logic: Error vs Locality Parameter 'l'

println("\nRunning Error vs Locality Parameter l (m=100)...")
m_fixed = 100
S_fixed = sample_data(m_fixed, 10000) # Use 10k samples for speed
l_values = 2:2:10

errors_vs_l = Float64[]
for l in l_values
    err = error_loc_est(S_fixed, l, m_fixed)
    push!(errors_vs_l, err)
    println("l = $l, Error = $err")
end


println("Done.")