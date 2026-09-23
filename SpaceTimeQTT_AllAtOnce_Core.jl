# =============================================================================
# SpaceTimeQTT_AllAtOnce_Core.jl
# Shared helpers for 1D/2D QTT/MPS space-time all-at-once benchmarks.
#
# Supported benchmark families:
#   1) SHO imaginary-time ground-state search (Backward Euler)
#   2) GPE imaginary-time ground-state search (Backward Euler + Picard)
#   3) SHO real-time evolution (Crank-Nicolson)
#
# Spatial convention:
#   - periodic grid on [-L, L) to match the GPE paper's periodic BC convention
#   - sequential QTT ordering [time][x] in 1D and [time][x][y] in 2D
#   - internal state uses the discrete-normalized scaled wavefunction psi_tilde:
#         sum |psi_tilde|^2 = 1
#     For d dimensions, psi_tilde = sqrt(DeltaV) * psi_continuum, so the
#     nonlinear coupling in the QTT/discrete representation is g/DeltaV.
#
# =============================================================================

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Serialization
using Pkg

ENV["GKSwstype"] = "100"  # headless plotting on remote Linux

# -----------------------------------------------------------------------------
# Project / package setup
# -----------------------------------------------------------------------------

const PROJECT_ROOT = @__DIR__
Pkg.activate(PROJECT_ROOT; io=devnull)

using MPSCore
using QTTCore
using Plots

# -----------------------------------------------------------------------------
# Generic MPS / MPO helpers
# -----------------------------------------------------------------------------

copy_mps(psi::MPS) = MPS([copy(psi[k]) for k in 1:length(psi)])
copy_mpo(W::MPO) = MPO([copy(W[k]) for k in 1:length(W)])

function scale_mps(psi::MPS, c::Number)
    T = promote_type(eltype(psi), typeof(c))
    tensors = [T.(psi[k]) for k in 1:length(psi)]
    tensors[1] .*= T(c)
    return MPS(tensors)
end

function scale_mpo(W::MPO, c::Number)
    T = promote_type(eltype(W), typeof(c))
    tensors = [T.(W[k]) for k in 1:length(W)]
    tensors[1] .*= T(c)
    return MPO(tensors)
end

function add_mps(a::MPS, b::MPS)
    T = promote_type(eltype(a), eltype(b))
    aa = MPS([T.(a[k]) for k in 1:length(a)])
    bb = MPS([T.(b[k]) for k in 1:length(b)])
    return mps_sum(aa, bb)
end

function add_mpo(A::MPO, B::MPO)
    T = promote_type(eltype(A), eltype(B))
    AA = MPO([T.(A[k]) for k in 1:length(A)])
    BB = MPO([T.(B[k]) for k in 1:length(B)])
    return mpo_sum(AA, BB)
end

function concat_mps(a::MPS, b::MPS)
    T = promote_type(eltype(a), eltype(b))
    return MPS(vcat([T.(a[k]) for k in 1:length(a)],
                    [T.(b[k]) for k in 1:length(b)]))
end

function concat_mpo(A::MPO, B::MPO)
    T = promote_type(eltype(A), eltype(B))
    return MPO(vcat([T.(A[k]) for k in 1:length(A)],
                    [T.(B[k]) for k in 1:length(B)]))
end

function mps_to_type(psi::MPS, ::Type{T}) where {T<:Number}
    return MPS([T.(psi[k]) for k in 1:length(psi)])
end

function mpo_to_type(W::MPO, ::Type{T}) where {T<:Number}
    return MPO([T.(W[k]) for k in 1:length(W)])
end

function identity_mpo(N::Int; T=Float64)
    tensors = [zeros(T, 1, 2, 2, 1) for _ in 1:N]
    for W in tensors
        W[1, 1, 1, 1] = one(T)
        W[1, 2, 2, 1] = one(T)
    end
    return MPO(tensors)
end

first_basis_mps(N::Int; T=Float64) =
    product_state(T, [[one(T), zero(T)] for _ in 1:N])

constant_one_mps(N::Int; T=Float64) =
    product_state(T, [[one(T), one(T)] for _ in 1:N])

function dense_to_mps(v::AbstractVector{T}, N::Int;
                      cutoff::Real=1e-12,
                      max_dim::Union{Int,Nothing}=nothing) where {T<:Number}
    length(v) == 2^N || error("length(v) must equal 2^N")
    tensors = Vector{Array{T,3}}(undef, N)
    rest = reshape(copy(v), 1, :)
    left_dim = 1

    for site in 1:N-1
        F = svd(reshape(rest, left_dim * 2, :))
        threshold = isempty(F.S) ? 0.0 : cutoff * F.S[1]
        rank = max(1, count(s -> s > threshold, F.S))
        max_dim !== nothing && (rank = min(rank, max_dim))

        tensors[site] = reshape(F.U[:, 1:rank], left_dim, 2, rank)
        rest = Diagonal(F.S[1:rank]) * F.Vt[1:rank, :]
        left_dim = rank
    end

    tensors[end] = reshape(rest, left_dim, 2, 1)
    return MPS(tensors)
end

function mps_to_dense(psi::MPS)
    N = length(psi)
    T = eltype(psi)
    values = zeros(T, 2^N)
    for index in 0:(2^N - 1)
        row = reshape(T[one(T)], 1, 1)
        for site in 1:N
            bit = (index >> (site - 1)) & 1
            row = row * Matrix(@view psi[site][:, bit + 1, :])
        end
        values[index + 1] = row[1]
    end
    return values
end

function normalize_mps_euclidean(psi::MPS)
    nrm = sqrt(max(real(inner(psi, psi)), 0.0))
    nrm > eps(Float64) || error("Cannot normalize a zero MPS.")
    out = scale_mps(psi, 1 / nrm)
    move_center!(out, 1)
    return out
end

function phase_align_mps(candidate::MPS, reference::MPS)
    overlap = inner(reference, candidate)
    abs(overlap) <= eps(Float64) && return copy_mps(candidate)
    out = scale_mps(candidate, conj(overlap) / abs(overlap))
    move_center!(out, 1)
    return out
end

function relative_mps_change(new::MPS, old::MPS)
    aligned = phase_align_mps(new, old)
    diff = add_mps(aligned, scale_mps(old, -1))
    move_center!(diff, 1)
    denom = sqrt(max(real(inner(aligned, aligned)), 0.0))
    num = sqrt(max(real(inner(diff, diff)), 0.0))
    return num / max(denom, eps(Float64))
end

function global_residual_components(A::MPO, x::MPS, b::MPS)
    Ax = exact_apply_mpo(A, x)
    move_center!(Ax, 1)
    bb = copy_mps(b)
    move_center!(bb, 1)
    r = add_mps(Ax, scale_mps(bb, -1))
    move_center!(r, 1)
    rn = sqrt(max(real(inner(r, r)), 0.0))
    bn = sqrt(max(real(inner(bb, bb)), 0.0))
    rel = rn / max(bn, eps(Float64))
    return (absolute=rn, rhs_norm=bn, relative=rel)
end

global_residual(A::MPO, x::MPS, b::MPS) =
    global_residual_components(A, x, b).relative

"""
    sweep_linear_with_diagnostics!(engine; max_dim=nothing, cutoff=0.0, num_center=2)

One full right-and-left ALS/DMRG sweep, but additionally records the *maximum*
SVD discarded weight in the sweep.  JuliaQTT's public `sweep!` only returns the
average truncation error, which is not sufficient for diagnosing whether one
particular bond is limiting convergence.

`cutoff` in JuliaMPS is applied to the normalized density-matrix eigenvalues
lambda_i = |s_i|^2 / sum_j |s_j|^2.  The returned truncation values are the
actual discarded weight sum(discarded lambda_i).
"""
function sweep_linear_with_diagnostics!(engine;
                                        max_dim::Union{Int,Nothing}=nothing,
                                        cutoff::Real=0.0,
                                        num_center::Int=2)
    num_center in (1, 2) || error("num_center must be 1 or 2.")
    MPSCore.center(engine.x) == 1 ||
        error("x.center must be 1 at the start of each sweep; got $(MPSCore.center(engine.x)).")

    N = length(engine.x)
    local_residuals = Float64[]
    truncs = Float64[]

    for pos in 1:N-1
        res, tr = QTTCore._local_update!(engine, pos, num_center,
                                         max_dim, cutoff, "right")
        push!(local_residuals, Float64(res))
        push!(truncs, Float64(tr))
    end

    left_start = N - num_center + 1
    left_stop = num_center == 2 ? 1 : 2
    for pos in left_start:-1:left_stop
        res, tr = QTTCore._local_update!(engine, pos, num_center,
                                         max_dim, cutoff, "left")
        push!(local_residuals, Float64(res))
        push!(truncs, Float64(tr))
    end

    max_local = isempty(local_residuals) ? 0.0 : maximum(local_residuals)
    avg_trunc = isempty(truncs) ? 0.0 : mean(truncs)
    max_trunc = isempty(truncs) ? 0.0 : maximum(truncs)
    return max_local, avg_trunc, max_trunc
end

function solve_linear_mps(A::MPO, b::MPS, x0::MPS;
                          max_sweeps::Int=16,
                          global_tol::Real=1e-5,
                          max_bond::Int=48,
                          cutoff::Real=1e-10,
                          local_tol::Real=1e-12,
                          krylovdim::Int=120,
                          num_center::Int=2,
                          verbose::Bool=true)
    xstart = copy_mps(x0)
    move_center!(xstart, 1)
    bstart = copy_mps(b)
    move_center!(bstart, 1)
    engine = LinearSolverEngine(xstart, A, bstart;
                                krylovdim=krylovdim, tol=local_tol)
    hist = Tuple[]
    gres = Inf
    used = max_sweeps
    for s in 1:max_sweeps
        max_local, avg_trunc, max_trunc = sweep_linear_with_diagnostics!(
            engine; max_dim=max_bond, cutoff=cutoff, num_center=num_center)
        move_center!(engine.x, 1)
        rdiag = global_residual_components(A, engine.x, bstart)
        gres = rdiag.relative
        # Keep the first five fields compatible with the v3 history format.
        push!(hist, (s, max_local, avg_trunc, gres, max_dim(engine.x),
                     max_trunc, rdiag.absolute, rdiag.rhs_norm))
        verbose && @printf("%6d %14.4e %14.4e %12.3e %12.3e %8d\n",
                           s, max_local, gres, avg_trunc, max_trunc,
                           max_dim(engine.x))
        if gres < global_tol
            used = s
            break
        end
    end
    return engine.x, hist, gres, used
end

function history_max_truncation(hist)
    isempty(hist) && return 0.0
    return maximum(Float64(h[6]) for h in hist)
end

function history_bond_cap_hit(hist, max_bond::Int)
    isempty(hist) && return false
    return maximum(Int(h[5]) for h in hist) >= max_bond
end

function relative_mps_change_raw(new::MPS, old::MPS)
    diff = add_mps(new, scale_mps(old, -1))
    move_center!(diff, 1)
    denom = sqrt(max(real(inner(new, new)), 0.0))
    num = sqrt(max(real(inner(diff, diff)), 0.0))
    return num / max(denom, eps(Float64))
end

# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------

function expand_param(value, dim::Int)
    if value isa Number
        return fill(value, dim)
    end
    length(value) == dim || error("Expected $dim values, got $(length(value)).")
    return collect(value)
end

"""
Periodic QTT grid representing [-L,L) with N=2^bits points.
JuliaQTT's finite-difference spacing is inferred as (x2-x1)/(N-1), so the
stored interval ends at L-dx.  This yields exactly dx=2L/N and periodic wrap.
"""
function make_spatial_grid(dim::Int; bits=5, halfwidth=6.0)
    bitsv = Int.(expand_param(bits, dim))
    Lv = Float64.(expand_param(halfwidth, dim))
    grids = GridInfo[]
    coords = Vector{Vector{Float64}}()
    dxs = Float64[]
    shape = Int[]
    for d in 1:dim
        N = 2^bitsv[d]
        dx = 2 * Lv[d] / N
        lo = -Lv[d]
        hi = Lv[d] - dx
        push!(grids, GridInfo(bitsv[d], (lo, hi)))
        push!(coords, collect(range(lo, hi; length=N)))
        push!(dxs, dx)
        push!(shape, N)
    end
    return (grids=grids, coords=coords, dxs=dxs, shape=shape,
            bits=bitsv, halfwidth=Lv, cell_volume=prod(dxs),
            total_bits=sum(bitsv))
end

function make_time_grid(time_bits::Int, final_time::Real)
    Nt = 2^time_bits
    dt = Float64(final_time) / Nt
    grid = GridInfo(time_bits, (dt, Float64(final_time)))
    times = collect(dt:dt:Float64(final_time))
    return (grid=grid, Nt=Nt, dt=dt, times=times)
end

# -----------------------------------------------------------------------------
# Spatial QTT operators: dimension-general, sequential [x][y]...
# -----------------------------------------------------------------------------

function concat_all_mpo(parts::AbstractVector{<:MPO})
    isempty(parts) && error("No MPO parts supplied.")
    out = parts[1]
    for k in 2:length(parts)
        out = concat_mpo(out, parts[k])
    end
    return out
end

function lift_dimension_operator(op::MPO, target::Int, grids::Vector{GridInfo}; T=Float64)
    parts = MPO[]
    for d in eachindex(grids)
        if d == target
            push!(parts, mpo_to_type(op, T))
        else
            push!(parts, identity_mpo(grids[d].num_bits; T=T))
        end
    end
    return concat_all_mpo(parts)
end

function build_spatial_harmonic(grids::Vector{GridInfo};
                                a_kin::Real=0.5,
                                b_trap::Real=0.5,
                                bc::String="periodic")
    H = nothing
    for d in eachindex(grids)
        g = grids[d]
        lo, hi = g.interval
        D2 = qtto_diff2(g.num_bits, lo, hi; bc=bc).mpo
        xq = qtt_linear(g.num_bits, lo, hi)
        x2q = qtt_prod(xq, xq)
        X2 = to_qtto(x2q).mpo
        term = add_mpo(scale_mpo(lift_dimension_operator(D2, d, grids), -a_kin),
                       scale_mpo(lift_dimension_operator(X2, d, grids), b_trap))
        H = H === nothing ? term : add_mpo(H, term)
    end
    return H
end

function spatial_identity(grids::Vector{GridInfo}; T=Float64)
    return concat_all_mpo([identity_mpo(g.num_bits; T=T) for g in grids])
end

function qtt_diagonal_from_array(A::AbstractArray, grids::Vector{GridInfo};
                                 cutoff::Real=1e-10,
                                 max_bond::Int=48)
    bits = sum(g.num_bits for g in grids)
    mps = dense_to_mps(vec(A), bits; cutoff=cutoff, max_dim=max_bond)
    move_center!(mps, 1)
    q = QTT(grids, "sequential")
    q.mps = mps
    return to_qtto(q).mpo, max_dim(mps)
end

# -----------------------------------------------------------------------------
# State construction and diagnostics in the scaled discrete convention
# -----------------------------------------------------------------------------

function gaussian_scaled(grid; beta::Real=0.2, centers=nothing, momenta=nothing)
    dim = length(grid.shape)
    centers === nothing && (centers = zeros(dim))
    momenta === nothing && (momenta = zeros(dim))
    cv = Float64.(expand_param(centers, dim))
    kv = Float64.(expand_param(momenta, dim))
    A = Array{ComplexF64}(undef, Tuple(grid.shape))
    for I in CartesianIndices(A)
        exponent = 0.0
        phase = 0.0
        for d in 1:dim
            x = grid.coords[d][I[d]]
            exponent += (x - cv[d])^2
            phase += kv[d] * x
        end
        A[I] = exp(-beta * exponent) * exp(im * phase)
    end
    A ./= norm(vec(A))
    return A
end

function sho_exact_ground_scaled(grid; a_kin::Real=0.5, b_trap::Real=0.5)
    gamma = 0.5 * sqrt(b_trap / a_kin)
    return gaussian_scaled(grid; beta=gamma, centers=zeros(length(grid.shape)))
end

function field_to_spatial_mps(A::AbstractArray, grid;
                              cutoff::Real=1e-12,
                              max_bond::Int=64)
    psi = dense_to_mps(ComplexF64.(vec(A)), grid.total_bits;
                       cutoff=cutoff, max_dim=max_bond)
    move_center!(psi, 1)
    return psi
end

function spatial_mps_to_array(psi::MPS, grid)
    return reshape(mps_to_dense(psi), Tuple(grid.shape))
end

function spacetime_mps_to_array(psi::MPS, tgrid::GridInfo, spatial_grid)
    dims = Tuple(vcat([2^tgrid.num_bits], spatial_grid.shape))
    return reshape(mps_to_dense(psi), dims)
end

function normalized_final_slice(PsiST::AbstractArray)
    Nt = size(PsiST, 1)
    inds = ntuple(_ -> Colon(), ndims(PsiST)-1)
    slice = copy(view(PsiST, Nt, inds...))
    v = vec(slice)
    v ./= norm(v)
    return reshape(v, size(slice))
end

function normalize_spacetime_slices(PsiST::AbstractArray)
    rho = zeros(Float64, size(PsiST))
    Nt = size(PsiST, 1)
    restinds = ntuple(_ -> Colon(), ndims(PsiST)-1)
    for j in 1:Nt
        slice = copy(view(PsiST, j, restinds...))
        nrm = norm(vec(slice))
        if nrm <= eps(Float64)
            error("Encountered a zero imaginary-time slice while constructing normalized GPE density.")
        end
        normalized = slice ./ nrm
        view(rho, j, restinds...) .= abs2.(normalized)
    end
    return rho
end

function fidelity_scaled(a::AbstractArray, b::AbstractArray)
    av = vec(a) ./ norm(vec(a))
    bv = vec(b) ./ norm(vec(b))
    return abs(dot(av, bv))^2
end

function phase_aligned_relative_error_scaled(candidate::AbstractArray, reference::AbstractArray)
    a = vec(candidate) ./ norm(vec(candidate))
    b = vec(reference) ./ norm(vec(reference))
    ov = dot(b, a)
    abs(ov) > eps(Float64) && (a .*= conj(ov) / abs(ov))
    return norm(a - b)
end

function spatial_density_physical(state_scaled::AbstractArray, grid)
    return abs2.(state_scaled) ./ grid.cell_volume
end

function boundary_density_physical(state_scaled::AbstractArray, grid)
    rho = spatial_density_physical(state_scaled, grid)
    dim = ndims(rho)
    vals = Float64[]
    for d in 1:dim
        idx1 = ntuple(k -> k == d ? 1 : Colon(), dim)
        idx2 = ntuple(k -> k == d ? size(rho, d) : Colon(), dim)
        append!(vals, vec(copy(view(rho, idx1...))))
        append!(vals, vec(copy(view(rho, idx2...))))
    end
    return maximum(vals)
end

function coordinate_observables(state_scaled::AbstractArray, grid)
    rho = abs2.(state_scaled)
    nrm = sum(rho)
    means = zeros(Float64, ndims(state_scaled))
    r2mean = 0.0
    for I in CartesianIndices(state_scaled)
        w = rho[I] / max(nrm, eps(Float64))
        r2 = 0.0
        for d in 1:ndims(state_scaled)
            x = grid.coords[d][I[d]]
            means[d] += w * x
            r2 += x^2
        end
        r2mean += w * r2
    end
    return (means=means, rms_radius=sqrt(max(r2mean, 0.0)))
end

function stationary_gpe_diagnostics(state_scaled::AbstractArray, H0::MPO, grid, g_phys::Real;
                                    cutoff::Real=1e-12,
                                    max_bond::Int=64)
    psiA = state_scaled ./ norm(vec(state_scaled))
    psi = field_to_spatial_mps(psiA, grid; cutoff=cutoff, max_bond=max_bond)
    g_disc = g_phys / grid.cell_volume
    rho = abs2.(psiA)
    rho_mpo, rho_bond = qtt_diagonal_from_array(rho, grid.grids;
                                                cutoff=cutoff, max_bond=max_bond)
    Hnl = add_mpo(H0, scale_mpo(rho_mpo, g_disc))
    Hpsi = exact_apply_mpo(Hnl, psi)
    move_center!(Hpsi, 1)
    mu = real(inner(psi, Hpsi)) / max(real(inner(psi, psi)), eps(Float64))
    residual_mps = add_mps(Hpsi, scale_mps(psi, -mu))
    move_center!(residual_mps, 1)
    stat_res = sqrt(max(real(inner(residual_mps, residual_mps)), 0.0)) /
               max(abs(mu), eps(Float64))

    H0psi = exact_apply_mpo(H0, psi)
    linear_energy = real(inner(psi, H0psi)) / max(real(inner(psi, psi)), eps(Float64))
    interaction = 0.5 * g_disc * sum(abs2.(psiA) .^ 2)
    energy = linear_energy + interaction
    obs = coordinate_observables(psiA, grid)
    return (norm2=sum(abs2, psiA), energy=energy, mu=mu,
            stationary_residual=stat_res, density_bond=rho_bond,
            boundary_density=boundary_density_physical(psiA, grid),
            means=obs.means, rms_radius=obs.rms_radius)
end

function sho_spatial_energy(state_scaled::AbstractArray, H0::MPO, grid;
                            cutoff::Real=1e-12, max_bond::Int=64)
    psiA = state_scaled ./ norm(vec(state_scaled))
    psi = field_to_spatial_mps(psiA, grid; cutoff=cutoff, max_bond=max_bond)
    Hpsi = exact_apply_mpo(H0, psi)
    return real(inner(psi, Hpsi)) / max(real(inner(psi, psi)), eps(Float64))
end


# -----------------------------------------------------------------------------
# Conventional sparse references for validation
# -----------------------------------------------------------------------------

function periodic_second_difference_sparse(N::Int, dx::Real)
    D2 = spdiagm(-1 => ones(Float64, N-1),
                  0 => fill(-2.0, N),
                  1 => ones(Float64, N-1))
    D2[1, N] = 1.0
    D2[N, 1] = 1.0
    return D2 / Float64(dx)^2
end

function sparse_identity(N::Int)
    return spdiagm(0 => ones(Float64, N))
end

# Julia arrays use the first index as the fastest-varying index.  The QTT
# ordering [x][y][...] in this code follows the same convention, so an operator
# acting on spatial dimension d is kron(I_D,...,Op_d,...,I_1).
function sparse_operator_on_dimension(op::SparseMatrixCSC, grid, d::Int)
    dim = length(grid.shape)
    factors = Vector{SparseMatrixCSC{Float64,Int}}(undef, dim)
    for k in 1:dim
        factors[k] = k == d ? op : sparse_identity(grid.shape[k])
    end
    result = factors[end]
    for k in dim-1:-1:1
        result = kron(result, factors[k])
    end
    return sparse(result)
end

function build_spatial_harmonic_sparse(grid; a_kin::Real=0.5, b_trap::Real=0.5)
    Ntot = prod(grid.shape)
    H = spzeros(Float64, Ntot, Ntot)
    for d in 1:length(grid.shape)
        N = grid.shape[d]
        D2 = periodic_second_difference_sparse(N, grid.dxs[d])
        V = spdiagm(0 => Float64(b_trap) .* grid.coords[d].^2)
        Hd = -Float64(a_kin) .* D2 + V
        H = H + sparse_operator_on_dimension(Hd, grid, d)
    end
    return sparse(H)
end

function dense_sho_imaginary_reference(psi0_scaled::AbstractArray, grid, time;
                                       a_kin::Real=0.5,
                                       b_trap::Real=0.5,
                                       energy_shift::Real=0.0)
    H = build_spatial_harmonic_sparse(grid; a_kin=a_kin, b_trap=b_trap)
    Nsp = prod(grid.shape)
    Isp = sparse_identity(Nsp)
    A = Isp + time.dt .* (H - Float64(energy_shift) .* Isp)
    F = lu(A)

    psi = ComplexF64.(vec(psi0_scaled))
    Psi = zeros(ComplexF64, Tuple(vcat([time.Nt], grid.shape)))
    restinds = ntuple(_ -> Colon(), length(grid.shape))
    for j in 1:time.Nt
        psi = F \ psi
        view(Psi, j, restinds...) .= reshape(psi, Tuple(grid.shape))
    end
    final_state = copy(view(Psi, time.Nt, restinds...))
    final_state ./= norm(vec(final_state))
    return (trajectory=Psi, final_state=final_state, H=H)
end

function qtt_hamiltonian_action_error(H0::MPO, Hsparse::SparseMatrixCSC,
                                      state_scaled::AbstractArray, grid;
                                      cutoff::Real=0.0, max_bond::Int=256)
    psi = field_to_spatial_mps(state_scaled, grid; cutoff=cutoff, max_bond=max_bond)
    Hpsi_qtt = mps_to_dense(exact_apply_mpo(H0, psi))
    Hpsi_dense = Hsparse * ComplexF64.(vec(state_scaled))
    return norm(Hpsi_qtt - Hpsi_dense) / max(norm(Hpsi_dense), eps(Float64))
end


# -----------------------------------------------------------------------------
# SHO spectral / diagonalization references for poster-quality validation
# -----------------------------------------------------------------------------

function sho_1d_sparse_hamiltonian(coords::AbstractVector, dx::Real;
                                   a_kin::Real=0.5, b_trap::Real=0.5)
    N = length(coords)
    D2 = periodic_second_difference_sparse(N, dx)
    V = spdiagm(0 => Float64(b_trap) .* Float64.(coords).^2)
    return sparse(-Float64(a_kin) .* D2 + V)
end

function sho_1d_eigensystem(coords::AbstractVector, dx::Real;
                            a_kin::Real=0.5, b_trap::Real=0.5)
    H = sho_1d_sparse_hamiltonian(coords, dx; a_kin=a_kin, b_trap=b_trap)
    F = eigen(Symmetric(Matrix(H)))
    return (values=F.values, vectors=F.vectors, H=H)
end

"""
    sho_discrete_gap_period(grid; a_kin=0.5, b_trap=0.5)

Compute the lowest discrete SHO energy gap from the same periodic central-FD
Hamiltonian used by the QTT operator.  For the continuum a=b=1/2 oscillator,
the gap is omega=1 and the period is 2pi.  Using the discrete gap makes the
chosen time step a property of the actual spatial discretization rather than an
arbitrary input.
"""
function sho_discrete_gap_period(grid; a_kin::Real=0.5, b_trap::Real=0.5)
    es = sho_1d_eigensystem(grid.coords[1], grid.dxs[1];
                            a_kin=a_kin, b_trap=b_trap)
    gap = Float64(es.values[2] - es.values[1])
    gap > 0 || error("Non-positive SHO energy gap.")
    return (gap=gap, period=2pi/gap,
            E0=Float64(es.values[1]), E1=Float64(es.values[2]))
end

function _is_power_of_two(n::Int)
    return n > 0 && (n & (n - 1)) == 0
end

function gap_controlled_time_parameters(grid;
                                        periods::Int=2,
                                        steps_per_period::Int=64,
                                        a_kin::Real=0.5,
                                        b_trap::Real=0.5)
    periods > 0 || error("periods must be positive")
    steps_per_period > 0 || error("steps_per_period must be positive")
    Nt = periods * steps_per_period
    _is_power_of_two(Nt) || error(
        "periods*steps_per_period=$Nt must be a power of two for QTT time encoding.")
    bits = trailing_zeros(Nt)
    gp = sho_discrete_gap_period(grid; a_kin=a_kin, b_trap=b_trap)
    tf = periods * gp.period
    return (time_bits=bits, Nt=Nt, t_final=tf, dt=gp.period/steps_per_period,
            gap=gp.gap, period=gp.period, E0=gp.E0, E1=gp.E1)
end

function _normalized_1d_gaussian(coords, beta::Real, center::Real)
    f = ComplexF64.(exp.(-Float64(beta) .* (Float64.(coords) .- Float64(center)).^2))
    f ./= norm(f)
    return f
end

function _spectral_evolve_1d(f0::AbstractVector, eig, t::Real; imaginary::Bool=false)
    coeff = eig.vectors' * ComplexF64.(f0)
    if imaginary
        # Remove E0 to avoid harmless exponential underflow; normalized shape is unchanged.
        phase = exp.(-(eig.values .- eig.values[1]) .* Float64(t))
    else
        phase = exp.(-im .* eig.values .* Float64(t))
    end
    f = ComplexF64.(eig.vectors) * (ComplexF64.(phase) .* coeff)
    imaginary && (f ./= norm(f))
    return f
end

"""
    sho_spectral_reference_trajectory(grid, times; ...)

Exact-in-time reference for the *spatially discretized* SHO.  Each 1D finite-
difference Hamiltonian is diagonalized and propagated with exp(-i E t).  In 2D
the Hamiltonian and Gaussian initial state are separable, so the reference is
built from products of 1D spectral evolutions instead of diagonalizing a dense
N_x N_y by N_x N_y matrix.
"""
function sho_spectral_reference_trajectory(grid, times;
                                           a_kin::Real=0.5,
                                           b_trap::Real=0.5,
                                           displacement=nothing,
                                           imaginary::Bool=false,
                                           beta=nothing)
    dim = length(grid.shape)
    dim in (1, 2) || error("Spectral poster reference currently supports dim=1 or 2.")
    displacement === nothing && (displacement = zeros(dim))
    dv = Float64.(expand_param(displacement, dim))
    beta === nothing && (beta = 0.5 * sqrt(b_trap / a_kin))

    eigs = [sho_1d_eigensystem(grid.coords[d], grid.dxs[d];
                               a_kin=a_kin, b_trap=b_trap) for d in 1:dim]
    f0s = [_normalized_1d_gaussian(grid.coords[d], beta, dv[d]) for d in 1:dim]
    Nt = length(times)
    Psi = zeros(ComplexF64, Tuple(vcat([Nt], grid.shape)))

    if dim == 1
        for j in 1:Nt
            Psi[j, :] .= _spectral_evolve_1d(f0s[1], eigs[1], times[j];
                                             imaginary=imaginary)
        end
    else
        for j in 1:Nt
            fx = _spectral_evolve_1d(f0s[1], eigs[1], times[j]; imaginary=imaginary)
            fy = _spectral_evolve_1d(f0s[2], eigs[2], times[j]; imaginary=imaginary)
            Psi[j, :, :] .= reshape(fx, :, 1) .* reshape(fy, 1, :)
        end
    end
    return Psi
end


function sho_cn_diagonal_reference_trajectory(grid, times;
                                              a_kin::Real=0.5,
                                              b_trap::Real=0.5,
                                              displacement=nothing,
                                              beta=nothing)
    dim = length(grid.shape)
    dim in (1, 2) || error("CN diagonal reference currently supports dim=1 or 2.")
    displacement === nothing && (displacement = zeros(dim))
    dv = Float64.(expand_param(displacement, dim))
    beta === nothing && (beta = 0.5 * sqrt(b_trap / a_kin))
    length(times) >= 1 || error("times must be nonempty")
    dt = length(times) == 1 ? Float64(times[1]) : mean(diff(Float64.(times)))

    eigs = [sho_1d_eigensystem(grid.coords[d], grid.dxs[d];
                               a_kin=a_kin, b_trap=b_trap) for d in 1:dim]
    f0s = [_normalized_1d_gaussian(grid.coords[d], beta, dv[d]) for d in 1:dim]
    Nt = length(times)
    Psi = zeros(ComplexF64, Tuple(vcat([Nt], grid.shape)))

    if dim == 1
        eig = eigs[1]
        c0 = eig.vectors' * f0s[1]
        R = (1 .- 0.5im * dt .* eig.values) ./ (1 .+ 0.5im * dt .* eig.values)
        for j in 1:Nt
            f = ComplexF64.(eig.vectors) * (ComplexF64.(R .^ j) .* c0)
            Psi[j, :] .= f
        end
    else
        ex, ey = eigs
        cx = ex.vectors' * f0s[1]
        cy = ey.vectors' * f0s[2]
        C0 = cx * transpose(cy)
        E2 = reshape(ex.values, :, 1) .+ reshape(ey.values, 1, :)
        R = (1 .- 0.5im * dt .* E2) ./ (1 .+ 0.5im * dt .* E2)
        Vx = ComplexF64.(ex.vectors)
        Vy = ComplexF64.(ey.vectors)
        for j in 1:Nt
            Cj = C0 .* (R .^ j)
            Psi[j, :, :] .= Vx * Cj * transpose(Vy)
        end
    end
    return Psi
end

function phase_aligned_relative_error_preserve_norm(candidate::AbstractArray,
                                                    reference::AbstractArray)
    a = ComplexF64.(vec(candidate))
    b = ComplexF64.(vec(reference))
    ov = dot(b, a)
    abs(ov) > eps(Float64) && (a .*= conj(ov) / abs(ov))
    return norm(a - b) / max(norm(b), eps(Float64))
end

function dominant_angular_frequency(times::AbstractVector, values::AbstractVector)
    length(times) == length(values) || error("times and values must have same length")
    N = length(values)
    N >= 4 || return NaN
    dt = mean(diff(Float64.(times)))
    y = Float64.(values) .- mean(Float64.(values))
    best_k = 1
    best_amp = -Inf
    for k in 1:fld(N, 2)
        omega = 2pi * k / (N * dt)
        amp = abs(sum(y[j] * exp(-im * omega * Float64(times[j])) for j in 1:N))
        if amp > best_amp
            best_amp = amp
            best_k = k
        end
    end
    return 2pi * best_k / (N * dt)
end

function gpe_energy_dense_scaled(state_scaled::AbstractArray, Hsparse::SparseMatrixCSC,
                                 grid, g_phys::Real)
    psi = ComplexF64.(vec(state_scaled))
    nrm = norm(psi)
    nrm > eps(Float64) || return NaN
    psi ./= nrm
    g_disc = Float64(g_phys) / grid.cell_volume
    linear = real(dot(psi, Hsparse * psi))
    interaction = 0.5 * g_disc * sum(abs2.(psi).^2)
    return linear + interaction
end

# -----------------------------------------------------------------------------
# Space-time operator builders
# -----------------------------------------------------------------------------

function build_imaginary_sho_allatonce(H0::MPO, spatial_grid, time;
                                       energy_shift::Real=0.0)
    T = Float64
    It = identity_mpo(time.grid.num_bits; T=T)
    Isp = spatial_identity(spatial_grid.grids; T=T)
    Ist = concat_mpo(It, Isp)
    Hst = concat_mpo(It, H0)
    Sback = shift_backward_mpo(time.grid.num_bits; bc="dirichlet")
    Sst = concat_mpo(Sback, Isp)
    A = add_mpo(Ist, scale_mpo(Hst, time.dt))
    if energy_shift != 0
        A = add_mpo(A, scale_mpo(Ist, -time.dt * energy_shift))
    end
    A = add_mpo(A, scale_mpo(Sst, -1.0))
    return A
end

function build_imaginary_gpe_allatonce(H0::MPO, density_st_mpo::MPO,
                                       spatial_grid, time;
                                       g_discrete::Real,
                                       energy_shift::Real=0.0)
    T = promote_type(eltype(H0), eltype(density_st_mpo))
    It = identity_mpo(time.grid.num_bits; T=T)
    Isp = spatial_identity(spatial_grid.grids; T=T)
    Ist = concat_mpo(It, Isp)
    Hst = concat_mpo(It, mpo_to_type(H0, T))
    Sback = mpo_to_type(shift_backward_mpo(time.grid.num_bits; bc="dirichlet"), T)
    Sst = concat_mpo(Sback, Isp)
    A = add_mpo(Ist, scale_mpo(Hst, time.dt))
    A = add_mpo(A, scale_mpo(density_st_mpo, time.dt * g_discrete))
    if energy_shift != 0
        A = add_mpo(A, scale_mpo(Ist, -time.dt * energy_shift))
    end
    A = add_mpo(A, scale_mpo(Sst, -1.0))
    return A
end

function build_realtime_cn_allatonce(H0::MPO, spatial_grid, time)
    T = ComplexF64
    Hc = mpo_to_type(H0, T)
    It = identity_mpo(time.grid.num_bits; T=T)
    Isp = spatial_identity(spatial_grid.grids; T=T)
    Aplus = add_mpo(Isp, scale_mpo(Hc, 0.5im * time.dt))
    Aminus = add_mpo(Isp, scale_mpo(Hc, -0.5im * time.dt))
    Sback = mpo_to_type(shift_backward_mpo(time.grid.num_bits; bc="dirichlet"), T)
    Ast = add_mpo(concat_mpo(It, Aplus),
                  scale_mpo(concat_mpo(Sback, Aminus), -1.0))
    return Ast, Aplus, Aminus
end

function spacetime_rhs_imaginary(psi0::MPS, time_bits::Int)
    e1 = first_basis_mps(time_bits; T=eltype(psi0))
    rhs = concat_mps(e1, psi0)
    move_center!(rhs, 1)
    return rhs
end

function spacetime_rhs_cn(psi0::MPS, Aminus::MPO, time_bits::Int)
    rhs_space = exact_apply_mpo(Aminus, psi0)
    move_center!(rhs_space, 1)
    e1 = first_basis_mps(time_bits; T=eltype(rhs_space))
    rhs = concat_mps(e1, rhs_space)
    move_center!(rhs, 1)
    return rhs
end

function repeated_spacetime_guess(psi0::MPS, time_bits::Int)
    ones_t = constant_one_mps(time_bits; T=eltype(psi0))
    guess = concat_mps(ones_t, psi0)
    move_center!(guess, 1)
    return guess
end

# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            vals = String[]
            for v in row
                if v isa Integer
                    push!(vals, string(v))
                elseif v isa AbstractString
                    push!(vals, v)
                else
                    push!(vals, @sprintf("%.16e", Float64(v)))
                end
            end
            println(io, join(vals, ","))
        end
    end
end

function save_ground_state_csv(path, state_scaled::AbstractArray, grid)
    rho_phys = spatial_density_physical(state_scaled, grid)
    open(path, "w") do io
        if ndims(state_scaled) == 1
            println(io, "x,real_psi_tilde,imag_psi_tilde,physical_density")
            for i in eachindex(grid.coords[1])
                @printf(io, "%.16e,%.16e,%.16e,%.16e\n",
                        grid.coords[1][i], real(state_scaled[i]), imag(state_scaled[i]), rho_phys[i])
            end
        elseif ndims(state_scaled) == 2
            println(io, "x,y,real_psi_tilde,imag_psi_tilde,physical_density")
            for j in eachindex(grid.coords[2]), i in eachindex(grid.coords[1])
                @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e\n",
                        grid.coords[1][i], grid.coords[2][j],
                        real(state_scaled[i,j]), imag(state_scaled[i,j]), rho_phys[i,j])
            end
        else
            error("CSV ground-state writer currently supports dim=1 or 2.")
        end
    end
end

function save_state_serialized(path, state_scaled, grid, metadata)
    open(path, "w") do io
        serialize(io, Dict("state_scaled" => state_scaled,
                           "coords" => grid.coords,
                           "dxs" => grid.dxs,
                           "cell_volume" => grid.cell_volume,
                           "metadata" => metadata))
    end
end

# -----------------------------------------------------------------------------
# Benchmark runner 1: SHO imaginary-time all-at-once
# -----------------------------------------------------------------------------

function run_sho_imaginary_allatonce(; dim::Int,
                                     bits=5,
                                     halfwidth=6.0,
                                     time_bits::Int=5,
                                     tau_final::Real=4.0,
                                     a_kin::Real=0.5,
                                     b_trap::Real=0.5,
                                     beta0::Real=0.2,
                                     max_sweeps::Int=20,
                                     global_tol::Real=1e-5,
                                     max_bond::Int=96,
                                     cutoff::Real=1e-10,
                                     truncation_target::Real=1e-8,
                                     output_name::String="sho_imaginary")
    outdir = joinpath(PROJECT_ROOT, "validation_outputs", output_name)
    mkpath(outdir)
    grid = make_spatial_grid(dim; bits=bits, halfwidth=halfwidth)
    time = make_time_grid(time_bits, tau_final)
    H0 = build_spatial_harmonic(grid.grids; a_kin=a_kin, b_trap=b_trap, bc="periodic")

    psi0A = gaussian_scaled(grid; beta=beta0, centers=zeros(dim))
    psi0 = field_to_spatial_mps(psi0A, grid; cutoff=cutoff, max_bond=max_bond)

    # A constant energy shift changes only the overall imaginary-time amplitude,
    # not the normalized state.  Choosing the exact SHO E0 prevents severe decay.
    E0_exact = dim * sqrt(a_kin * b_trap)
    Ast = build_imaginary_sho_allatonce(H0, grid, time; energy_shift=E0_exact)
    rhs = spacetime_rhs_imaginary(psi0, time_bits)
    guess = repeated_spacetime_guess(psi0, time_bits)

    @printf("SHO imaginary-time all-at-once, dim=%d\n", dim)
    @printf("grid=%s, Nt=%d, dtau=%.6e, domain halfwidth=%s\n",
            string(grid.shape), time.Nt, time.dt, string(grid.halfwidth))
    @printf("%6s %14s %14s %12s %12s %8s\n",
            "sweep", "local_res", "global_res", "avg_trunc", "max_trunc", "chi")
    sol, hist, gres, used = solve_linear_mps(Ast, rhs, guess;
                                              max_sweeps=max_sweeps,
                                              global_tol=global_tol,
                                              max_bond=max_bond,
                                              cutoff=cutoff,
                                              verbose=true)
    max_sweep_trunc = history_max_truncation(hist)
    bond_cap_hit = history_bond_cap_hit(hist, max_bond)

    PsiST = spacetime_mps_to_array(sol, time.grid, grid)
    final_state = normalized_final_slice(PsiST)

    # Two references are deliberately kept separate:
    #   (1) dense_ref solves exactly the SAME finite-difference + Backward-Euler
    #       discretization and therefore validates the QTT/all-at-once solver;
    #   (2) exact_state is the continuum SHO Gaussian and therefore measures grid
    #       discretization error, not tensor-network solver error.
    dense_ref = dense_sho_imaginary_reference(psi0A, grid, time;
                                               a_kin=a_kin, b_trap=b_trap,
                                               energy_shift=E0_exact)
    dense_state = dense_ref.final_state
    exact_state = sho_exact_ground_scaled(grid; a_kin=a_kin, b_trap=b_trap)

    fid_dense = fidelity_scaled(final_state, dense_state)
    relerr_dense = phase_aligned_relative_error_scaled(final_state, dense_state)
    fid_continuum = fidelity_scaled(final_state, exact_state)
    relerr_continuum = phase_aligned_relative_error_scaled(final_state, exact_state)
    dense_grid_error = phase_aligned_relative_error_scaled(dense_state, exact_state)

    # Verify that the QTT Hamiltonian itself matches the conventional sparse
    # finite-difference Hamiltonian on the same grid.
    H_action_err = qtt_hamiltonian_action_error(H0, dense_ref.H, psi0A, grid;
                                                 cutoff=0.0, max_bond=max_bond)

    # Verify the all-at-once operator independently by inserting the conventional
    # sequential BE trajectory into the QTT operator.
    dense_st_mps = dense_to_mps(vec(dense_ref.trajectory),
                                time.grid.num_bits + grid.total_bits;
                                cutoff=0.0, max_dim=nothing)
    move_center!(dense_st_mps, 1)
    dense_reference_operator_residual = global_residual(Ast, dense_st_mps, rhs)

    E = sho_spatial_energy(final_state, H0, grid; cutoff=cutoff, max_bond=max_bond)
    boundary = boundary_density_physical(final_state, grid)

    write_csv(joinpath(outdir, "sweep_history.csv"),
              ["sweep","max_local_residual","average_truncation","global_residual",
               "max_bond","max_truncation","absolute_residual_2norm","rhs_2norm"], hist)
    save_ground_state_csv(joinpath(outdir, "ground_state.csv"), final_state, grid)

    # PASS means the QTT/MPS solver reproduces the SAME discrete problem.
    # Agreement with the continuum Gaussian is reported separately as a grid check.
    truncation_pass = max_sweep_trunc <= truncation_target && !bond_cap_hit
    discrete_pass = gres < global_tol &&
                    dense_reference_operator_residual < 1e-10 &&
                    H_action_err < 1e-10 &&
                    relerr_dense < 1e-3 &&
                    boundary < 1e-6 && truncation_pass
    continuum_grid_check = relerr_continuum < 1e-3
    pass = discrete_pass
    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "SHO imaginary-time space-time all-at-once")
        println(io, "dim = ", dim)
        println(io, "validation_pass = ", pass)
        println(io, "discrete_validation_pass = ", discrete_pass)
        println(io, "continuum_grid_check_pass = ", continuum_grid_check)
        println(io, "grid_shape = ", grid.shape)
        println(io, "domain_halfwidth = ", grid.halfwidth)
        println(io, "periodic_bc = true")
        println(io, "tau_final = ", tau_final)
        println(io, "dtau = ", time.dt)
        println(io, "energy_shift = ", E0_exact)
        println(io, "global_residual = ", gres)
        println(io, "used_sweeps = ", used)
        println(io, "solution_max_bond = ", max_dim(sol))
        println(io, "max_allowed_bond = ", max_bond)
        println(io, "bond_cap_hit = ", bond_cap_hit)
        println(io, "svd_cutoff = ", cutoff)
        println(io, "max_sweep_discarded_weight = ", max_sweep_trunc)
        println(io, "truncation_target = ", truncation_target)
        println(io, "truncation_pass = ", truncation_pass)
        println(io, "QTT_vs_dense_fidelity = ", fid_dense)
        println(io, "QTT_vs_dense_relative_L2_error = ", relerr_dense)
        println(io, "dense_reference_residual_in_QTT_operator = ", dense_reference_operator_residual)
        println(io, "QTT_Hamiltonian_action_error = ", H_action_err)
        println(io, "QTT_vs_continuum_fidelity = ", fid_continuum)
        println(io, "QTT_vs_continuum_relative_L2_error = ", relerr_continuum)
        println(io, "dense_discretization_relative_L2_error_vs_continuum = ", dense_grid_error)
        println(io, "numerical_energy = ", E)
        println(io, "continuum_exact_energy = ", E0_exact)
        println(io, "boundary_physical_density = ", boundary)
    end

    if dim == 1
        p = plot(grid.coords[1], spatial_density_physical(final_state, grid);
                 label="all-at-once", linewidth=2, xlabel="x", ylabel="density",
                 title="SHO imaginary-time ground state")
        plot!(p, grid.coords[1], spatial_density_physical(dense_state, grid);
              label="dense BE", linestyle=:dot, linewidth=2)
        plot!(p, grid.coords[1], spatial_density_physical(exact_state, grid);
              label="continuum analytic", linestyle=:dash, linewidth=2)
        savefig(p, joinpath(outdir, "ground_state.png"))
    else
        p = heatmap(grid.coords[1], grid.coords[2],
                    permutedims(spatial_density_physical(final_state, grid), (2,1));
                    xlabel="x", ylabel="y", aspect_ratio=:equal,
                    title="2D SHO ground-state density")
        savefig(p, joinpath(outdir, "ground_state.png"))
    end

    @printf("\nValidation: %s\n", pass ? "PASS" : "CHECK")
    @printf("global residual = %.6e\n", gres)
    @printf("dense reference residual in QTT operator = %.6e\n", dense_reference_operator_residual)
    @printf("QTT Hamiltonian action error = %.6e\n", H_action_err)
    @printf("QTT vs dense: fidelity = %.12f, relL2 = %.6e\n", fid_dense, relerr_dense)
    @printf("QTT vs continuum: fidelity = %.12f, relL2 = %.6e (grid error)\n", fid_continuum, relerr_continuum)
    @printf("SVD max discarded weight = %.3e (target %.1e), bond cap hit=%s\n",
            max_sweep_trunc, truncation_target, string(bond_cap_hit))
    @printf("Output: %s\n", outdir)
    return nothing
end

# -----------------------------------------------------------------------------
# Benchmark runner 2: GPE imaginary-time all-at-once
# -----------------------------------------------------------------------------

function run_gpe_imaginary_allatonce(; dim::Int,
                                     bits=5,
                                     halfwidth=6.0,
                                     time_bits::Int=4,
                                     tau_final::Real=2.0,
                                     a_kin::Real=0.5,
                                     b_trap::Real=0.5,
                                     g_phys::Real=12.5,
                                     beta0::Real=0.2,
                                     relaxation::Real=0.5,
                                     max_picard::Int=20,
                                     picard_tol::Real=1e-5,
                                     max_inner_sweeps::Int=12,
                                     inner_global_tol::Real=2e-5,
                                     max_bond::Int=128,
                                     density_max_bond::Int=128,
                                     cutoff::Real=1e-10,
                                     density_cutoff::Real=1e-10,
                                     truncation_target::Real=1e-8,
                                     output_name::String="gpe_imaginary")
    outdir = joinpath(PROJECT_ROOT, "validation_outputs", output_name)
    mkpath(outdir)
    grid = make_spatial_grid(dim; bits=bits, halfwidth=halfwidth)
    time = make_time_grid(time_bits, tau_final)
    H0 = build_spatial_harmonic(grid.grids; a_kin=a_kin, b_trap=b_trap, bc="periodic")
    g_disc = g_phys / grid.cell_volume

    psi0A = gaussian_scaled(grid; beta=beta0, centers=zeros(dim))
    psi0 = field_to_spatial_mps(psi0A, grid; cutoff=cutoff, max_bond=max_bond)
    rhs = spacetime_rhs_imaginary(psi0, time_bits)
    guess = repeated_spacetime_guess(psi0, time_bits)

    # Fixed energy shift estimated from the initial normalized state.  A constant
    # shift does not alter normalized imaginary-time states; it only improves the
    # raw amplitude scale of the all-at-once trajectory.
    init_diag = stationary_gpe_diagnostics(psi0A, H0, grid, g_phys;
                                           cutoff=cutoff, max_bond=max_bond)
    energy_shift = init_diag.mu

    st_grids = vcat([time.grid], grid.grids)
    hist = Tuple[]
    final_change = Inf
    final_inner = Inf
    final_density_bond = 1
    max_inner_trunc = 0.0
    bond_cap_hit = false
    converged = false

    @printf("GPE imaginary-time all-at-once, dim=%d, g=%.6g\n", dim, g_phys)
    @printf("grid=%s, Nt=%d, dtau=%.6e, g_discrete=%.6e\n",
            string(grid.shape), time.Nt, time.dt, g_disc)
    @printf("%7s %13s %13s %8s %8s %8s %12s\n",
            "Picard", "change", "inner_res", "sweeps", "chi", "chi_rho", "max_trunc")

    for p in 1:max_picard
        PsiGuess = spacetime_mps_to_array(guess, time.grid, grid)
        rho_st = normalize_spacetime_slices(PsiGuess)
        rho_mpo, rho_bond = qtt_diagonal_from_array(rho_st, st_grids;
                                                    cutoff=density_cutoff,
                                                    max_bond=density_max_bond)
        final_density_bond = rho_bond
        Ast = build_imaginary_gpe_allatonce(H0, rho_mpo, grid, time;
                                            g_discrete=g_disc,
                                            energy_shift=energy_shift)
        candidate, inner_hist, final_inner, used_sweeps = solve_linear_mps(
            Ast, rhs, guess;
            max_sweeps=max_inner_sweeps,
            global_tol=inner_global_tol,
            max_bond=max_bond,
            cutoff=cutoff,
            verbose=false)
        iter_max_trunc = history_max_truncation(inner_hist)
        max_inner_trunc = max(max_inner_trunc, iter_max_trunc)
        bond_cap_hit = bond_cap_hit || history_bond_cap_hit(inner_hist, max_bond)

        candidate = phase_align_mps(candidate, guess)
        mixed = add_mps(scale_mps(guess, 1-relaxation),
                        scale_mps(candidate, relaxation))
        mixed = svd_compress_mps(mixed; max_dim=max_bond, cutoff=cutoff)
        move_center!(mixed, 1)
        final_change = relative_mps_change(mixed, guess)
        guess = mixed
        push!(hist, (p, final_change, final_inner, used_sweeps,
                     max_dim(guess), rho_bond, iter_max_trunc))
        @printf("%7d %13.4e %13.4e %8d %8d %8d %12.3e\n",
                p, final_change, final_inner, used_sweeps,
                max_dim(guess), rho_bond, iter_max_trunc)

        if final_change < picard_tol && final_inner < inner_global_tol
            converged = true
            break
        end
    end

    PsiST = spacetime_mps_to_array(guess, time.grid, grid)
    final_state = normalized_final_slice(PsiST)
    diag = stationary_gpe_diagnostics(final_state, H0, grid, g_phys;
                                      cutoff=cutoff, max_bond=max_bond)

    write_csv(joinpath(outdir, "picard_history.csv"),
              ["picard","relative_change","inner_global_residual","inner_sweeps",
               "solution_max_bond","density_max_bond","max_inner_truncation"],
              hist)
    save_ground_state_csv(joinpath(outdir, "ground_state.csv"), final_state, grid)
    metadata = Dict("dim"=>dim, "g"=>Float64(g_phys), "a"=>Float64(a_kin),
                    "b"=>Float64(b_trap), "halfwidth"=>grid.halfwidth,
                    "bits"=>grid.bits, "bc"=>"periodic",
                    "normalization"=>"sum(abs2(psi_tilde))=1")
    save_state_serialized(joinpath(outdir, "ground_state_state.jls"),
                          final_state, grid, metadata)

    truncation_pass = max_inner_trunc <= truncation_target && !bond_cap_hit
    pass = converged && diag.stationary_residual < 2e-4 &&
           abs(diag.norm2 - 1) < 1e-10 && diag.boundary_density < 1e-5 &&
           truncation_pass
    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "GPE imaginary-time space-time all-at-once")
        println(io, "dim = ", dim)
        println(io, "validation_pass = ", pass)
        println(io, "picard_converged = ", converged)
        println(io, "grid_shape = ", grid.shape)
        println(io, "domain_halfwidth = ", grid.halfwidth)
        println(io, "periodic_bc = true")
        println(io, "g_physical = ", g_phys)
        println(io, "g_discrete = ", g_disc)
        println(io, "tau_final = ", tau_final)
        println(io, "dtau = ", time.dt)
        println(io, "relaxation = ", relaxation)
        println(io, "energy_shift = ", energy_shift)
        println(io, "final_picard_change = ", final_change)
        println(io, "final_inner_global_residual = ", final_inner)
        println(io, "solution_max_bond = ", max_dim(guess))
        println(io, "density_max_bond = ", final_density_bond)
        println(io, "max_allowed_solution_bond = ", max_bond)
        println(io, "bond_cap_hit = ", bond_cap_hit)
        println(io, "svd_cutoff = ", cutoff)
        println(io, "max_inner_sweep_discarded_weight = ", max_inner_trunc)
        println(io, "truncation_target = ", truncation_target)
        println(io, "truncation_pass = ", truncation_pass)
        println(io, "ground_state_norm2 = ", diag.norm2)
        println(io, "energy = ", diag.energy)
        println(io, "chemical_potential = ", diag.mu)
        println(io, "stationary_residual = ", diag.stationary_residual)
        println(io, "boundary_physical_density = ", diag.boundary_density)
        println(io, "rms_radius = ", diag.rms_radius)
    end

    if dim == 1
        pfig = plot(grid.coords[1], spatial_density_physical(final_state, grid);
                    xlabel="x", ylabel="density", linewidth=2, label="GPE GS",
                    title="1D GPE ground state, g=$(g_phys)")
        savefig(pfig, joinpath(outdir, "ground_state.png"))
    else
        pfig = heatmap(grid.coords[1], grid.coords[2],
                       permutedims(spatial_density_physical(final_state, grid), (2,1));
                       xlabel="x", ylabel="y", aspect_ratio=:equal,
                       title="2D GPE ground-state density, g=$(g_phys)")
        savefig(pfig, joinpath(outdir, "ground_state.png"))
    end

    @printf("\nValidation: %s\n", pass ? "PASS" : "CHECK")
    @printf("Picard converged=%s, stationary residual=%.6e, boundary density=%.6e\n",
            string(converged), diag.stationary_residual, diag.boundary_density)
    @printf("SVD max discarded weight=%.3e (target %.1e), bond cap hit=%s\n",
            max_inner_trunc, truncation_target, string(bond_cap_hit))
    @printf("Reusable state: %s\n", joinpath(outdir, "ground_state_state.jls"))
    @printf("Output: %s\n", outdir)
    return nothing
end

# -----------------------------------------------------------------------------
# Benchmark runner 3: SHO real-time all-at-once + quantitative poster diagnostics
# -----------------------------------------------------------------------------

function run_sho_realtime_allatonce(; dim::Int,
                                    bits=5,
                                    halfwidth=6.0,
                                    time_bits::Int=5,
                                    t_final::Real=2pi,
                                    time_from_gap::Bool=false,
                                    periods::Int=1,
                                    steps_per_period::Int=32,
                                    a_kin::Real=0.5,
                                    b_trap::Real=0.5,
                                    displacement=nothing,
                                    max_sweeps::Int=24,
                                    global_tol::Real=1e-5,
                                    max_bond::Int=128,
                                    cutoff::Real=1e-10,
                                    truncation_target::Real=1e-8,
                                    spectral_rmse_tol::Real=1e-2,
                                    cn_reference_rmse_tol::Real=1e-3,
                                    make_movie::Bool=true,
                                    output_name::String="sho_realtime")
    outdir = joinpath(PROJECT_ROOT, "validation_outputs", output_name)
    mkpath(outdir)
    grid = make_spatial_grid(dim; bits=bits, halfwidth=halfwidth)

    gap_info = sho_discrete_gap_period(grid; a_kin=a_kin, b_trap=b_trap)
    if time_from_gap
        tp = gap_controlled_time_parameters(grid; periods=periods,
                                            steps_per_period=steps_per_period,
                                            a_kin=a_kin, b_trap=b_trap)
        time_bits = tp.time_bits
        t_final = tp.t_final
        gap_info = (gap=tp.gap, period=tp.period, E0=tp.E0, E1=tp.E1)
    end
    time = make_time_grid(time_bits, t_final)

    H0r = build_spatial_harmonic(grid.grids; a_kin=a_kin, b_trap=b_trap, bc="periodic")
    H0 = mpo_to_type(H0r, ComplexF64)

    displacement === nothing && (displacement = vcat([2.0], zeros(dim-1)))
    dv = Float64.(expand_param(displacement, dim))
    gamma = 0.5 * sqrt(b_trap / a_kin)
    psi0A = gaussian_scaled(grid; beta=gamma, centers=dv)
    psi0 = field_to_spatial_mps(psi0A, grid; cutoff=cutoff, max_bond=max_bond)

    Ast, _, Aminus = build_realtime_cn_allatonce(H0, grid, time)
    rhs = spacetime_rhs_cn(psi0, Aminus, time_bits)
    guess = repeated_spacetime_guess(psi0, time_bits)

    @printf("SHO real-time all-at-once, dim=%d\n", dim)
    @printf("grid=%s, Nt=%d, dt=%.6e, t_final=%.6e, displacement=%s\n",
            string(grid.shape), time.Nt, time.dt, t_final, string(dv))
    @printf("discrete gap DeltaE=%.8e, period=%.8e, steps/period=%.2f\n",
            gap_info.gap, gap_info.period, gap_info.period/time.dt)
    @printf("%6s %14s %14s %12s %12s %8s\n",
            "sweep", "local_res", "global_res", "avg_trunc", "max_trunc", "chi")

    sol, hist, gres, used = solve_linear_mps(Ast, rhs, guess;
                                              max_sweeps=max_sweeps,
                                              global_tol=global_tol,
                                              max_bond=max_bond,
                                              cutoff=cutoff,
                                              verbose=true)
    PsiST = spacetime_mps_to_array(sol, time.grid, grid)
    spectral = sho_spectral_reference_trajectory(grid, time.times;
                                                  a_kin=a_kin, b_trap=b_trap,
                                                  displacement=dv,
                                                  imaginary=false,
                                                  beta=gamma)
    cn_reference = sho_cn_diagonal_reference_trajectory(grid, time.times;
                                                          a_kin=a_kin, b_trap=b_trap,
                                                          displacement=dv, beta=gamma)

    max_sweep_trunc = history_max_truncation(hist)
    bond_cap_hit = history_bond_cap_hit(hist, max_bond)
    rdiag = global_residual_components(Ast, sol, rhs)

    rows = Tuple[]
    norm_drifts = Float64[]
    energy_values = Float64[]
    qtt_cn_errors = Float64[]
    cn_exact_errors = Float64[]
    qtt_exact_errors = Float64[]
    cn_fidelities = Float64[]
    exact_fidelities = Float64[]
    center_errors_cn = Float64[]
    center_errors_exact = Float64[]

    for j in 1:time.Nt
        restinds = ntuple(_ -> Colon(), dim)
        state = copy(view(PsiST, j, restinds...))
        ref_cn = copy(view(cn_reference, j, restinds...))
        ref_exact = copy(view(spectral, j, restinds...))
        norm2 = sum(abs2, state)
        obs = coordinate_observables(state, grid)
        obs_cn = coordinate_observables(ref_cn, grid)
        obs_exact = coordinate_observables(ref_exact, grid)
        E = sho_spatial_energy(state, H0r, grid; cutoff=cutoff, max_bond=max_bond)

        err_qtt_cn = phase_aligned_relative_error_scaled(state, ref_cn)
        err_cn_exact = phase_aligned_relative_error_scaled(ref_cn, ref_exact)
        err_qtt_exact = phase_aligned_relative_error_scaled(state, ref_exact)
        fid_cn = fidelity_scaled(state, ref_cn)
        fid_exact = fidelity_scaled(state, ref_exact)
        cerr_cn = norm(obs.means .- obs_cn.means)
        cerr_exact = norm(obs.means .- obs_exact.means)

        push!(norm_drifts, abs(norm2 - 1.0))
        push!(energy_values, E)
        push!(qtt_cn_errors, err_qtt_cn)
        push!(cn_exact_errors, err_cn_exact)
        push!(qtt_exact_errors, err_qtt_exact)
        push!(cn_fidelities, fid_cn)
        push!(exact_fidelities, fid_exact)
        push!(center_errors_cn, cerr_cn)
        push!(center_errors_exact, cerr_exact)

        if dim == 1
            push!(rows, (time.times[j], norm2, obs.means[1], obs_cn.means[1], obs_exact.means[1],
                         obs.rms_radius, E, err_qtt_cn, err_cn_exact, err_qtt_exact,
                         fid_cn, fid_exact, cerr_cn, cerr_exact))
        else
            push!(rows, (time.times[j], norm2, obs.means[1], obs.means[2],
                         obs_cn.means[1], obs_cn.means[2], obs_exact.means[1], obs_exact.means[2],
                         obs.rms_radius, E, err_qtt_cn, err_cn_exact, err_qtt_exact,
                         fid_cn, fid_exact, cerr_cn, cerr_exact))
        end
    end

    energy_drift = maximum(abs.(energy_values .- energy_values[1])) /
                   max(abs(energy_values[1]), eps(Float64))
    norm_drift = maximum(norm_drifts)
    center_rmse = sqrt(mean(center_errors_cn .^ 2))
    qtt_cn_rmse = sqrt(mean(qtt_cn_errors .^ 2))
    qtt_cn_max = maximum(qtt_cn_errors)
    cn_exact_rmse = sqrt(mean(cn_exact_errors .^ 2))
    cn_exact_max = maximum(cn_exact_errors)
    spectral_rmse = sqrt(mean(qtt_exact_errors .^ 2))
    spectral_max = maximum(qtt_exact_errors)
    min_cn_fidelity = minimum(cn_fidelities)
    min_fidelity = minimum(exact_fidelities)

    write_csv(joinpath(outdir, "sweep_history.csv"),
              ["sweep","max_local_residual","average_truncation","global_residual",
               "max_bond","max_truncation","absolute_residual_2norm","rhs_2norm"], hist)
    if dim == 1
        write_csv(joinpath(outdir, "time_observables.csv"),
                  ["time","norm2","x_mean","cn_diag_x_mean","exact_spectral_x_mean","rms_radius","energy",
                   "qtt_vs_cn_relL2","cn_vs_exact_relL2","qtt_vs_exact_relL2",
                   "qtt_vs_cn_fidelity","qtt_vs_exact_fidelity","center_error_cn","center_error_exact"], rows)
    else
        write_csv(joinpath(outdir, "time_observables.csv"),
                  ["time","norm2","x_mean","y_mean","cn_diag_x_mean","cn_diag_y_mean",
                   "exact_spectral_x_mean","exact_spectral_y_mean","rms_radius","energy",
                   "qtt_vs_cn_relL2","cn_vs_exact_relL2","qtt_vs_exact_relL2",
                   "qtt_vs_cn_fidelity","qtt_vs_exact_fidelity","center_error_cn","center_error_exact"], rows)
    end

    truncation_pass = max_sweep_trunc <= truncation_target && !bond_cap_hit
    solver_pass = gres < global_tol
    physics_pass = norm_drift < 1e-6 && energy_drift < 1e-6
    reference_pass = qtt_cn_rmse < cn_reference_rmse_tol
    pass = solver_pass && truncation_pass && physics_pass && reference_pass

    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "SHO real-time space-time all-at-once")
        println(io, "dim = ", dim)
        println(io, "validation_pass = ", pass)
        println(io, "solver_pass = ", solver_pass)
        println(io, "truncation_pass = ", truncation_pass)
        println(io, "physics_conservation_pass = ", physics_pass)
        println(io, "cn_diagonal_reference_pass = ", reference_pass)
        println(io, "grid_shape = ", grid.shape)
        println(io, "domain_halfwidth = ", grid.halfwidth)
        println(io, "periodic_bc = true")
        println(io, "t_final = ", t_final)
        println(io, "dt = ", time.dt)
        println(io, "discrete_energy_gap = ", gap_info.gap)
        println(io, "discrete_period = ", gap_info.period)
        println(io, "steps_per_discrete_period = ", gap_info.period/time.dt)
        println(io, "displacement = ", dv)
        println(io, "global_relative_residual = ", gres)
        println(io, "absolute_residual_2norm = ", rdiag.absolute)
        println(io, "rhs_2norm = ", rdiag.rhs_norm)
        println(io, "used_sweeps = ", used)
        println(io, "solution_max_bond = ", max_dim(sol))
        println(io, "max_allowed_bond = ", max_bond)
        println(io, "bond_cap_hit = ", bond_cap_hit)
        println(io, "svd_cutoff_lambda = ", cutoff)
        println(io, "max_sweep_discarded_weight = ", max_sweep_trunc)
        println(io, "truncation_target = ", truncation_target)
        println(io, "max_norm_drift = ", norm_drift)
        println(io, "relative_energy_drift = ", energy_drift)
        println(io, "center_RMSE_vs_CN_diagonal_reference = ", center_rmse)
        println(io, "QTT_vs_CN_diagonal_RMSE_relL2 = ", qtt_cn_rmse)
        println(io, "QTT_vs_CN_diagonal_max_relL2 = ", qtt_cn_max)
        println(io, "minimum_QTT_vs_CN_fidelity = ", min_cn_fidelity)
        println(io, "CN_vs_exact_time_RMSE_relL2 = ", cn_exact_rmse)
        println(io, "CN_vs_exact_time_max_relL2 = ", cn_exact_max)
        println(io, "QTT_vs_exact_time_RMSE_relL2 = ", spectral_rmse)
        println(io, "QTT_vs_exact_time_max_relL2 = ", spectral_max)
        println(io, "minimum_QTT_vs_exact_time_fidelity = ", min_fidelity)
    end

    # Poster-friendly trajectory plot against a diagonalization reference.
    ts = Float64.(time.times)
    if dim == 1
        xm = [Float64(r[3]) for r in rows]
        xcn = [Float64(r[4]) for r in rows]
        xex = [Float64(r[5]) for r in rows]
        pcenter = plot(ts, xm; linewidth=2, label="QTT/MPS", xlabel="time", ylabel="<x>",
                       title="SHO center trajectory")
        plot!(pcenter, ts, xcn; linestyle=:dash, linewidth=2, label="CN diagonal reference")
        plot!(pcenter, ts, xex; linestyle=:dot, linewidth=2, label="exact-time spectral")
    else
        xm = [Float64(r[3]) for r in rows]
        ym = [Float64(r[4]) for r in rows]
        xcn = [Float64(r[5]) for r in rows]
        ycn = [Float64(r[6]) for r in rows]
        xex = [Float64(r[7]) for r in rows]
        yex = [Float64(r[8]) for r in rows]
        pcenter = plot(ts, xm; linewidth=2, label="<x> QTT", xlabel="time", ylabel="center",
                       title="2D SHO center trajectory")
        plot!(pcenter, ts, ym; linewidth=2, label="<y> QTT")
        plot!(pcenter, ts, xcn; linestyle=:dash, label="<x> CN diagonal")
        plot!(pcenter, ts, ycn; linestyle=:dash, label="<y> CN diagonal")
        plot!(pcenter, ts, xex; linestyle=:dot, label="<x> exact-time")
    end
    savefig(pcenter, joinpath(outdir, "center_trajectory.png"))

    perr = plot(ts, qtt_cn_errors; yscale=:log10, linewidth=2,
                xlabel="time", ylabel="phase-aligned relative L2 error",
                label="QTT vs CN diagonal", title="Solver and time-discretization error")
    plot!(perr, ts, cn_exact_errors; linewidth=2, label="CN vs exact-time spectral")
    plot!(perr, ts, qtt_exact_errors; linewidth=2, linestyle=:dash, label="QTT vs exact-time")
    savefig(perr, joinpath(outdir, "reference_state_errors.png"))

    sweeps = [Int(h[1]) for h in hist]
    gvals = [Float64(h[4]) for h in hist]
    tvals = [Float64(h[6]) for h in hist]
    pdiag = plot(sweeps, gvals; yscale=:log10, marker=:circle,
                 xlabel="DMRG sweep", ylabel="dimensionless error",
                 label="global residual", title="Solver convergence and SVD truncation")
    plot!(pdiag, sweeps, tvals; marker=:diamond, label="max discarded weight")
    hline!(pdiag, [global_tol]; linestyle=:dash, label="residual target")
    hline!(pdiag, [truncation_target]; linestyle=:dot, label="truncation target")
    savefig(pdiag, joinpath(outdir, "solver_diagnostics.png"))

    if make_movie
        if dim == 1
            y0 = spatial_density_physical(psi0A, grid)
            ymax = max(maximum(y0), maximum(spatial_density_physical(normalized_final_slice(PsiST), grid))) * 1.10
            anim = @animate for frame in 0:time.Nt
                if frame == 0
                    state = psi0A
                    tnow = 0.0
                else
                    state = Array(@view PsiST[frame, :])
                    tnow = time.times[frame]
                end
                plot(grid.coords[1], abs2.(state) ./ grid.cell_volume;
                     xlabel="x", ylabel="|psi|^2",
                     xlims=(-grid.halfwidth[1], grid.halfwidth[1]),
                     ylims=(0, ymax), linewidth=2, label="density",
                     title=@sprintf("1D SHO real time, t = %.3f", tnow))
            end
            mp4(anim, joinpath(outdir, "sho_realtime_1d.mp4"), fps=12)
        else
            allmax = maximum(abs2.(PsiST)) / grid.cell_volume
            allmax = max(allmax, maximum(abs2.(psi0A)) / grid.cell_volume)
            anim = @animate for frame in 0:time.Nt
                if frame == 0
                    state = psi0A
                    tnow = 0.0
                else
                    state = Array(@view PsiST[frame, :, :])
                    tnow = time.times[frame]
                end
                heatmap(grid.coords[1], grid.coords[2],
                        permutedims(abs2.(state) ./ grid.cell_volume, (2,1));
                        xlabel="x", ylabel="y", aspect_ratio=:equal,
                        clims=(0, allmax), colorbar_title="density",
                        title=@sprintf("2D SHO real time, t = %.3f", tnow))
            end
            mp4(anim, joinpath(outdir, "sho_realtime_2d.mp4"), fps=12)
        end
    end

    @printf("\nValidation: %s\n", pass ? "PASS" : "CHECK")
    @printf("global residual=%.6e (absolute ||Ax-b||2=%.6e, ||b||2=%.6e)\n",
            gres, rdiag.absolute, rdiag.rhs_norm)
    @printf("SVD max discarded weight=%.3e (target %.1e), chi=%d/%d, cap hit=%s\n",
            max_sweep_trunc, truncation_target, max_dim(sol), max_bond, string(bond_cap_hit))
    @printf("norm drift=%.3e, energy drift=%.3e\n", norm_drift, energy_drift)
    @printf("QTT vs CN-diagonal reference: RMSE=%.3e, max=%.3e, min fidelity=%.12f\n",
            qtt_cn_rmse, qtt_cn_max, min_cn_fidelity)
    @printf("CN time-discretization vs exact spectral: RMSE=%.3e, max=%.3e\n",
            cn_exact_rmse, cn_exact_max)
    @printf("QTT vs exact-time spectral: RMSE=%.3e, max=%.3e, min fidelity=%.12f\n",
            spectral_rmse, spectral_max, min_fidelity)
    @printf("Output: %s\n", outdir)

    return (pass=pass, solver_pass=solver_pass, truncation_pass=truncation_pass,
            physics_pass=physics_pass, reference_pass=reference_pass,
            global_residual=gres, absolute_residual=rdiag.absolute, rhs_norm=rdiag.rhs_norm,
            max_truncation=max_sweep_trunc, solution_max_bond=max_dim(sol),
            max_allowed_bond=max_bond, bond_cap_hit=bond_cap_hit,
            norm_drift=norm_drift, energy_drift=energy_drift,
            cn_reference_rmse=qtt_cn_rmse, cn_reference_max=qtt_cn_max,
            cn_exact_rmse=cn_exact_rmse, cn_exact_max=cn_exact_max,
            spectral_rmse=spectral_rmse, spectral_max=spectral_max,
            min_fidelity=min_fidelity, gap=gap_info.gap, period=gap_info.period,
            dt=time.dt, Nt=time.Nt, output=outdir)
end

# -----------------------------------------------------------------------------
# Nonlinear real-time GPE: CN + Picard all-at-once breathing-mode application
# -----------------------------------------------------------------------------

function build_realtime_gpe_cn_allatonce(H0::MPO, density_st_mpo::MPO,
                                         spatial_grid, time;
                                         g_discrete::Real,
                                         mpo_cutoff::Real=1e-10,
                                         mpo_max_bond::Int=256)
    T = ComplexF64
    Hc = mpo_to_type(H0, T)
    R = mpo_to_type(density_st_mpo, T)
    It = identity_mpo(time.grid.num_bits; T=T)
    Isp = spatial_identity(spatial_grid.grids; T=T)
    Ist = concat_mpo(It, Isp)
    Hst = concat_mpo(It, Hc)
    Sback = mpo_to_type(shift_backward_mpo(time.grid.num_bits; bc="dirichlet"), T)
    Sst = concat_mpo(Sback, Isp)
    SHst = concat_mpo(Sback, Hc)

    # Standard trapezoidal/CN Picard linearization:
    # [I + i dt/2 H(rho_j)] psi_j
    #   - [I - i dt/2 H(rho_{j-1})] psi_{j-1} = 0.
    # R is diagonal in full space-time; S*R produces rho_{j-1} psi_{j-1}.
    SR = mpo_product(Sst, R)
    SR = svd_compress_mpo(SR; max_dim=mpo_max_bond, cutoff=mpo_cutoff)

    A = add_mpo(Ist, scale_mpo(Hst, 0.5im * time.dt))
    A = add_mpo(A, scale_mpo(R, 0.5im * time.dt * g_discrete))
    A = add_mpo(A, scale_mpo(Sst, -1.0))
    A = add_mpo(A, scale_mpo(SHst, 0.5im * time.dt))
    A = add_mpo(A, scale_mpo(SR, 0.5im * time.dt * g_discrete))
    return A
end

function spacetime_rhs_gpe_cn(psi0::MPS, psi0A::AbstractArray, H0::MPO,
                              spatial_grid, time; g_discrete::Real,
                              cutoff::Real=1e-10, max_bond::Int=128)
    T = ComplexF64
    Isp = spatial_identity(spatial_grid.grids; T=T)
    rho0 = abs2.(ComplexF64.(psi0A) ./ norm(vec(psi0A)))
    rho0_mpo, _ = qtt_diagonal_from_array(real.(rho0), spatial_grid.grids;
                                           cutoff=cutoff, max_bond=max_bond)
    Hinit = add_mpo(mpo_to_type(H0, T), scale_mpo(mpo_to_type(rho0_mpo, T), g_discrete))
    Aminus0 = add_mpo(Isp, scale_mpo(Hinit, -0.5im * time.dt))
    rhs_space = exact_apply_mpo(Aminus0, mps_to_type(psi0, T))
    move_center!(rhs_space, 1)
    e1 = first_basis_mps(time.grid.num_bits; T=eltype(rhs_space))
    rhs = concat_mps(e1, rhs_space)
    move_center!(rhs, 1)
    return rhs
end

function load_saved_ground_state(path::AbstractString)
    isfile(path) || error("Ground-state file not found: $path")
    data = open(path, "r") do io
        deserialize(io)
    end
    return data
end

function run_gpe_realtime_breathing_allatonce(;
        ground_state_file::AbstractString,
        g1::Real=12.5,
        g2::Real=6.25,
        time_bits::Int=6,
        t_final::Real=2pi,
        a_kin::Real=0.5,
        b_trap::Real=0.5,
        relaxation::Real=0.5,
        max_picard::Int=30,
        picard_tol::Real=1e-5,
        max_inner_sweeps::Int=18,
        inner_global_tol::Real=2e-4,
        max_bond::Int=160,
        density_max_bond::Int=160,
        nonlinear_mpo_max_bond::Int=256,
        cutoff::Real=1e-10,
        density_cutoff::Real=1e-10,
        truncation_target::Real=1e-8,
        norm_drift_tol::Real=1e-4,
        energy_drift_tol::Real=1e-3,
        frequency_rel_tol::Real=0.10,
        make_movie::Bool=true,
        output_name::String="ST_07_GPE_realtime_2D_breathing")

    saved = load_saved_ground_state(ground_state_file)
    psi0A = ComplexF64.(saved["state_scaled"])
    ndims(psi0A) == 2 || error("Breathing runner requires a 2D saved ground state.")
    meta = saved["metadata"]
    if haskey(meta, "g") && abs(Float64(meta["g"]) - Float64(g1)) > 1e-8
        @warn "Saved ground state g does not match requested g1" saved_g=meta["g"] requested_g1=g1
    end

    # Rebuild the same spatial grid from serialized metadata.
    bits = Int.(meta["bits"])
    halfwidth = Float64.(meta["halfwidth"])
    grid = make_spatial_grid(2; bits=bits, halfwidth=halfwidth)
    size(psi0A) == Tuple(grid.shape) || error(
        "Saved state size $(size(psi0A)) does not match reconstructed grid $(grid.shape).")
    psi0A ./= norm(vec(psi0A))

    outdir = joinpath(PROJECT_ROOT, "validation_outputs", output_name)
    mkpath(outdir)
    time = make_time_grid(time_bits, t_final)
    H0r = build_spatial_harmonic(grid.grids; a_kin=a_kin, b_trap=b_trap, bc="periodic")
    H0 = mpo_to_type(H0r, ComplexF64)
    Hsparse = build_spatial_harmonic_sparse(grid; a_kin=a_kin, b_trap=b_trap)
    g_disc = Float64(g2) / grid.cell_volume

    psi0 = field_to_spatial_mps(psi0A, grid; cutoff=cutoff, max_bond=max_bond)
    rhs = spacetime_rhs_gpe_cn(psi0, psi0A, H0, grid, time;
                               g_discrete=g_disc, cutoff=density_cutoff,
                               max_bond=density_max_bond)
    guess = repeated_spacetime_guess(mps_to_type(psi0, ComplexF64), time_bits)

    st_grids = vcat([time.grid], grid.grids)
    hist = Tuple[]
    final_change = Inf
    final_inner = Inf
    final_density_bond = 1
    max_inner_trunc = 0.0
    bond_cap_hit = false
    converged = false

    @printf("2D GPE real-time breathing mode, QTT/MPS all-at-once\n")
    @printf("Initial state: g1=%.6g ground state; quench g1 -> g2 = %.6g -> %.6g\n",
            g1, g1, g2)
    @printf("grid=%dx%d, Nt=%d, t_final=%.6g, dt=%.6e, ordering=[time][x][y]\n",
            grid.shape[1], grid.shape[2], time.Nt, t_final, time.dt)
    @printf("g_discrete = g2/DeltaV = %.8e\n", g_disc)
    @printf("%7s %13s %13s %8s %8s %8s %12s\n",
            "Picard", "change", "inner_res", "sweeps", "chi", "chi_rho", "max_trunc")

    for p in 1:max_picard
        PsiGuess = spacetime_mps_to_array(guess, time.grid, grid)
        # Real-time GPE conserves norm.  Normalizing only the density used in the
        # Picard Hamiltonian prevents an intermediate solver norm error from
        # changing the physical interaction strength; the solved wavefunction
        # itself is NOT artificially renormalized.
        rho_st = normalize_spacetime_slices(PsiGuess)
        rho_mpo, rho_bond = qtt_diagonal_from_array(rho_st, st_grids;
                                                    cutoff=density_cutoff,
                                                    max_bond=density_max_bond)
        final_density_bond = rho_bond
        Ast = build_realtime_gpe_cn_allatonce(H0, rho_mpo, grid, time;
                                               g_discrete=g_disc,
                                               mpo_cutoff=density_cutoff,
                                               mpo_max_bond=nonlinear_mpo_max_bond)
        candidate, inner_hist, final_inner, used_sweeps = solve_linear_mps(
            Ast, rhs, guess;
            max_sweeps=max_inner_sweeps,
            global_tol=inner_global_tol,
            max_bond=max_bond,
            cutoff=cutoff,
            verbose=false)

        iter_max_trunc = history_max_truncation(inner_hist)
        max_inner_trunc = max(max_inner_trunc, iter_max_trunc)
        bond_cap_hit = bond_cap_hit || history_bond_cap_hit(inner_hist, max_bond)

        # The RHS fixes the global phase in real time, so do not phase-align before mixing.
        mixed = add_mps(scale_mps(guess, 1-relaxation),
                        scale_mps(candidate, relaxation))
        mixed = svd_compress_mps(mixed; max_dim=max_bond, cutoff=cutoff)
        move_center!(mixed, 1)
        final_change = relative_mps_change_raw(mixed, guess)
        guess = mixed

        push!(hist, (p, final_change, final_inner, used_sweeps,
                     max_dim(guess), rho_bond, iter_max_trunc))
        @printf("%7d %13.4e %13.4e %8d %8d %8d %12.3e\n",
                p, final_change, final_inner, used_sweeps,
                max_dim(guess), rho_bond, iter_max_trunc)

        if final_change < picard_tol && final_inner < inner_global_tol
            converged = true
            break
        end
    end

    PsiST = spacetime_mps_to_array(guess, time.grid, grid)

    # Include the initial state at t=0 in the observables.
    times_all = vcat([0.0], Float64.(time.times))
    widths = Float64[]
    norms = Float64[]
    energies = Float64[]
    boundaries = Float64[]

    for j in 0:time.Nt
        state = j == 0 ? psi0A : Array(@view PsiST[j, :, :])
        n2 = sum(abs2, state)
        obs = coordinate_observables(state, grid)
        E = gpe_energy_dense_scaled(state, Hsparse, grid, g2)
        bd = boundary_density_physical(state ./ max(norm(vec(state)), eps(Float64)), grid)
        push!(widths, obs.rms_radius)
        push!(norms, n2)
        push!(energies, E)
        push!(boundaries, bd)
    end

    norm_drift = maximum(abs.(norms .- norms[1]))
    energy_drift = maximum(abs.(energies .- energies[1])) /
                   max(abs(energies[1]), eps(Float64))
    max_boundary = maximum(boundaries)

    omega_trap = 2 * sqrt(Float64(a_kin) * Float64(b_trap))
    omega_expected = 2 * omega_trap
    omega_measured = dominant_angular_frequency(times_all, widths)
    freq_rel_error = abs(omega_measured - omega_expected) / omega_expected

    # Fit only amplitude and phase at the theoretically expected breathing frequency.
    X = hcat(ones(length(times_all)), cos.(omega_expected .* times_all),
             sin.(omega_expected .* times_all))
    coeff = X \ widths
    width_expected_frequency_fit = X * coeff
    width_fit_rmse = sqrt(mean((widths .- width_expected_frequency_fit).^2))

    write_csv(joinpath(outdir, "picard_history.csv"),
              ["picard","relative_change","inner_global_residual","inner_sweeps",
               "solution_max_bond","density_max_bond","max_inner_truncation"], hist)
    obsrows = [(times_all[j], norms[j], widths[j], energies[j], boundaries[j],
                width_expected_frequency_fit[j]) for j in eachindex(times_all)]
    write_csv(joinpath(outdir, "breathing_observables.csv"),
              ["time","norm2","rms_radius","gpe_energy_g2","boundary_density",
               "fit_at_expected_omega_2omega"], obsrows)

    truncation_pass = max_inner_trunc <= truncation_target && !bond_cap_hit
    physics_pass = norm_drift < norm_drift_tol && energy_drift < energy_drift_tol &&
                   max_boundary < 1e-5
    frequency_pass = isfinite(omega_measured) && freq_rel_error < frequency_rel_tol
    pass = converged && final_inner < inner_global_tol && truncation_pass &&
           physics_pass && frequency_pass

    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "2D GPE breathing-mode space-time all-at-once")
        println(io, "validation_pass = ", pass)
        println(io, "picard_converged = ", converged)
        println(io, "g1 = ", g1)
        println(io, "g2 = ", g2)
        println(io, "g_discrete = ", g_disc)
        println(io, "grid_shape = ", grid.shape)
        println(io, "t_final = ", t_final)
        println(io, "Nt = ", time.Nt)
        println(io, "dt = ", time.dt)
        println(io, "final_picard_change = ", final_change)
        println(io, "final_inner_global_residual = ", final_inner)
        println(io, "solution_max_bond = ", max_dim(guess))
        println(io, "max_allowed_solution_bond = ", max_bond)
        println(io, "density_max_bond = ", final_density_bond)
        println(io, "bond_cap_hit = ", bond_cap_hit)
        println(io, "svd_cutoff_lambda = ", cutoff)
        println(io, "max_inner_sweep_discarded_weight = ", max_inner_trunc)
        println(io, "truncation_target = ", truncation_target)
        println(io, "norm_drift = ", norm_drift)
        println(io, "relative_energy_drift = ", energy_drift)
        println(io, "max_boundary_density = ", max_boundary)
        println(io, "expected_breathing_omega = ", omega_expected)
        println(io, "measured_breathing_omega = ", omega_measured)
        println(io, "frequency_relative_error = ", freq_rel_error)
        println(io, "expected_frequency_fit_RMSE = ", width_fit_rmse)
    end

    pwidth = plot(times_all, widths; linewidth=2, marker=:circle,
                  xlabel="time", ylabel="sqrt(<x^2+y^2>)",
                  label="QTT/MPS", title="2D GPE breathing mode")
    plot!(pwidth, times_all, width_expected_frequency_fit;
          linestyle=:dash, linewidth=2, label="fit at expected omega = 2")
    savefig(pwidth, joinpath(outdir, "breathing_width.png"))

    ppic = plot([Int(r[1]) for r in hist], [Float64(r[2]) for r in hist];
                yscale=:log10, marker=:circle, xlabel="Picard iteration",
                ylabel="relative change", label="Picard change",
                title="Nonlinear Picard convergence")
    hline!(ppic, [picard_tol]; linestyle=:dash, label="target")
    savefig(ppic, joinpath(outdir, "picard_convergence.png"))

    # Plot four equally spaced density snapshots.
    snap_ids = unique(round.(Int, range(0, time.Nt, length=4)))
    ps = Any[]
    allmax = max(maximum(abs2.(psi0A)) / grid.cell_volume,
                 maximum(abs2.(PsiST)) / grid.cell_volume)
    for j in snap_ids
        state = j == 0 ? psi0A : Array(@view PsiST[j, :, :])
        tnow = j == 0 ? 0.0 : time.times[j]
        push!(ps, heatmap(grid.coords[1], grid.coords[2],
                          permutedims(abs2.(state) ./ grid.cell_volume, (2,1));
                          aspect_ratio=:equal, xlabel="x", ylabel="y",
                          clims=(0, allmax), colorbar=false,
                          title=@sprintf("t=%.2f", tnow)))
    end
    psnap = plot(ps...; layout=(1, length(ps)), size=(1200, 300))
    savefig(psnap, joinpath(outdir, "breathing_snapshots.png"))

    if make_movie
        anim = @animate for frame in 0:time.Nt
            if frame == 0
                state = psi0A
                tnow = 0.0
            else
                state = Array(@view PsiST[frame, :, :])
                tnow = time.times[frame]
            end
            heatmap(grid.coords[1], grid.coords[2],
                    permutedims(abs2.(state) ./ grid.cell_volume, (2,1));
                    xlabel="x", ylabel="y", aspect_ratio=:equal,
                    clims=(0, allmax), colorbar_title="density",
                    title=@sprintf("2D GPE breathing, t = %.3f", tnow))
        end
        mp4(anim, joinpath(outdir, "gpe_breathing_2d.mp4"), fps=12)
    end

    @printf("\nValidation: %s\n", pass ? "PASS" : "CHECK")
    @printf("Picard converged=%s, change=%.3e, inner residual=%.3e\n",
            string(converged), final_change, final_inner)
    @printf("SVD max discarded weight=%.3e (target %.1e), chi=%d/%d, cap hit=%s\n",
            max_inner_trunc, truncation_target, max_dim(guess), max_bond,
            string(bond_cap_hit))
    @printf("norm drift=%.3e, energy drift=%.3e, max boundary density=%.3e\n",
            norm_drift, energy_drift, max_boundary)
    @printf("breathing omega: expected=%.6f, measured=%.6f, rel.error=%.3e\n",
            omega_expected, omega_measured, freq_rel_error)
    @printf("Poster figure: %s\n", joinpath(outdir, "breathing_width.png"))
    @printf("Output: %s\n", outdir)

    return (pass=pass, converged=converged, final_change=final_change,
            inner_residual=final_inner, max_truncation=max_inner_trunc,
            solution_max_bond=max_dim(guess), density_max_bond=final_density_bond,
            bond_cap_hit=bond_cap_hit, norm_drift=norm_drift,
            energy_drift=energy_drift, expected_omega=omega_expected,
            measured_omega=omega_measured, frequency_rel_error=freq_rel_error,
            width_fit_rmse=width_fit_rmse, output=outdir)
end

