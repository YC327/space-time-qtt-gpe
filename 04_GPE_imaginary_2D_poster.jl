include(joinpath(@__DIR__, "SpaceTimeQTT_AllAtOnce_Core.jl"))

# Main nonlinear ground-state panel for the poster.
# The already-tested 32x32 spatial grid is used to obtain a robust validated
# g1=12.5 ground state for the subsequent real-time breathing calculation.
run_gpe_imaginary_allatonce(
    dim=2,
    bits=[5,5],
    halfwidth=[6.0,6.0],
    time_bits=5,          # Nt = 32
    tau_final=4.0,        # dtau = 0.125
    g_phys=12.5,
    relaxation=0.5,
    max_picard=32,
    picard_tol=1e-5,
    max_inner_sweeps=18,
    inner_global_tol=1e-4,
    max_bond=256,
    density_max_bond=256,
    cutoff=1e-14,
    density_cutoff=1e-12,
    truncation_target=1e-8,
    output_name="POSTER_04_GPE_imaginary_2D_g12p5",
)
