include(joinpath(@__DIR__, "SpaceTimeQTT_AllAtOnce_Core.jl"))

# Accept a saved g=12.5 ground state, with compatibility for existing output folders.
length(ARGS) <= 1 || error("Usage: julia --project=. 07_GPE_realtime_2D_breathing_poster.jl [ground_state_state.jls]")
candidates = isempty(ARGS) ? [
    joinpath(PROJECT_ROOT, "inputs", "ground_state_state.jls"),
    joinpath(PROJECT_ROOT, "validation_outputs", "POSTER_04_GPE_imaginary_2D_g12p5", "ground_state_state.jls"),
] : [abspath(ARGS[1])]

function first_existing_ground_state(paths)
    for f in paths
        isfile(f) && return f
    end
    return nothing
end

gsfile = first_existing_ground_state(candidates)
if gsfile === nothing
    println("Searched for the initial ground state in:")
    foreach(f -> println("  ", f), candidates)
    error("Ground-state input is missing. Supply its path as the command-line argument or place it in inputs/ground_state_state.jls. See README.md.")
end
println("Using ground state: ", gsfile)

# Interaction quench g1=12.5 -> g2=6.25 in oscillator units (trap frequency=1).
# The expected breathing angular frequency is 2; t_final=4pi spans four periods.
run_gpe_realtime_breathing_allatonce(
    ground_state_file=gsfile,
    g1=12.5,
    g2=6.25,
    time_bits=8,          # Nt = 256
    t_final=4pi,          # four expected breathing periods; dt ~= 0.0491
    relaxation=0.5,
    max_picard=40,
    picard_tol=1e-5,
    max_inner_sweeps=24,
    inner_global_tol=1e-4,
    max_bond=256,
    density_max_bond=256,
    nonlinear_mpo_max_bond=384,
    cutoff=1e-14,
    density_cutoff=1e-12,
    truncation_target=1e-8,
    norm_drift_tol=1e-4,
    energy_drift_tol=1e-3,
    frequency_rel_tol=0.05,
    make_movie=false,
    output_name=get(ENV, "QTT_OUTPUT_NAME", "RUN_07_GPE_realtime_2D_breathing_4period"),
)
