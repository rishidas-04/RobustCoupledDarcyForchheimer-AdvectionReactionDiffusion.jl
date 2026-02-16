module PertubedSaddlePointPreconditioners

    using Gridap
    using Gridap.ReferenceFEs
    using Gridap.FESpaces
    using LinearOperators
    using LinearAlgebra
    using Krylov
    using Printf

    include("Preconditioners.jl")
    include("DarcyTools.jl")

    export assemble_perturbed_darcy, assemble_darcy, assemble_darcy_noExact, compute_errors_darcy, compute_errors_perturbed_darcy, compute_operator_norm, compute_errors_robust_darcy,  assemble_linear_elasticity, compute_errors_linear_elasticity
    export generate_model2d, generate_rectangle, setup_model_labels_rectangle_channel!,setup_model_labels_unit_square!, computation_error_darcy_minimization
    export GridapLinearSolverPreconditioner
    export assemble_darcy_forchheimer_noExact, assemble_darcy_forchheimer_riesz_mapping_preconditioner_blocks, compute_errors_robust_darcy_forchheimer
    export assemble_df_ard_noExact, assemble_darcy, compute_errors_darcy
    export assemble_df_ard, compute_errors_df_ard
end
