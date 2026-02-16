# Linearised DF-ARD convergence test
module linearised_DF_ARD_ConvergenceTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test

  const write_solution_to_vtk = false

  # Arbitrary model parameters
  const κ = 1.0e-0
  const ν = 1
  const ρ0 = 1
  const r = 4
  const FF = 1.0E0
  const α0 = 1.0E-0
  const α1 = 1.0E-0
  const α = 1

  grav = VectorValue(0.0,1.0)

  function transport(ϕ)
    grav*(α0+α1*ϕ)
  end

  u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
                          -sin(π*x[1])*cos(π*x[2]))
  p_ex(x)  = sin(π*x[1])*sin(π*x[2])
  ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2])

  f_ex(x) = κF(x)*u_ex(x) + ∇(p_ex)(x) - (transport∘ϕ_ex)(x)#+ FF*u_ex(x)*(normu∘u_ex)(x) 
  g_ex(x) = (∇⋅u_ex)(x)
  ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
  m_ex(x) =  (1.0-g_ex(x))*ϕ_ex(x) - (∇⋅ζ_ex)(x)

κF(x) = ((1.0/κ) + FF*((normu(u_ex(x)))^(r-2)))
κinv(x) = 1.0 / κF(x)
 function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end
  function solve_df_ard(op; kwargs...)
    solve(op)
  end

  for k in 0:0
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    eζ = Float64[]
    eϕ = Float64[]
    rζ = Float64[]
    rϕ = Float64[]
    hh = Float64[]
    nn = Int[]

    push!(ru,0.)
    push!(rp,0.)
    push!(rζ,0.)
    push!(rϕ,0.)

    nkmax=6
    for nk in 1:nkmax
        println("******** Refinement step: $nk")
       model=generate_model2d(nk)
       setup_model_labels_unit_square!(model)
        op,dΩ=assemble_df_ard(model, k, κ, κF, FF, f_ex, g_ex, m_ex, u_ex, p_ex, ζ_ex, ϕ_ex)
        xh=solve_df_ard(op)
        uh, ph, ζh, ϕh = xh
        ndofs=num_free_dofs(Gridap.FESpaces.get_fe_space(xh))

        if (write_solution_to_vtk)
          Ω = Triangulation(model)
          writevtk(Ω, "convergence_results_$nk",
                  cellfields=["uh"=>uh, "ph"=>ph, "ζh"=>ζh, "ϕh"=>ϕh])
        end

        error_u,error_p,error_ζ,error_ϕ = compute_errors_df_ard(xh, dΩ, κ, u_ex, p_ex, ζ_ex, ϕ_ex)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        push!(eζ,error_ζ)
        push!(eϕ,error_ϕ)
        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rζ, log(eζ[nk]/eζ[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rϕ, log(eϕ[nk]/eϕ[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) &   e(ζ)   &  r(ζ) &   e(ϕ)   &  r(ϕ)")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk], eζ[nk], rζ[nk], eϕ[nk], rϕ[nk]);
    end
    println("=======================================================")
  end
end