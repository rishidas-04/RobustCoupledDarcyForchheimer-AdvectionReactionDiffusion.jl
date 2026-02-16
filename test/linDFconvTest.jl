# Linearised DF convergence test
module DarcyConvergenceTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test

  const write_solution_to_vtk = false

  # Arbitrary model parameters
  const κ = 1.0e-0
  const ν = 1
  const c0 = 1
  const r = 4
  const FF = 1.0E0

u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
                       -sin(π*x[1])*cos(π*x[2]))

p_ex(x)  = sin(π*x[1])*cos(π*x[2])

f_rhs(x) = (κF(x))*u_ex(x) + ∇(p_ex)(x) #+ FF*u_ex(x)*((normu∘u_ex)(x))^(r-2)
g_rhs(x) = (∇⋅u_ex)(x)

κF(x) = ((1.0/κ) + FF*((normu(u_ex(x)))^(r-2)))
κinv(x) = 1.0 / κF(x)
 function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end
  function solve_darcy(op; kwargs...)
    solve(op)
  end

  for k in 0:0
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    hh = Float64[]
    nn = Int[]

    push!(ru,0.)
    push!(rp,0.)

    nkmax=6
    for nk in 1:nkmax
        println("******** Refinement step: $nk")
       model=generate_model2d(nk)
       setup_model_labels_unit_square!(model)
        op,dΩ=assemble_darcy(model, k, κ, κF, FF, f_rhs, g_rhs, u_ex, p_ex)
        xh=solve_darcy(op)
        uh, ph = xh
        ndofs=num_free_dofs(Gridap.FESpaces.get_fe_space(xh))

        if (write_solution_to_vtk)
          Ω = Triangulation(model)
          writevtk(Ω, "convergence_results_$nk",
                  cellfields=["uh"=>uh, "ph"=>ph])
        end

        error_u,error_p =
          compute_errors_darcy(xh, dΩ, κ, u_ex, p_ex)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk]);
    end
    println("=======================================================")
  end
end