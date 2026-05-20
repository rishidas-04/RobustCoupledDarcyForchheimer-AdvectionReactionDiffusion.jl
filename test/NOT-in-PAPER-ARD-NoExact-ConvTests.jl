# advection-reaction-diffusion convergence test with no exact solution.

module DarcyForchheimerConvergenceTests_NoExactSolution
  using GridapRobustDarcyForchheimer
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra
  using Gridap.CellData
  using Krylov
  using LinearOperators

  const α = 1.0e-06
  const ρ0 = 1.0e-03

  m_ex = 1.0
  ζin(x) = VectorValue(5*(1-x[1])*(x[2]),2.5*(1-x[1]))
  ϕout  = 1.0
  u_ex(x) = VectorValue(0.0000001,0.000001)

function solve_ARDNoExact(model; k=k, generate_output=false)
reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
reffe_ϕ = ReferenceFE(lagrangian,Float64,k)

Wh_ =  TestFESpace(model,reffe_ζ,dirichlet_tags=["Gamma_in","Gamma_topbot"],conformity=:HDiv)
Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)

Wh = TrialFESpace(Wh_,[ζin,ζin])
Ψh = TrialFESpace(Ψh_)

Yh = MultiFieldFESpace([Wh_,Ψh_])
Xh = MultiFieldFESpace([Wh,Ψh])

Ω = Triangulation(model)
dΩ = Measure(Ω,2*(k+2))
Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ,2*(k+2)-1)
h_e = CellField(get_array(∫(1)dΛ), Λ)
n_Λ = get_normal_vector(Λ)

Γout = BoundaryTriangulation(model, tags = "Gamma_out")
n_Γout = get_normal_vector(Γout)
dΓout = Measure(Γout,2*(k+2)-1)
h_e_Γout = CellField(get_array(∫(1)dΓout), Γout)

a(ζ,τ) = ∫((1.0/α)*(ζ⋅τ))*dΩ
b(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
c(ϕ,ψ) = ∫((ρ0)*ϕ*ψ)*dΩ
e(ϕ,τ) = ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ
H(τ) = ∫(ϕout*(τ⋅n_Γout))*dΓout #+ ∫(h_ex⋅τ)*dΩ
M(ψ) = ∫(-(m_ex*ψ))*dΩ

lhs((ζ, ϕ),(τ, ψ)) = a(ζ,τ) + b(τ,ϕ) + e(ϕ,τ) +
                         b(ζ,ψ) - c(ϕ,ψ)  
  
rhs((τ, ψ)) = H(τ) + M(ψ)

op = AffineFEOperator(lhs,rhs,Xh,Yh)
ζh, ϕh = solve(op)


    # if generate_output 
    #   writevtk(Ω,"DFNoExactExtreme_ncells=$(num_cells(model))",order=1,cellfields=["uh"=>uh,"ph"=>ph, "κ"=>κ])
    # end

   
   ζh, ϕh, dΩ, dΛ, dΓout, h_e, h_e_Γout, Wh, Ψh, Ψh_, Gridap.FESpaces.num_free_dofs(Xh)
  end

  function compute_errorNoExact_simpleNorms(eζh, eϕh, dΩ)
    eζ =  (sum(∫(((∇⋅eζh)*(∇⋅eζh)).^(0.5*6.0/5.0))dΩ))^(5.0/6.0) + sqrt(sum(∫(eζh⋅eζh)dΩ))
    eϕ = (sum(∫((eϕh*eϕh).^(0.5*6.0))dΩ))^(1.0/6.0)

    eζ, eϕ
  end

  function compute_errorNoExact_preconditionerNorms(eζh, eϕh, pref,Ψh, Ψh_, dΩ, dΛ, dΓout, h_e, h_e_Γout; k=0)

    app1(ϕ,ψ) = ∫(ϕ*ψ)dΩ

    app2(ϕ,ψ) = ∫(α*(∇(ϕ))⋅(∇(ψ)))dΩ +
                ∫(α/h_e*jump(ϕ)*jump(ψ))dΛ +
                ∫(α/h_e_Γout*ϕ*ψ)dΓout 
         
  
    A22_m1, A22_m2 = assemble_matrix(app1,Ψh,Ψh_), assemble_matrix(app2,Ψh,Ψh_)

    A22lo = LinearOperator(GridapLinearSolverPreconditioner(A22_m1)) +
            LinearOperator(GridapLinearSolverPreconditioner(A22_m2))       
    
    A22inv1 = inv(Array(A22_m1))
    A22inv2 = inv(Array(A22_m2))
    
    A22lo_inv = A22inv1 + A22inv2 
    eϕh_int = interpolate_everywhere(eϕh, Ψh)
    eϕh_int_vec = get_free_dof_values(eϕh_int)

    A = A22lo_inv  
    b = eϕh_int_vec
    r = similar(b)

    rtol=1.0e-14
    minres_callback(solver) = custom_stopping_condition(solver, A, b, r, rtol)
    (newfreedofs, stats) = minres(A, b, M = A22lo; itmax=2000, verbose=0,
                                 callback=minres_callback,
                                 atol=0.0,
                                 rtol=rtol)
  
    eζ = sqrt(sum(∫((1.0/α)*(eζh⋅eζh))dΩ)) + sqrt(sum(∫((∇⋅eζh)*(∇⋅eζh))*dΩ))
    eϕ = sqrt(dot(newfreedofs, b)) 
    eζ, eϕ
  end

  function  convergence_test(; nkmax, k=k, generate_output=false)
    eζ = Float64[]
    eϕ = Float64[]
    rζ = Float64[]
    rϕ = Float64[]
    eζpre = Float64[]
    eϕpre = Float64[]
    rζpre = Float64[]
    rϕpre = Float64[]
    hh = Float64[]
    # losses_divh = Float64[]
    nn = Int[]
    push!(rζ,0.)
    push!(rϕ,0.)
    push!(rζpre,0.)
    push!(rϕpre,0.)

    modelfine = generate_rectangle(8)
    setup_model_labels_rectangle_channel!(modelfine)
    ζref,ϕref,_,_,_,_,_,_ = solve_ARDNoExact(modelfine; k=k, generate_output=false)

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_rectangle(nk-1)
        setup_model_labels_rectangle_channel!(model)
        ζh,ϕh, dΩ, dΛ, dΓout,  h_e, h_e_Γout, Wh,Ψh, Ψh_, ndofs=solve_ARDNoExact(model; k=k, generate_output=generate_output)
        ζr_ = Interpolable(ζref)
        ϕr_ = Interpolable(ϕref)  
        ζr = interpolate_everywhere(ζr_, Wh)
        ϕr = interpolate_everywhere(ϕr_, Ψh)

        error_ζ,error_ϕ = compute_errorNoExact_simpleNorms(ζr - ζh,ϕr-ϕh,dΩ)
        error_ζpre,error_ϕpre = compute_errorNoExact_preconditionerNorms(ζr-ζh,ϕr-ϕh,ϕr,Ψh,Ψh_,dΩ,dΛ,dΓout,h_e,h_e_Γout;k=k)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eζ,error_ζ)
        push!(eϕ,error_ϕ)
        push!(eζpre,error_ζpre)
        push!(eϕpre,error_ϕpre)
        # push!(losses_divh,loss_divh)

        if nk>1
            push!(rζ, log(eζ[nk]/eζ[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rϕ, log(eϕ[nk]/eϕ[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rζpre, log(eζpre[nk]/eζpre[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rϕpre, log(eϕpre[nk]/eϕpre[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(ζ)   &  r(ζ) &   e(ϕ)   &  r(ϕ) ")
    println("===================== usual norms =====================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f\n", 
                nn[nk], hh[nk], eζ[nk], rζ[nk], eϕ[nk], rϕ[nk]);
    end
    
    println("=================== p with preconditioning ============")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f \n", 
                nn[nk], hh[nk], eζpre[nk], rζpre[nk], eϕpre[nk], rϕpre[nk]);
    end
    println("=======================================================")

  end
  convergence_test(;nkmax=6,k=0,generate_output=false)
end