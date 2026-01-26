module DarcyForchheimerConvergenceTests_mixedBC
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra

  
  const κ = 1.0E-0 
  const r = 4.0 
  const FF = 1.0E0  
  const α = 1.0E0
  const ρ0 = 1.0E0

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v
  end

  u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
                          -sin(π*x[1])*cos(π*x[2]))
  p_ex(x)  = sin(π*x[1])*cos(π*x[2])
  #ζ_ex(x) = VectorValue(sin(2π*x[1]), cos(2π*x[2]))
  ϕ_ex(x) = cos(2π*x[1])*sin(2π*x[2])


  f_ex(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*(normu∘u_ex)(x) 
  g_ex(x) = (∇⋅u_ex)(x)
  #h_ex(x) = (1.0/α)*ζ_ex(x) - ∇(ϕ_ex)(x) + u_ex(x) * (1.0/α)*ϕ_ex(x)
  ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
  m_ex(x) =  (ρ0+g_ex(x))*ϕ_ex(x) - (∇⋅ζ_ex)(x) 

  function solve_Forchheimer(model; k=k, generate_output=false)
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)
    reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
    reffe_ϕ = ReferenceFE(lagrangian,Float64,k)

    # FESpaces
    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="Gamma_u",conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
    Wh_ = TestFESpace(model,reffe_ζ,dirichlet_tags="Gamma_p",conformity=:HDiv)
    Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)


    Uh = TrialFESpace(Uh_,u_ex)
    Qh = TrialFESpace(Qh_)
    Wh = TrialFESpace(Wh_,ζ_ex)
    Ψh = TrialFESpace(Ψh_)

    Yh = MultiFieldFESpace([Uh_,Qh_,Wh_,Ψh_])
    Xh = MultiFieldFESpace([Uh,Qh,Wh,Ψh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))
    Γp = BoundaryTriangulation(model, tags = "Gamma_p")
    n_Γp = get_normal_vector(Γp)
    dΓp = Measure(Γp,2*(k+2)-1)
    Γu = BoundaryTriangulation(model, tags = "Gamma_u")
    dΓu = Measure(Γu,2*(k+2)-1)
    n_Γu = get_normal_vector(Γu)

    a1(u,v)   = ∫( (1.0/κ)*(u⋅v))*dΩ
    c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
    b1(v,q)   = ∫(-(∇⋅v)*q)*dΩ
    F(v)     = ∫(f_ex⋅v)*dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫(g_ex*q)dΩ
    
    a2(ζ,τ) = ∫( (1.0/α)*(ζ⋅τ))*dΩ
    b2(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
    c2(ϕ,ψ) = ∫(g_ex*(ϕ * ψ))*dΩ
    e(u,ϕ,τ) = ∫((1.0/α)* (u⋅τ) * ϕ)*dΩ  
    M(τ) = ∫(ϕ_ex*(τ⋅n_Γu))dΓu
    H(ψ) = ∫(m_ex*ψ)dΩ

    resid((u,p,ζ,ϕ),(v,q,τ,ψ)) = a1(u,v) + c(u,u,v) + b1(v,p) +
                                 b1(u,q) - F(v) + G(q) + 
                                 a2(ζ,τ) + b2(τ,ϕ) + e(u,ϕ,τ) + 
                                 b2(ζ,ψ) - c2(ϕ,ψ) - M(τ) + H(ψ)

    jacob((u,p,ζ,ϕ),(du,dp,dζ,dϕ),(v,q,τ,ψ)) = a1(du,v) + d1(u,du,v) + d2(u,du,v) + b1(v,dp) + 
                                               b1(du,q) + 
                                               a2(dζ,τ) + b2(τ,dϕ) + e(u,dϕ,τ) + e(du,ϕ,τ) + 
                                               b2(dζ,ψ) - c2(dϕ,ψ) 

    oper = FEOperator(resid,jacob,Xh,Yh)
    nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking(), iterations=25, ftol=1.0E-7)
    solver = FESolver(nls)
    x0=0.01*ones(num_free_dofs(Xh)) 
    xh=FEFunction(Xh,x0)
    (uh,ph,ζh,ϕh), _ = solve!(xh,solver,oper)


    pa(p,q) = ∫(p*q)*dΩ
    pdivl(q) = ∫(q*(∇⋅uh - g_ex))*dΩ
    projection = AffineFEOperator(pa,pdivl,Qh,Qh)
    loss_div = solve(projection)
    loss_divh=norm(get_free_dof_values(loss_div),Inf)

    euh = u_ex - uh
    eph = p_ex - ph
    eζh = ζ_ex - ζh
    eϕh = ϕ_ex - ϕh
    
    eu = (sum(∫((euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
    ep = sqrt(sum(∫(eph*eph)*dΩ))
    eζ =  (sum(∫(((∇⋅eζh)*(∇⋅eζh)).^(0.5*6.0/5.0))dΩ))^(5.0/6.0) + sqrt(sum(∫(eζh⋅eζh)dΩ))
    eϕ = (sum(∫((eϕh*eϕh).^(0.5*6.0))dΩ))^(1.0/6.0)

    if generate_output 
      writevtk(Ω,"DFconvergence_ncells=$(num_cells(model))",order=1,cellfields=["uh"=>uh,"ph"=>ph,"ζh"=>ζh,"ϕh"=>ϕh])
   end

   eu, ep, eζ, eϕ, loss_divh, Gridap.FESpaces.num_free_dofs(Xh)

  end

  function  convergence_test(; nkmax, k=k, generate_output=false)
    eu = Float64[]
    ep = Float64[]
    ru = Float64[]
    rp = Float64[]
    hh = Float64[]
    eζ = Float64[]
    eϕ = Float64[]
    rζ = Float64[]
    rϕ = Float64[]
    losses_divh = Float64[]
    nn = Int[]
    newton_iters = Int[]

    push!(ru,0.)
    push!(rp,0.)
    push!(rζ,0.)
    push!(rϕ,0.) 

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_model2d(nk)
        setup_model_labels_unit_square!(model)
        error_u,error_p,error_ζ,error_ϕ,loss_divh,ndofs=solve_Forchheimer(model; k=k, generate_output=generate_output)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eu,error_u)
        push!(ep,error_p)
        push!(eζ,error_ζ)
        push!(eϕ,error_ϕ)
        push!(losses_divh,loss_divh)

        if nk>1
            push!(ru, log(eu[nk]/eu[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rp, log(ep[nk]/ep[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rζ, log(eζ[nk]/eζ[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rϕ, log(eϕ[nk]/eϕ[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=========================================================================================================")
    println("   DoF  &    h   &   e(u)   &  r(u) &   e(p)   &  r(p) &   e(ζ)   &  r(ζ) &   e(ϕ)   &  r(ϕ) & loss_divh ")
    println("=========================================================================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %.2e & %.3f & %.2e\n", 
                nn[nk], hh[nk], eu[nk], ru[nk], ep[nk], rp[nk], eζ[nk], rζ[nk], eϕ[nk], rϕ[nk], losses_divh[nk]);
    end
    println("=========================================================================================================")
  end
  convergence_test(;nkmax=6,k=0,generate_output=false)
end