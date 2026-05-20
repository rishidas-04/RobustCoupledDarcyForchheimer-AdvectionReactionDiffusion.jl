module DarcyForchheimerARDCavity
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches 
  using LinearAlgebra
  
  const κ = 1.0 
  const r = 3.0 
  const FF = 20.0  
  const α = 1.0

  const ϕ_top = 20.0  

  const α1 = 10.0

  grav = VectorValue(1.0,1.0)

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))
  end

  function deri(v) #derivative of |v| wrt v
    (v⋅v>0)*(sqrt(v⋅v))^(-1)*v
  end

  function setup_model_labels_2d_cavity!(model)
    labels = get_face_labeling(model)
    add_tag!(labels,"Top",[6]) #6top
    add_tag!(labels,"Rest",[1,2,3,4,5,7,8]) # corners + 5bottom 7left 8right
  end 

  k = 0; nk = 7
  model=generate_model2d(nk)
  setup_model_labels_2d_cavity!(model)

  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)
  reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
  reffe_ϕ = ReferenceFE(lagrangian,Float64,k)

  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="boundary",conformity=:HDiv)
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  Wh_ = TestFESpace(model,reffe_ζ,conformity=:HDiv)
  Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)


  Uh = TrialFESpace(Uh_,VectorValue(0,0))
  Qh = TrialFESpace(Qh_)
  Wh = TrialFESpace(Wh_)
  Ψh = TrialFESpace(Ψh_)

  Yh = MultiFieldFESpace([Uh_,Qh_,Wh_,Ψh_])
  Xh = MultiFieldFESpace([Uh,Qh,Wh,Ψh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γtop = BoundaryTriangulation(model, tags = "Top")
  n_Γtop = get_normal_vector(Γtop)
  dΓtop = Measure(Γtop,2*(k+2)-1)

  a1(u,v)   = ∫( (1.0/κ)*(u⋅v))*dΩ
  c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
  d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
  d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
  b1(v,q)   = ∫(-(∇⋅v)*q)*dΩ
  F(v,ϕ)  = ∫(α1*ϕ*(grav⋅v))*dΩ
  
  a2(ζ,τ) = ∫( (1.0/α)*(ζ⋅τ))*dΩ
  b2(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
  e(u,ϕ,τ) = ∫((1.0/α)* (u⋅τ) * ϕ)*dΩ  
  M(τ) = ∫(ϕ_top*(τ⋅n_Γtop))dΓtop

  resid((u,p,ζ,ϕ),(v,q,τ,ψ)) = a1(u,v) + c(u,u,v) + b1(v,p) +
                                b1(u,q) - F(v,ϕ) + 
                                a2(ζ,τ) + b2(τ,ϕ) + e(u,ϕ,τ) + 
                                b2(ζ,ψ) - M(τ)

  jacob((u,p,ζ,ϕ),(du,dp,dζ,dϕ),(v,q,τ,ψ)) = a1(du,v) + d1(u,du,v) + d2(u,du,v) + b1(v,dp) -F(v,dϕ) + 
                                              b1(du,q) + 
                                              a2(dζ,τ) + b2(τ,dϕ) + e(u,dϕ,τ) + e(du,ϕ,τ) + 
                                              b2(dζ,ψ) 

  oper = FEOperator(resid,jacob,Xh,Yh)
  nls = NLSolver(show_trace=true, method=:newton, linesearch=HagerZhang(),  #linesearch=BackTracking(), 
  iterations=25, ftol=1.0E-6)
  solver = FESolver(nls)
  x0=0.001*ones(num_free_dofs(Xh)) 
  xh=FEFunction(Xh,x0)
  (uh,ph,ζh,ϕh), _ = solve!(xh,solver,oper)
    
  writevtk(Ω,"DFARD_Cavity_cells=$(num_cells(model))",order=1,cellfields=["uh"=>uh,"ph"=>ph,"ζh"=>ζh,"ϕh"=>ϕh])
end