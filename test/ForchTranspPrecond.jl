
module BaerlandDarcyForchheimerPreconditionerTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Gridap.Algebra
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators

function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
  end
function deri(v) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
  end

 const r = 4

    u_ex(x)  = VectorValue(0.001*cos(π*x[1])*sin(π*x[2]),
                           -0.001*sin(π*x[1])*cos(π*x[2]))
    p_ex(x)  = 0.001*sin(π*x[1])*cos(π*x[2])
    w_ex(x)  = VectorValue(0.001*cos(π*x[1])*sin(π*x[2]),
                           -0.001*sin(π*x[1])*cos(π*x[2]))
    ϕ_ex(x)  = 0.001*sin(π*x[1])*cos(π*x[2])


  f_rhs(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*((normu∘u_ex)(x))^(r-2)
  g_rhs(x) = (∇⋅u_ex)(x)
  h_rhs(x) = (1.0/α)*w_ex(x) - ∇(ϕ_ex)(x) 
  m_rhs(x) = ρ*ϕ_ex(x) - (∇⋅w_ex)(x)

    κF(x) = ((1.0/κ) + FF*((normu(u_ex(x)))^(r-2)))
    κinv(x) = 1.0 / κF(x)
    Finv(x) = 1.0/(FF*((normu(u_ex(x)))^(r-2)))
    rec(x) = (normu(p_ex(x)))^(r-2)
    rec1(x) = (normu(p_ex(x)))^(r-4)


    function assemble_forchheimer_transport_noExact(model, k, κ, FF, α, ρ, f_rhs, g_rhs, h_rhs, m_rhs, u_ex, p_ex, w_ex, ϕ_ex)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)
  reffe_w = ReferenceFE(raviart_thomas,Float64,k)
  reffe_ϕ = ReferenceFE(lagrangian,Float64,k)


  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags = "Gamma_u",conformity=:HDiv)#,dirichlet_tags="boundary")
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  Wh_ = TestFESpace(model,reffe_u,dirichlet_tags = "boundary",conformity=:HDiv)
  Φh_ = TestFESpace(model,reffe_p,conformity=:L2)
  
  Uh = TrialFESpace(Uh_,u_ex)#VectorValue(0,0))
  Qh = TrialFESpace(Qh_)
  Wh = TrialFESpace(Wh_,w_ex)
  Φh = TrialFESpace(Φh_,ϕ_ex)
  
  Yh = MultiFieldFESpace([Uh_,Qh_,Wh_,Φh_])
  Xh = MultiFieldFESpace([Uh,Qh,Wh,Φh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  dΓ = Measure(Γ,2*(k+2)-1)
  n_Γ = get_normal_vector(Γ)

  a((u, p, w1, ϕ1),(v, q, w2, ϕ2)) = ∫( (1/κ)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ + ∫(κF*(v⋅u))dΩ + (r-2)*∫((x -> (κF(x))^(r-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ + ∫((1.0/α) * (w1⋅w2))*dΩ + ∫(ϕ1 * (∇⋅w2))*dΩ + ∫(ρ*ϕ1*ϕ2)*dΩ - ∫(ϕ2 * (∇⋅w1))*dΩ
  b((v, q, w2, ϕ2)) = ∫(f_rhs⋅v)*dΩ -∫(p_ex*(v⋅n_Γ))dΓ - ∫(q*g_rhs)*dΩ + ∫(m_rhs*ϕ2)*dΩ + ∫(h_rhs⋅w2)*dΩ

  op = AffineFEOperator(a,b,Xh,Yh)
  op
end

 
  function assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, α, ρ; prec_variant=:prec_variant)
    X1,X2,X3,X4 = op.trial
    Y1,Y2,Y3,Y4 = op.test

    N  = num_free_dofs(op.trial)
    Ns = [num_free_dofs(U) for U in op.trial] 
    Np = zeros(Int,length(Ns)+1)
    Np[1]=1
      for i=2:length(Ns)+1
          Np[i]=Np[i-1]+Ns[i-1]
      end
    Pinv=zeros(N,N); 
    @show size(Pinv)
    Adense = Array(op.op.matrix)
    range1=Np[1]:Np[2]-1
    range2=Np[2]:Np[3]-1
    range3=Np[3]:Np[4]-1
    range4=Np[4]:Np[5]-1
    
    @assert prec_variant in (:B1,:B2)
    if (prec_variant==:B2)
    
        a11_B2(u,v) = ∫(κF*(u⋅v) + (∇⋅u)*(∇⋅v))dΩ
        a22a_B2(p,q)= ∫(p*q)dΩ
        a22b_B2(p,q) = ∫(κ*(∇(p))⋅∇(q))dΩ + ∫((κ/h_e)*jump(p)*jump(q))dΛ + ∫((κ/h_e_Γ)*p*q)dΓ
        a22c_B2(p,q) = ∫(Finv*(∇(p))⋅∇(q))dΩ + ∫((Finv/h_e)*jump(p)*jump(q))dΛ + ∫((Finv/h_e_Γ)*p*q)dΓ
        a33_B2(w1,w2) =  ∫((1.0/α)*(w1⋅w2))dΩ + ∫(abs(((∇⋅w1) * (∇⋅w2))).^0.6)*dΩ
        a44a_B2(ϕ1,ϕ2) = ∫((abs(ϕ1)) * (abs(ϕ1)) * (abs(ϕ1)) *(abs(ϕ1)) *ϕ1 * ϕ2)dΩ
        a44b_B2(ϕ1,ϕ2) = (1/(1+ρ))*∫(α*(∇(ϕ1))⋅∇(ϕ2))dΩ + (1/(1+ρ))*∫((α/h_e)*jump(ϕ1)*jump(ϕ2))dΛ + (1/(1+ρ))*∫((α/h_e_Γ)*ϕ1*ϕ2)dΓ


     A11_,A22a_,A22b_,A22c_,A33_,A44a_,A44b_ = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2),assemble_matrix(a22c_B2,X2,Y2),assemble_matrix(a33_B2,X3,Y3),assemble_matrix(a44a_B2,X4,Y4),assemble_matrix(a44b_B2,X4,Y4)
     A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_)) + LinearOperator(GridapLinearSolverPreconditioner(A22c_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_)); 
      A22cinv=inv(Array(A22c_));  

    Pinv[range2,range2]=A22ainv+A22binv+A22cinv;         
    
    A44=LinearOperator(GridapLinearSolverPreconditioner(A44a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A44b_)) #+ LinearOperator(GridapLinearSolverPreconditioner(A22c_))

      A44ainv=inv(Array(A44a_));
      A44binv=inv(Array(A44b_)); 
    #   A22cinv=inv(Array(A22c_));  

    Pinv[range4,range4]=A44ainv+A44binv;         
    

    end

    A11=LinearOperator(GridapLinearSolverPreconditioner(A11_))
    A11inv=inv(Array(A11_));  
    
    Pinv[range1,range1]=A11inv;

    A33=LinearOperator(GridapLinearSolverPreconditioner(A33_))
    A33inv=inv(Array(A33_));  
    
    Pinv[range3,range3]=A33inv;
    # A44=LinearOperator(GridapLinearSolverPreconditioner(A44_))
    # A44inv=inv(Array(A44_));  
    
    # Pinv[range4,range4]=A44inv;
    
    evals=eigvals(Pinv*Adense)
    @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))

    
    return BlockDiagonalOperator(A11,A22,A33,A44)
  end

k=0; κ = 1e-07; nk = 5; FF = 1e04; ρ = 1e-02; α=1e-05

  model=generate_model2d(nk)
  setup_model_labels_unit_square!(model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*(k+2)-1)
  h_e = CellField(get_array(∫(1)dΛ),Λ)
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  dΓ = Measure(Γ,2*(k+2)-1)
  h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)
  
  
  oper  = assemble_forchheimer_transport_noExact(model, k, κ, FF, α, ρ, f_rhs, g_rhs, h_rhs, m_rhs, u_ex,p_ex, w_ex, ϕ_ex)
  riesz = assemble_baerland_preconditioner(oper, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, α, ρ; prec_variant=:B2)

  
end