module LinearisedPrecondDFARDtest
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Gridap.Algebra
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators

const r = 4
function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
end
function deri(v) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
end


const α0 = 1.0E-0
const α1 = 1.0E-0
grav = VectorValue(0.0,1.0)
function transport(ϕ)
    grav*(α0+α1*ϕ)
end

const α = 1.0e-0
const ρ0 = 1.0e-0

re(x) = (1.0 + ρ0) * 3.0 * ϕ_ex(x) * ϕ_ex(x)* ϕ_ex(x)* ϕ_ex(x)

  u_ex(x)  = VectorValue(0.0000001*cos(π*x[1])*sin(π*x[2]),
                       -0.0000001*sin(π*x[1])*cos(π*x[2]))
  p_ex(x) = sin(π*x[1])*sin(π*x[2])
  ϕ_ex(x) = 0.0000001 * cos(2π*x[1])*cos(2π*x[2])
  w_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)

  f_ex(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*(normu∘u_ex)(x) - (transport∘ϕ_ex)(x)
  g_ex(x) = (∇⋅u_ex)(x)
  m_ex(x) = (ρ0-g_ex(x))*ϕ_ex(x) - (∇⋅w_ex)(x)

  κF(x) = ((1.0/κ) + FF*((normu(u_ex(x)))^(r-2)))
  Finv(x) = 1.0 / κF(x)

  re(x) = 5.0*ϕ_ex(x) *ϕ_ex(x) *ϕ_ex(x) *ϕ_ex(x) 
  re1(x) = (1.0/5.0) * (∇⋅w_ex)(x)
  re2(x) = abs(re1(x))^(-4.0/5.0)

function assemble_df_ard_noExact(model, k, κ, FF, α, ρ0, f_ex, g_ex, m_ex, u_ex, p_ex, ϕ_ex)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)
  reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
  reffe_ϕ = ReferenceFE(lagrangian,Float64,k)

  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="Gamma_u",conformity=:HDiv)
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  Wh_ = TestFESpace(model,reffe_ζ,dirichlet_tags="Gamma_p",conformity=:HDiv)
  Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)

  Uh = TrialFESpace(Uh_,u_ex)
  Qh = TrialFESpace(Qh_)
  Wh = TrialFESpace(Wh_,w_ex)
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

    a1(u,v)   = ∫( (κF)*(u⋅v))*dΩ
    # c(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    # d1(u,w,v) = ∫(FF*v⋅((normu∘(u))*w))dΩ
    # d2(u,w,v) = ∫(FF*(u⋅v)*((deri∘(u))⋅w))dΩ 
    b1(v,q)   = ∫(-(∇⋅v)*q)*dΩ
    F0(v,ϕ)  = ∫((transport∘ϕ)⋅v)*dΩ
    F(v)     = ∫(f_ex⋅v)*dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫(g_ex*q)dΩ
    
    a2(ζ,τ) = ∫( (1.0/α)*(ζ⋅τ))*dΩ
    b2(τ,ψ) = ∫(-(∇⋅τ)*ψ)*dΩ
    c2(ϕ,ψ) = ∫(ρ0*(ϕ * ψ))*dΩ - ∫(g_ex*(ϕ * ψ))*dΩ
    e(u,ϕ,τ) = ∫((1.0/α)*(u⋅τ) * ϕ)*dΩ  
    M(τ) = ∫(ϕ_ex*(τ⋅n_Γu))dΓu
    H(ψ) = ∫(m_ex*ψ)dΩ

    resi((u,p,ζ,ϕ),(v,q,τ,ψ)) =  a1(u,v) + b1(v,p) +
                                 b1(u,q) - F0(v,ϕ) + 
                                 a2(ζ,τ) + b2(τ,ϕ) + e(u_ex,ϕ,τ) + e(u,ϕ_ex,τ) +
                                 b2(ζ,ψ) - c2(ϕ,ψ) 

    jaco((v,q,τ,ψ)) = F(v) - G(q) + M(τ) - H(ψ) 

  op = AffineFEOperator(resi,jaco,Xh,Yh)
  op

end

  function assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, α, ρ0; prec_variant=:prec_variant)
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

        a33_B2(w1,w2) =  ∫((1.0/α)*(w1⋅w2))dΩ + ∫((∇⋅w1)*(∇⋅w2))dΩ#+ ∫((1.0/5.0) * (abs(divergence(w_ex))^(-4.0/5.0)) * (divergence(w1)) * (divergence(w2)) )*dΩ 

        a44a_B2(ϕ1,ϕ2)  = ∫((1.0 + ρ0) * ϕ1 * ϕ2)*dΩ #+ ρ0 *∫((x -> (5 * ϕ_ex(x) * ϕ_ex(x) * ϕ_ex(x) * ϕ_ex(x))) * ϕ1 * ϕ2)*dΩ 
        a44b_B2(ϕ1,ϕ2) = ∫(ρ0 * ϕ1 * ϕ2)dΩ + ∫(α*(∇(ϕ1))⋅∇(ϕ2))dΩ + ∫((α/h_eu)*jump(ϕ1)*jump(ϕ2))dΛu + ∫((α/h_e_Γu)*ϕ1*ϕ2)dΓu
        
     A11_,A22a_,A22b_,A22c_,A33_,A44a_,A44b_ = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2),assemble_matrix(a22c_B2,X2,Y2),assemble_matrix(a33_B2,X3,Y3),assemble_matrix(a44a_B2,X4,Y4),assemble_matrix(a44b_B2,X4,Y4)
     A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_)) + LinearOperator(GridapLinearSolverPreconditioner(A22c_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_)); 
      A22cinv=inv(Array(A22c_));  

    Pinv[range2,range2]=A22ainv+A22binv+A22cinv;         
    
    A44=LinearOperator(GridapLinearSolverPreconditioner(A44a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A44b_)) #+ LinearOperator(GridapLinearSolverPreconditioner(A44c_))

      A44ainv=inv(Array(A44a_));
      A44binv=inv(Array(A44b_)); 
      #A44cinv=inv(Array(A44c_));  

    Pinv[range4,range4]=A44ainv+A44binv;         
    
    end

    A11=LinearOperator(GridapLinearSolverPreconditioner(A11_))
    A11inv=inv(Array(A11_));  
    
    Pinv[range1,range1]=A11inv;

    A33=LinearOperator(GridapLinearSolverPreconditioner(A33_))
    A33inv=inv(Array(A33_));  
    
    Pinv[range3,range3]=A33inv;
    
    evals=eigvals(Pinv*Adense)
    @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))

    
    return BlockDiagonalOperator(A11,A22,A33,A44)
  end

k=0; κ = 1e-10; nk = 4; FF = 1e06

  model=generate_model2d(nk)
  setup_model_labels_unit_square!(model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2*(k+2)-1)
  h_e = CellField(get_array(∫(1)dΛ),Λ)
  Λu = SkeletonTriangulation(model)
  dΛu = Measure(Λu,2*(k+2)-1)
  h_eu = CellField(get_array(∫(1)dΛu),Λu)
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  Γu = BoundaryTriangulation(model, tags = "Gamma_u")
  dΓ = Measure(Γ,2*(k+2)-1)
  dΓu = Measure(Γu,2*(k+2)-1)
  h_e_Γ = CellField(get_array(∫(1)dΓ),Γ)
  h_e_Γu = CellField(get_array(∫(1)dΓu),Γu)
  
  
  op  = assemble_df_ard_noExact(model, k, κ, FF, α, ρ0, f_ex, g_ex, m_ex, u_ex, p_ex, ϕ_ex)
  riesz = assemble_baerland_preconditioner(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, α, ρ0; prec_variant=:B2)

end