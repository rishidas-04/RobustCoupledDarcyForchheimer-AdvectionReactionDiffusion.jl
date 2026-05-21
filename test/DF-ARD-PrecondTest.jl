
module LinearisedDarcyForchheimerPreconditionerTests
  using PertubedSaddlePointPreconditioners
  using Gridap
  import Gridap: ∇
  using Gridap.Algebra
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators
  using DataFrames

  function normu(v,r)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
  end
  function deri(v,r) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
  end

#  const α = 1.0e-09
 const α0 = 1.0E-01
 const α1 = 1.0E-01

grav = VectorValue(0.0,1.0)

function transport(ϕ)
    grav*(α0+α1*ϕ)
end

  dim = 2; I = TensorValue(1,0,0,1)

  # NOTE: this is for the **linearised** problem around um. 
  # First we take only the linearisation F|u|^{r-2}u \approx F|um|^{r-2}u = FHm*u, 
  # where FHm=F*(normu∘um), since normu already contains the r-2 power. This still 
  # accommodates u in L3 if we keep the nonlinearity not in Linfty

  u_ex(x)  = VectorValue((1e-10)*sqrt(2)*cos(π*x[1])*sin(π*x[2]),
                       -(1e-10)*sqrt(2)*sin(π*x[1])*cos(π*x[2]))
#   u_ex(x) = begin
#     u = VectorValue(cos(π*x[1])*sin(π*x[2]),
#                     -sin(π*x[1])*cos(π*x[2]))
#     norm_u = sqrt(u[1]^2 + u[2]^2 + 1e-12)
#     α * u / norm_u
# end
  p_ex(x)  = sin(π*x[1])*cos(π*x[2])
  ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2])
  g_ex(x) = (∇⋅u_ex)(x)
  

  function assemble_linearised_darcy_forchheimer_and_precond(model, u_ex, p_ex, ϕ_ex, g_ex; k=k, κ=κ, FF=FF, r=r, α=α, ρ0=ρ0, prec_variant=:prec_variant)
    
    um(x) = u_ex(x) 
    
    # in the simplest case:
    # FHum(x) = FF*I*(x-> normu(um(x),r))(x)
    # if FH has the full linearisation then:  
    FHum(x) = FF*(I*(x-> normu(um(x),r))(x) + (x-> deri(um(x),r))(x)⊗um(x)) 
    
    κinvpFHum(x) = (1/κ*I + FHum(x))
    gradp_ex(x) = ∇(p_ex)(x)
    f_ex(x) = (1.0/κ)*u_ex(x) + gradp_ex(x) + FHum(x)⋅u_ex(x) - (transport∘ϕ_ex)(x)
    ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
    m_ex(x) =  (ρ0-g_ex(x))*ϕ_ex(x) - (∇⋅ζ_ex)(x) 

    #rprime = 1/(r-1); 

    pweight(x) = (x->inv(FHum(x)))(x) 
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)
    reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
    reffe_ϕ = ReferenceFE(lagrangian,Float64,k)


    # FESpaces
    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags = "Gamma_u",conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
    Wh_ = TestFESpace(model,reffe_ζ,dirichlet_tags="Gamma_p",conformity=:HDiv)
    Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)


    Uh = TrialFESpace(Uh_,um)
    Qh = TrialFESpace(Qh_)
    Wh = TrialFESpace(Wh_,ζ_ex)
    Ψh = TrialFESpace(Ψh_)

    Yh = MultiFieldFESpace([Uh_,Qh_,Wh_,Ψh_])
    Xh = MultiFieldFESpace([Uh,Qh,Wh,Ψh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))

    Λ = SkeletonTriangulation(model)
    dΛ = Measure(Λ,2*(k+2)-1)
    n_Λ = get_normal_vector(Λ)
    h_e = CellField(get_array(∫(1)dΛ),Λ)
    
    Γp = BoundaryTriangulation(model, tags = "Gamma_p")
    dΓp = Measure(Γp,2*(k+2)-1)
    n_Γp = get_normal_vector(Γp)
    h_e_Γp = CellField(get_array(∫(1)dΓp),Γp)

    Γu = BoundaryTriangulation(model, tags = "Gamma_u")
    dΓu = Measure(Γu,2*(k+2)-1)
    n_Γu = get_normal_vector(Γu)
    Λu = SkeletonTriangulation(model)
    dΛu = Measure(Λu,2*(k+2)-1)
    h_eu = CellField(get_array(∫(1)dΛu),Λu)
    h_e_Γu = CellField(get_array(∫(1)dΓu),Γu)
   

    a(u,v)   = ∫( (1/κ)*(u⋅v))*dΩ
    d1(u,v) = ∫((FHum⋅u)⋅v)dΩ
    b(v,q)   = ∫(-(∇⋅v)*q)dΩ
    F0(v,ϕ)  = ∫((transport∘ϕ)⋅v)*dΩ
    F(v)     = ∫(f_ex⋅v)dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫((-1.0)*(g_ex*q))dΩ

    a2(ζ,τ) = ∫( (1.0/α)*(ζ⋅τ))*dΩ
    b2(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
    c2(ϕ,ψ) = ∫(ρ0*(ϕ * ψ))*dΩ - ∫(g_ex*(ϕ * ψ))*dΩ
    e(u,ϕ,τ) = ∫((1/α)*(u⋅τ) * ϕ)*dΩ  
    M(τ) = ∫(ϕ_ex*(τ⋅n_Γu))dΓu
    H(ψ) = ∫(m_ex*ψ)dΩ

    lhs((u, p, ζ, ϕ),(v, q, τ, ψ)) = a(u,v) + d1(u,v) +  
                         b(v,p) + b(u,q) +
                         a2(ζ,τ) + b2(τ,ϕ) + e(u_ex,ϕ,τ) +
                         b2(ζ,ψ) - c2(ϕ,ψ) 
  
    rhs((v, q, τ, ψ)) = F(v) + G(q) + M(τ) - H(ψ)

    op = AffineFEOperator(lhs,rhs,Xh,Yh)

ufield = CellField(u_ex, Ω)
hK = CellField(get_array(∫(1)dΩ), Ω)
# hK = CellDiameter(Ω)
unorm = CellField(x -> norm(u_ex(x))^2, Ω)
β = ufield⋅n_Λ;
βu = ufield⋅n_Γu; βp = ufield⋅n_Γp
# τK = CellField(x -> unorm(x) *(1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2)), Ω)
τK = CellField(x -> (1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2 + ρ0)), Ω)


    # block diagonal preconditioner 
    # with blocks spectrally equivalent to the diagonal blocks of 
    # the Jacobian of the linearised problem.
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
    if (prec_variant==:B1)
      
      a11_B1(u,v) = ∫((1/κ)*(u⋅v) + ((FHum⋅u)⋅v) + (∇⋅u)*(∇⋅v))dΩ
      
      a22a_B1(p,q)= ∫(p*q)dΩ

      a22b_B1(p,q) = ∫(κ*(∇(p))⋅∇(q))dΩ + 
                     ∫((κ/h_e)*jump(p)*jump(q))dΛ + 
                     ∫((κ/h_e_Γp)*p*q)dΓp
                         
      a22c_B1(p,q) = ∫((pweight⋅∇(p))⋅∇(q))dΩ +
                     ∫(1.0/h_e*(x->pweight(x)⊙I/dim)*jump(p)*jump(q))dΛ + 
                     ∫(1.0/h_e_Γp*(x->pweight(x)⊙I/dim)*p*q)dΓp

      small = 1e-12
      a33_B1(w1,w2) =   ∫(((1.0/α))*(w1⋅w2))dΩ + ∫((∇⋅w1)*(∇⋅w2))*dΩ #+ ∫((1.0/α)*(((∇⋅w1)*((x->u_ex(x))⋅w2))+small))*dΩ 
      a44a_B1(ϕ, ψ) = ∫((1.0+ρ0)*ϕ*ψ)dΩ 
      # a44b_B1(ϕ, ψ) = ∫((ρ0)*ϕ*ψ)dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
      #                 ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
      #                 ∫((α/h_e_Γu)*ϕ*ψ)dΓu 
      
      a44b_B1(ϕ, ψ) = ∫((ρ0)*ϕ*ψ)dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
               ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
               ∫((α/h_e_Γu)*ϕ*ψ)dΓu - 
               ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ -
               ∫(α*((∇(ψ))⋅n_Γu)*ϕ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
               ∫((u_ex⋅∇(ϕ))*ψ)dΩ +  
               ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ + #skeleton central
               ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ + #skeleton upwind
               ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu + #boundary upwind
               ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp + #boundary upwind
              #  ∫(((u_ex*ϕ)⋅(u_ex*ψ)))dΩ +
               ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ +
               ∫(τK*ϕ*ρ0*(u_ex⋅(∇(ψ))))dΩ #+
              #  ∫(τK*(ϕ)*unorm*(u_ex⋅(∇(ψ))))dΩ

  
      A11_,A22a_,A22b_,A22c_ = assemble_matrix(a11_B1,Uh,Uh_),assemble_matrix(a22a_B1,Qh,Qh_),assemble_matrix(a22b_B1,Qh,Qh_),assemble_matrix(a22c_B1,Qh,Qh_)
      A33_ = assemble_matrix(a33_B1,Wh,Wh_)
      A44a_ = assemble_matrix(a44a_B1,Ψh,Ψh_)
      A44b_ = assemble_matrix(a44b_B1,Ψh,Ψh_)

      A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_)) +
             LinearOperator(GridapLinearSolverPreconditioner(A22c_))

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
    else
      a11_B2(u,v)=∫((κinvpFHum⋅u)⋅v + (x->κinvpFHum(x)⊙I/dim)*(∇⋅u)*(∇⋅v))dΩ
      a22_B2(p,q)= ∫((x->inv(κinvpFHum(x))⊙I/dim)*p*q)dΩ

      A11_,A22_= assemble_matrix(a11_B2,Uh,Uh_),assemble_matrix(a22_B2,Qh,Qh_)  
      A22=LinearOperator(GridapLinearSolverPreconditioner(A22_)) 

      A22inv=inv(Array(A22_));  
      Pinv[range2,range2]=A22inv;  
      small = 1e-12
      a33_B2(w1,w2) =  ∫(((1.0/α))*(w1⋅w2))dΩ + ∫((∇⋅w1)*(∇⋅w2))*dΩ + ∫((1.0/α)*(((∇⋅w1)*((x->u_ex(x))⋅w2))+small))*dΩ # + ∫((1.0/α)*(∇⋅w1)*((x->u_ex(x))⋅w2))*dΩ
      a44a_B2(ϕ, ψ) = ∫((1.0+ρ0)*ϕ*ψ)dΩ #+ ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ
      
      a44b_B2(ϕ,ψ) = ∫(ρ0 * ϕ * ψ)*dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))dΩ + 
                     ∫((α/h_eu)*jump(ϕ)*jump(ψ))dΛu + 
                     ∫((α/h_e_Γu)*ϕ*ψ)dΓu 

      # a44b_B2(ϕ, ψ) = ∫((ρ0)*ϕ*ψ)dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
      #          ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
      #          ∫((α/h_e_Γu)*ϕ*ψ)dΓu - 
      #          ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ -
      #          ∫(α*((∇(ψ))⋅n_Γu)*ϕ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
      #          ∫((u_ex⋅∇(ϕ))*ψ)dΩ +  
      #          ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ + #skeleton central
      #          ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ + #skeleton upwind
      #          ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu + #boundary upwind
      #          ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp + #boundary upwind
      #         #  ∫(((u_ex*ϕ)⋅(u_ex*ψ)))dΩ +
      #          ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ +
      #          ∫(τK*ϕ*ρ0*(u_ex⋅(∇(ψ))))dΩ #+
      #         #  ∫(τK*(ϕ)*unorm*(u_ex⋅(∇(ψ))))dΩ


       A33_ = assemble_matrix(a33_B2,Wh,Wh_)
      A44a_ = assemble_matrix(a44a_B2,Ψh,Ψh_)
      A44b_ = assemble_matrix(a44b_B2,Ψh,Ψh_)
       
    
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
    
    return op, BlockDiagonalOperator(A11,A22,A33,A44), evals
  end

  # run now the tests 
  table = DataFrame(nk=Int[], r=Float64[], κ=Float64[], FF=Float64[], α=Float64[], ρ0=Float64[], niter=Int[], cond_num=Float64[])
  
  for nk in (2,3,4)

    model = generate_model2d(nk)
    setup_model_labels_unit_square!(model)
    # for r in (3,3.5)
    r = 3
      #  for κ in (1e-8, 1e-4, 1)
        #  for FF in (1e6, 1, 1e-6)
        κ = 1e-08
        FF = 1e06
          #  for α in (1e-9, 1e-6, 1)
           α = 1e-09
            for ρ0 in (1, 1e-5, 1e-9)
            #  ρ0 = 1e-0
          println("\n--- Running for nk=$nk, r=$r, kappa=$κ, FF=$FF, α=$α, ρ0=$ρ0 ---")

          oper,riesz,evals = assemble_linearised_darcy_forchheimer_and_precond(model, u_ex, p_ex, ϕ_ex, g_ex; k=0, κ=κ, FF=FF, r=r, α=α, ρ0=ρ0, prec_variant=:B1)
          
          A = oper.op.matrix
          b = oper.op.vector
          x0 = 0.001*zeros(num_free_dofs(oper.trial))
          x, hist = minres(A, b, M = riesz, itmax=1000, atol=1e-9, rtol=1e-9, verbose=0)
          # @printf("-------------------------------------------------")
          @printf("MINRES converged in %d iterations\n", hist.niter)
          @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))
        # @printf("-------------------------------------------------")
          push!(table,(r=r,κ=κ,FF=FF,α=α,ρ0=ρ0,nk=nk,niter=hist.niter,cond_num=maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals))))
      
#     end  
  end
# end 
# # end
# end
end

  @show(table) 
end
