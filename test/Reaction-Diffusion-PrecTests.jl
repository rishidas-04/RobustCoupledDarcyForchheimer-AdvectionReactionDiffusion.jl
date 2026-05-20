# Baerland B1 type preconditioner for reaction-diffusion equation



module AdvectionRecationDifusionPreconditionerTests
  using GridapRobustDarcyForchheimer
  using Gridap
  import Gridap: ∇
  using Gridap.Algebra
  using Printf
  using Test
  using Krylov
  using LinearAlgebra
  using LinearOperators
  using DataFrames
  using Gridap.Geometry: simplexify
  # using AlgebraicMultigrid

  
ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2])
u_ex(x)  = VectorValue(0.01*cos(π*x[1])*sin(π*x[2]),
                      -0.01*sin(π*x[1])*cos(π*x[2]))
  # u_ex(x)  = VectorValue(1e-06,-1e-06) # almost zero velocity to test the preconditioner in the diffusion-dominated regime

  function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

function assemble_advection_reaction_difusion_precond(model, u_ex, ϕ_ex; k=k, α=α, ρ0=ρ0, prec_variant=:prec_variant)


ζ_ex(x) = α*∇(ϕ_ex)(x) 
m_ex(x) = (ρ0)*ϕ_ex(x) - (∇⋅ζ_ex)(x)
h_ex(x) = (1.0/α)*ζ_ex(x) - (∇(ϕ_ex))(x) 



reffe_ζ = ReferenceFE(raviart_thomas,Float64,k)
reffe_ϕ = ReferenceFE(lagrangian,Float64,k)

Wh_ = TestFESpace(model,reffe_ζ,dirichlet_tags="Gamma_p",conformity=:HDiv)
Ψh_ = TestFESpace(model,reffe_ϕ,conformity=:L2)

Wh = TrialFESpace(Wh_,ζ_ex)
Ψh = TrialFESpace(Ψh_)

Yh = MultiFieldFESpace([Wh_,Ψh_])
Xh = MultiFieldFESpace([Wh,Ψh])

Ω = Triangulation(model)
dΩ = Measure(Ω,2*(k+2))

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ,2*(k+2)-1)
h_e = CellField(get_array(∫(1)dΛ),Λ)
 
Γu = BoundaryTriangulation(model, tags = "Gamma_u")
dΓu = Measure(Γu,2*(k+2)-1)
n_Γu = get_normal_vector(Γu)
Γp = BoundaryTriangulation(model, tags = "Gamma_p")
dΓp = Measure(Γp,2*(k+2)-1)
n_Γp = get_normal_vector(Γp)
Λu = SkeletonTriangulation(model)
dΛu = Measure(Λu,2*(k+2)-1)
h_eu = CellField(get_array(∫(1)dΛu),Λu)
h_e_Γu = CellField(get_array(∫(1)dΓu),Γu)
h_e_Γp = CellField(get_array(∫(1)dΓp),Γp)


a(ζ,τ) = ∫((1.0/α)*(ζ⋅τ))*dΩ
b(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
c(ϕ,ψ) = ∫((ρ0)*ϕ*ψ)*dΩ
e(ϕ,τ) = ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ
H(τ) = ∫(ϕ_ex*(τ⋅n_Γu))*dΓu + ∫(h_ex⋅τ)*dΩ
M(ψ) = ∫(-(m_ex*ψ))*dΩ

lhs((ζ, ϕ),(τ, ψ)) = a(ζ,τ) + b(τ,ϕ) + #e(ϕ,τ) +
                         b(ζ,ψ) - c(ϕ,ψ)
  
rhs((τ, ψ)) = H(τ) + M(ψ)

op = AffineFEOperator(lhs,rhs,Xh,Yh)



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

f = CellField(x -> norm(u_ex(x)), Ω)
@assert prec_variant in (:B1,:B2)
    if (prec_variant==:B1)
       a11_B1(ζ,τ) = ∫((1/α)*(ζ⋅τ) + (∇⋅ζ)*(∇⋅τ))*dΩ 

       a22a_B1(ϕ,ψ)= ∫((1.0+ρ0) * ϕ * ψ)*dΩ
       a22b_B1(ϕ,ψ) = ∫(ρ0 * ϕ * ψ)*dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))dΩ + 
                     ∫((α/h_eu)*jump(ϕ)*jump(ψ))dΛu + 
                     ∫((α/h_e_Γu)*ϕ*ψ)dΓu #+∫(((1.0))*f *f * ϕ * ψ)dΩ #+ ∫((1.0/α)*((u_ex⋅(∇(ϕ)))*ψ))dΩ
  
      A11_,A22a_,A22b_ = assemble_matrix(a11_B1,Wh,Wh_),assemble_matrix(a22a_B1,Ψh,Ψh_),assemble_matrix(a22b_B1,Ψh,Ψh_)
      A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_)) 

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_));  
      
      Pinv[range2,range2]=A22ainv+A22binv;         
    
    else
      a11_B2(ζ,τ)=∫((1.0/α)*(ζ⋅τ) + (1.0/α)*(∇⋅ζ)*(∇⋅τ))*dΩ 
    #    a22_B2(ϕ,ψ) =  ∫((ρ0 + 1.0)*(ϕ*ψ))*dΩ
      a22_B2(ϕ,ψ) = ∫((ρ0+(1.0/α)) * ϕ * ψ)*dΩ + ∫((α)*(∇(ϕ))⋅∇(ψ))dΩ + 
                     ∫(((α)/h_eu)*jump(ϕ)*jump(ψ))dΛu + 
                     ∫(((α)/h_e_Γu)*ϕ*ψ)dΓu

      A11_,A22_= assemble_matrix(a11_B2,Wh,Wh_),assemble_matrix(a22_B2,Ψh,Ψh_)  
      A22=LinearOperator(GridapLinearSolverPreconditioner(A22_)) 

      A22inv=inv(Array(A22_));  
      Pinv[range2,range2]=A22inv;  
    end
A11=LinearOperator(GridapLinearSolverPreconditioner(A11_))
A11inv=inv(Array(A11_));  
    
Pinv[range1,range1]=A11inv;
    
evals=eigvals(Pinv*Adense)
    
return op, BlockDiagonalOperator(A11,A22), evals
end


# Run the tests for a range of parameters and store results in a table

table = DataFrame(nk=Int[], α=Float64[],  ρ0=Float64[], niter=Int[], cond_num=Float64[])
  
  for nk in (2,3,4)

    model = generate_model2d(nk)
    setup_model_labels_unit_square!(model)
       for α in (1e-5, 1e-4, 1e-3)
        for ρ0 in (1e-9, 1e-6, 1e-3)
          #  α = ρ0
          println("\n--- Running for nk=$nk, α=$α, ρ0=$ρ0 ---")

          oper,riesz,evals = assemble_advection_reaction_difusion_precond(model, u_ex, ϕ_ex; k=0, α=α, ρ0=ρ0, prec_variant=:B2)

          A = oper.op.matrix
          b = oper.op.vector
          x0 = 0.001*zeros(num_free_dofs(oper.trial))
          x, hist = minres(A, b, M = riesz, itmax=1000, atol=1e-8, rtol=1e-8, verbose=0)
          @printf("MINRES converged in %d iterations\n", hist.niter)
          @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))
        #   push!(table,(α=α,ρ0=ρ0,nk=nk,niter=hist.niter,cond_num=maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals))))
      cond = maximum(abs.(evals)) / minimum(abs.(evals))
cond = round(cond, digits=1)
push!(table,(α=α,ρ0=ρ0,nk=nk,niter=hist.niter,cond_num=cond))
         end
       end
  end
  @show(table) 
end