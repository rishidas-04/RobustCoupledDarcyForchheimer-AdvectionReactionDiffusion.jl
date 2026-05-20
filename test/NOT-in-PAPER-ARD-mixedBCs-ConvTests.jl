module ARDConvergence2D_mixedBCs
  using GridapRobustDarcyForchheimer
  using Gridap
  import Gridap: ∇
  using Printf
  using Test
  using LineSearches: BackTracking
  using LinearAlgebra
  
  const α = 1.0
  const ρ0 = 1.0

  
ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2])
u_ex(x)  = VectorValue(cos(π*x[1])*sin(π*x[2]),
                      -sin(π*x[1])*cos(π*x[2]))

ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
m_ex(x) =  (-1.0)*(∇⋅ζ_ex)(x) #(ρ0)*ϕ_ex(x) - (∇⋅ζ_ex)(x)
h_ex(x) = (1.0/α)*ζ_ex(x) - (∇(ϕ_ex))(x) + (1.0/α)*u_ex(x)*ϕ_ex(x)


function solve_ARD(model; k=k, generate_output=false)
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
H(τ) = ∫(ϕ_ex*(τ⋅n_Γu))*dΓu #+ ∫(h_ex⋅τ)*dΩ
M(ψ) = ∫(-(m_ex*ψ))*dΩ

lhs((ζ, ϕ),(τ, ψ)) = a(ζ,τ) + b(τ,ϕ) + e(ϕ,τ) +
                         b(ζ,ψ) - c(ϕ,ψ)  
  
rhs((τ, ψ)) = H(τ) + M(ψ)

op = AffineFEOperator(lhs,rhs,Xh,Yh)
ζh, ϕh = solve(op)


    eζh = ζ_ex - ζh
    eϕh = ϕ_ex - ϕh
    eζ =  (sum(∫(((∇⋅eζh)*(∇⋅eζh)).^(0.5*6.0/5.0))dΩ))^(5.0/6.0) + sqrt(sum(∫(eζh⋅eζh)dΩ))
    eϕ = (sum(∫((eϕh*eϕh).^(0.5*6.0))dΩ))^(1.0/6.0)


   eζ, eϕ, Gridap.FESpaces.num_free_dofs(Xh)

  end

  
  function  convergence_test(; nkmax, k=k, generate_output=false)
    eζ = Float64[]
    eϕ = Float64[]
    rζ = Float64[]
    rϕ = Float64[]
    hh = Float64[]
    nn = Int[]
    
    push!(rζ,0.)
    push!(rϕ,0.)

    for nk in 1:nkmax
        println("******** Refinement step: $nk")
        model=generate_model2d(nk)
        setup_model_labels_unit_square!(model)
        error_ζ,error_ϕ,ndofs=solve_ARD(model; k=k, generate_output=generate_output)

        push!(nn,ndofs)
        push!(hh,sqrt(2)/(2^nk))
        println("******** Total DoFs: ", nn[nk])
        
        push!(eζ,error_ζ)
        push!(eϕ,error_ϕ)
      
        if nk>1
            push!(rζ, log(eζ[nk]/eζ[nk-1])/log(hh[nk]/hh[nk-1]))
            push!(rϕ, log(eϕ[nk]/eϕ[nk-1])/log(hh[nk]/hh[nk-1]))
        end
    end
    println("=======================================================")
    println("   DoF  &    h   &   e(ζ)   &  r(ζ) &   e(ϕ)   &  r(ϕ) ")
    println("=======================================================")
    for nk in 1:nkmax
        @printf("%7d & %.4f & %.2e & %.3f & %.2e & %.3f\n", 
                nn[nk], hh[nk], eζ[nk], rζ[nk], eϕ[nk], rϕ[nk]);
    end
    println("=======================================================")
  end
  convergence_test(;nkmax=5,k=0,generate_output=false)
end
