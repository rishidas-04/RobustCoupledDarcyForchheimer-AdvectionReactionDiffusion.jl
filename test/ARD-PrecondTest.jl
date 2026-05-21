# Baerland preconditioners for advection-reaction-diffusion equation

# We take

# alpha^{-1} A B^* + alpha^{-1}D
# B 0 - ρ0 C

# A is (zeta,tau), B is (div tau, phi), D is (u.tau,phi)
module AdvectionReactionDiffusionPreconditionerTests

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

function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2)
  end

ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2]) + exp(π*x[1]*x[2])

u_ex(x)  = VectorValue((1e-1)*sqrt(2)*cos(π*x[1])*sin(π*x[2]),
                       -(1e-1)*sqrt(2)*sin(π*x[1])*cos(π*x[2]))

# u_ex(x) = VectorValue(1e-0,-1e-0) # almost zero velocity to test the preconditioner in the diffusion-dominated regime
############ Generating meshes #######################################

function generate_model2d(nk)
domain = (0,1,0,1)
n = 2^nk
partition = (n,n)
CartesianDiscreteModel(domain, partition) |> simplexify
end

function setup_model_labels_unit_square!(model)
labels = get_face_labeling(model)
add_tag!(labels, "Gamma_p", [6,2,3,4,8]) # 6 top, right, and other 3 corners
add_tag!(labels, "Gamma_u", [1,2,3,5,7]) # bottom and 7 left and low-left corner
end

###################### Preconditioning mechanism ###############################

function custom_stopping_condition(solver, A, b, r, tol)
mul!(r, A, solver.x)
r .-= b
bool = norm(r) ≤ tol*norm(b)
@printf("ERR NORM: ||b-Ax||/||b||: %16.7e\n", norm(r)/norm(b))
return bool
end

struct GridapLinearSolverPreconditioner{A,B}
numerical_setup::A
matrix::B
function GridapLinearSolverPreconditioner(matrix; ls=LUSolver())
ss = symbolic_setup(ls, matrix)
ns = numerical_setup(ss, matrix)
A = typeof(ns)
B = typeof(matrix)
new{A,B}(ns, matrix)
end
end

function LinearAlgebra.mul!(z::AbstractVector{T},
M::GridapLinearSolverPreconditioner,
r::AbstractVector{T}) where T
Gridap.solve!(z, M.numerical_setup, r)
end

function LinearOperators.LinearOperator(solver::GridapLinearSolverPreconditioner)
function apply(res, x, α, β)
if β ≈ 0.0
mul!(res, solver, x)
else
@assert α ≈ 1.0
tmp = copy(res)
mul!(tmp, solver, x)
res .= β .* res .+ α .* tmp
end
end
LinearOperator(
Float64,
size(solver.matrix, 1),
size(solver.matrix, 2),
true,
true,
apply,
)
end

####################### Assembling the preconditioner and testing its performance #######################

function assemble_advection_reaction_difusion_precond(model, u_ex, ϕ_ex; k=k, α=α, ρ0=ρ0, prec_variant=:prec_variant)
ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
m_ex(x) = (-1.0)*(∇⋅ζ_ex)(x) + ρ0*ϕ_ex(x)
h_ex(x) = (1.0/α)*ζ_ex(x) - (∇(ϕ_ex))(x) + (1.0/α)*u_ex(x)*ϕ_ex(x)

reffe_ζ = ReferenceFE(raviart_thomas, Float64, k)
reffe_ϕ = ReferenceFE(lagrangian, Float64, k)

Wh_ = TestFESpace(model, reffe_ζ, dirichlet_tags="Gamma_p", conformity=:HDiv)
Ψh_ = TestFESpace(model, reffe_ϕ, conformity=:L2)

Wh = TrialFESpace(Wh_, ζ_ex)
Ψh = TrialFESpace(Ψh_)

Yh = MultiFieldFESpace([Wh_, Ψh_])
Xh = MultiFieldFESpace([Wh, Ψh])

Ω = Triangulation(model)
dΩ = Measure(Ω, 2*(k+2))

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 2*(k+2)-1)
h_e = CellField(get_array(∫(1)dΛ), Λ)

Γu = BoundaryTriangulation(model, tags = "Gamma_u")
dΓu = Measure(Γu, 2*(k+2)-1)
n_Γu = get_normal_vector(Γu)
Γp = BoundaryTriangulation(model, tags = "Gamma_p")
dΓp = Measure(Γp, 2*(k+2)-1)
n_Γp = get_normal_vector(Γp)
Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ, 2*(k+2)-1)
n_Λ = get_normal_vector(Λ)
h_e = CellField(get_array(∫(1)dΛ), Λ)
h_e_Γu = CellField(get_array(∫(1)dΓu), Γu)
h_e_Γp = CellField(get_array(∫(1)dΓp), Γp)

a(ζ, τ) = ∫((1.0/α)*(ζ⋅τ))*dΩ
b(τ, ψ) = ∫((∇⋅τ)*ψ)*dΩ
c(ϕ,ψ) = ∫((ρ0)*ϕ*ψ)*dΩ
e(ϕ, τ) = ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ
H(τ) = ∫(ϕ_ex*(τ⋅n_Γu))*dΓu # + ∫(h_ex⋅τ)*dΩ
M(ψ) = ∫(-(m_ex*ψ))*dΩ

lhs((ζ, ϕ), (τ, ψ)) = a(ζ, τ) + b(τ, ϕ) + e(ϕ, τ) +
                      b(ζ, ψ) - c(ϕ, ψ)

rhs((τ, ψ)) = H(τ) + M(ψ)

op = AffineFEOperator(lhs, rhs, Xh, Yh)
ufield = CellField(u_ex, Ω)
hK = CellField(get_array(∫(1)dΩ), Ω)
# hK = CellDiameter(Ω)
unorm = CellField(x -> norm(u_ex(x))^2, Ω)
β = ufield⋅n_Λ;
βu = ufield⋅n_Γu; βp = ufield⋅n_Γp
# τK = CellField(x -> unorm(x) *(1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2)), Ω)
τK = CellField(x -> (1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2 + ρ0)), Ω)



N = num_free_dofs(op.trial)
Ns = [num_free_dofs(U) for U in op.trial]
Np = zeros(Int, length(Ns)+1)
Np[1] = 1
for i = 2:length(Ns)+1
Np[i] = Np[i-1] + Ns[i-1]
end

Pinv = zeros(N, N)
@show size(Pinv)
Adense = Array(op.op.matrix)
range1 = Np[1]:Np[2]-1
range2 = Np[2]:Np[3]-1

f = CellField(x -> norm(u_ex(x)), Ω)

@assert prec_variant in (:B1, :B2)
if prec_variant == :B1
a11_B1(ζ, τ) = ∫((1/α)*(ζ⋅τ)+ (1/α)*(∇⋅ζ)*(∇⋅τ))*dΩ 
a22_B1(ϕ, ψ) = ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
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
               ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ

#  a22_B1(ϕ, ψ) = ∫((α)*ϕ*ψ)*dΩ

A11_, A22_ = assemble_matrix(a11_B1, Wh, Wh_), assemble_matrix(a22_B1, Ψh, Ψh_)
A22 = LinearOperator(GridapLinearSolverPreconditioner(A22_))
A22inv = inv(Array(A22_))
Pinv[range2, range2] = A22inv
else
a11_B2(ζ, τ) = ∫((1.0/α)*(ζ⋅τ) +  (∇⋅ζ)*(∇⋅τ))*dΩ

a22a_B2(ϕ, ψ) = ∫((ρ0+1.0)*ϕ*ψ)dΩ 


a22b_B2(ϕ, ψ) = ∫((ρ0)*ϕ*ψ)dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
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

A11_, A22a_, A22b_ = assemble_matrix(a11_B2, Wh, Wh_), assemble_matrix(a22a_B2, Ψh, Ψh_), assemble_matrix(a22b_B2, Ψh, Ψh_)
A22 = LinearOperator(GridapLinearSolverPreconditioner(A22a_)) +
LinearOperator(GridapLinearSolverPreconditioner(A22b_))

A22ainv = inv(Array(A22a_))
A22binv = inv(Array(A22b_))
Pinv[range2, range2] = A22ainv + A22binv 
end

A11 = LinearOperator(GridapLinearSolverPreconditioner(A11_))
A11inv = inv(Array(A11_))
Pinv[range1, range1] = A11inv

evals = eigvals(Pinv*Adense)
return op, BlockDiagonalOperator(A11, A22), evals
end

# Run the tests for a range of parameters and store results in a table

table = DataFrame(nk=Int[], α=Float64[], ρ0=Float64[], niter=Int[], cond_num=Float64[])

for nk in (2,3,4)
model = generate_model2d(nk)
setup_model_labels_unit_square!(model)
 for α in (1e-9,1e-6,1e-3, 1)
#   α = 1e-04
# ρ0 = 1e-9
 for ρ0 in (1e-8,1e-4,1)
println("\n--- Running for nk=$nk, α=$α, ρ0=$ρ0---")

oper, riesz, evals = assemble_advection_reaction_difusion_precond(model,u_ex,ϕ_ex;k=0,α=α,ρ0=ρ0,prec_variant=:B2,)

A = oper.op.matrix
b = oper.op.vector
x0 = 0.001*zeros(num_free_dofs(oper.trial))
# x, hist = minres(A, b, M = riesz, itmax=1000, atol=1e-8, rtol=1e-8, verbose=0)
 x, hist = gmres(A, b, M = riesz, itmax=1000, atol=1e-8, rtol=1e-8, verbose=0)
@printf("GMRES converged in %d iterations\n", hist.niter)
@printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs, evals))/minimum(broadcast(abs, evals)))
cond = maximum(abs.(evals)) / minimum(abs.(evals))
cond = round(cond, digits=1)
push!(table, (α=α, ρ0=ρ0, nk=nk, niter=hist.niter, cond_num=cond))
end
end
end
@show(table); nothing;
end  