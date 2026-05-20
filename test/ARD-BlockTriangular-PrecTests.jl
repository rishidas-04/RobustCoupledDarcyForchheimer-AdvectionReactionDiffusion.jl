# Appendix  : Block-triangular preconditioners for the ard system




# module AdvectionReactionDiffusionPreconditionerTests

# using Gridap
# import Gridap: ∇
# using Gridap.Algebra
# using Printf
# using Krylov
# using LinearAlgebra
# using LinearOperators
# using DataFrames
# using Gridap.Geometry: simplexify

# ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2]) + exp(π*x[1]*x[2])
# u_ex(x) = VectorValue(1e-1*cos(π*x[1])*sin(π*x[2]),
#                        1e-1*sin(π*x[1])*cos(π*x[2]))
# # ---------------- Mesh ----------------
# function generate_model2d(nk)
#   domain = (0,1,0,1)
#   n = 2^nk
#   partition = (n,n)
#   CartesianDiscreteModel(domain, partition) |> simplexify
# end

# function setup_model_labels_unit_square!(model)
#   labels = get_face_labeling(model)
#   add_tag!(labels, "Gamma_p", [6,2,3,4,8])
#   add_tag!(labels, "Gamma_u", [1,2,3,5,7])
# end

# # ---------------- Linear Solver Wrapper ----------------
# struct GridapLinearSolverPreconditioner{A,B}
#   numerical_setup::A
#   matrix::B
#   function GridapLinearSolverPreconditioner(matrix; ls=LUSolver())
#     ss = symbolic_setup(ls, matrix)
#     ns = numerical_setup(ss, matrix)
#     new{typeof(ns), typeof(matrix)}(ns, matrix)
#   end
# end

# function LinearAlgebra.mul!(z, M::GridapLinearSolverPreconditioner, r)
#   Gridap.solve!(z, M.numerical_setup, r)
# end

# function LinearOperators.LinearOperator(solver::GridapLinearSolverPreconditioner)
#   function apply(res, x, α, β)
#     tmp = similar(x)
#     mul!(tmp, solver, x)
#     if β ≈ 0.0
#       res .= tmp
#     else
#       res .= β .* res .+ α .* tmp
#     end
#   end
#   LinearOperator(Float64, size(solver.matrix,1), size(solver.matrix,2), true, true, apply)
# end

# # ---------------- Triangular Preconditioner ----------------
# function TriangularPreconditioner(A11inv, A22inv, F, range1, range2, N)

#   function apply(res, x, α, β)

#     r1 = view(x, range1)
#     r2 = view(x, range2)

#     z = zeros(eltype(x), N)

#     # Step 1: solve Schur
#     z2 = A22inv * r2

#     # Step 2: correction
#     r1_hat = r1 - F * z2

#     # Step 3: solve A11
#     z1 = A11inv * r1_hat

#     z[range1] .= z1
#     z[range2] .= z2

#     if β ≈ 0.0
#       res .= z
#     else
#       res .= β .* res .+ α .* z
#     end
#   end

#   LinearOperator(Float64, N, N, true, true, apply)
# end

# # ---------------- Assembly ----------------
# function assemble_precond(model; k=0, α=α, ρ0=ρ0)
# ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
# m_ex(x) = (-1.0)*(∇⋅ζ_ex)(x) + ρ0*ϕ_ex(x)
#   reffe_ζ = ReferenceFE(raviart_thomas, Float64, k)
#   reffe_ϕ = ReferenceFE(lagrangian, Float64, k)

#   Wh_ = TestFESpace(model, reffe_ζ, dirichlet_tags="Gamma_p", conformity=:HDiv)
#   Ψh_ = TestFESpace(model, reffe_ϕ, conformity=:L2)

#   Wh = TrialFESpace(Wh_, x->zero(VectorValue{2,Float64}))
#   Ψh = TrialFESpace(Ψh_)

#   Yh = MultiFieldFESpace([Wh_, Ψh_])
#   Xh = MultiFieldFESpace([Wh, Ψh])

#  Ω = Triangulation(model)
# dΩ = Measure(Ω, 2*(k+2))

# Λ = SkeletonTriangulation(model)
# dΛ = Measure(Λ, 2*(k+2)-1)
# h_e = CellField(get_array(∫(1)dΛ), Λ)

# Γu = BoundaryTriangulation(model, tags = "Gamma_u")
# dΓu = Measure(Γu, 2*(k+2)-1)
# n_Γu = get_normal_vector(Γu)
# Γp = BoundaryTriangulation(model, tags = "Gamma_p")
# dΓp = Measure(Γp, 2*(k+2)-1)
# n_Γp = get_normal_vector(Γp)
# Λ = SkeletonTriangulation(model)
# dΛ = Measure(Λ, 2*(k+2)-1)
# n_Λ = get_normal_vector(Λ)
# h_e = CellField(get_array(∫(1)dΛ), Λ)
# h_e_Γu = CellField(get_array(∫(1)dΓu), Γu)
# h_e_Γp = CellField(get_array(∫(1)dΓp), Γp)


#   # Bilinear forms
  
# a(ζ, τ) = ∫((1.0/α)*(ζ⋅τ))*dΩ
# b(τ, ψ) = ∫((∇⋅τ)*ψ)*dΩ
# c(ϕ,ψ) = ∫((ρ0)*ϕ*ψ)*dΩ
# e(ϕ, τ) = ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ
# H(τ) = ∫(ϕ_ex*(τ⋅n_Γu))*dΓu # + ∫(h_ex⋅τ)*dΩ
# M(ψ) = ∫(-(m_ex*ψ))*dΩ

# lhs((ζ, ϕ), (τ, ψ)) = a(ζ, τ) + b(τ, ϕ) + e(ϕ, τ) +
#                       b(ζ, ψ) - c(ϕ, ψ)

# rhs((τ, ψ)) = H(τ) + M(ψ)

#   op = AffineFEOperator(lhs, rhs, Xh, Yh)
# ufield = CellField(u_ex, Ω)
# hK = CellField(get_array(∫(1)dΩ), Ω)
# # hK = CellDiameter(Ω)
# unorm = CellField(x -> norm(u_ex(x))^2, Ω)
# β = ufield⋅n_Λ;
# βu = ufield⋅n_Γu; βp = ufield⋅n_Γp
# # τK = CellField(x -> unorm(x) *(1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2)), Ω)
# τK = CellField(x -> (1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2)), Ω)

#   A = op.op.matrix
#   bvec = op.op.vector

#   # Block sizes
#   Ns = [num_free_dofs(U) for U in op.trial]
#   Np = cumsum([1; Ns])
#   range1 = Np[1]:Np[2]-1
#   range2 = Np[2]:Np[3]-1
#   N = size(A,1)

#   # ---------------- A11 ----------------
#   a11(ζ, τ) = ∫((1.0/α)*(ζ⋅τ) + (∇⋅ζ)*(∇⋅τ))*dΩ
#   A11_ = assemble_matrix(a11, Wh, Wh_)
#   A11inv = inv(Array(A11_))

#   # ---------------- A22 (Schur approx) ----------------
#   # a22(ϕ, ψ) = ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
#   #              ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
#   #              ∫((α/h_e_Γu)*ϕ*ψ)dΓu - 
#   #              ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ -
#   #              ∫(α*((∇(ψ))⋅n_Γu)*ϕ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
#   #              ∫((u_ex⋅∇(ϕ))*ψ)dΩ +
#   #              ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ + #skeleton central
#   #              ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ + #skeleton upwind
#   #              ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu + #boundary upwind
#   #              ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp + #boundary upwind
#   #             #  ∫(((u_ex*ϕ)⋅(u_ex*ψ)))dΩ +
#   #              ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ
#   # A22_ = assemble_matrix(a22, Ψh, Ψh_)
#   # A22inv = inv(Array(A22_))


# a22a_B2(ϕ, ψ) = ∫((α+unorm+ρ0)*ϕ*ψ)dΩ
# # a22b_B2(ϕ, ψ) =∫(ρ0 * ϕ * ψ)*dΩ +  ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ - ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
# #                 ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
# #                 ∫((α/h_e_Γu)*ϕ*ψ)dΓu + ∫((u_ex⋅∇(ϕ))*ψ)dΩ + ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ + #skeleton central
# #                 ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ + #skeleton upwind
# #                 ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu + #boundary upwind
# #                 ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp + ∫((u_ex*ϕ)⋅(u_ex*ψ))dΩ

# a22b_B2(ϕ, ψ) = ∫(ρ0*ϕ*ψ)*dΩ+∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
#                ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
#                ∫((α/h_e_Γu)*ϕ*ψ)dΓu - 
#                ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ -
#                ∫(α*((∇(ψ))⋅n_Γu)*ϕ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
#                ∫((u_ex⋅∇(ϕ))*ψ)dΩ +  # -∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ -
#               #  ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ +
#                ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ + #skeleton central
#                ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ + #skeleton upwind
#                ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu + #boundary upwind
#                ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp + #boundary upwind
#               #  ∫(((u_ex*ϕ)⋅(u_ex*ψ)))dΩ +
#                ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ

# A11_, A22a_, A22b_ = assemble_matrix(a11, Wh, Wh_), assemble_matrix(a22a_B2, Ψh, Ψh_), assemble_matrix(a22b_B2, Ψh, Ψh_)
# A22 = LinearOperator(GridapLinearSolverPreconditioner(A22a_)) +
# LinearOperator(GridapLinearSolverPreconditioner(A22b_))

# A22ainv = inv(Array(A22a_))
# A22binv = inv(Array(A22b_))
# A22inv = A22binv + A22ainv


#   # ---------------- (1,2) BLOCK ----------------
#   a12(ϕ, τ) = ∫((∇(ϕ))⋅τ)*dΩ + ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ
#   F_ = assemble_matrix(a12, Ψh, Wh_)
#   F = Array(F_)

#   # ---------------- Preconditioner ----------------
#   P = TriangularPreconditioner(A11inv, A22inv, F, range1, range2, N)

#   return A, bvec, P
# end

# # ---------------- Driver ----------------
# table = DataFrame(nk=Int[], α=Float64[], ρ0=Float64[], niter=Int[])

# for nk in (2,3,4)
#   model = generate_model2d(nk)
#   setup_model_labels_unit_square!(model)

#    for α in (1e-9, 1e-6, 1e-3, 1)
#     for ρ0 in (1e-3, 1e-2, 1e-1, 1)
#       println("\n--- nk=$nk α=$α ρ0=$ρ0 ---")

#       A, b, P = assemble_precond(model; k=0, α=α, ρ0=ρ0)

#       x, hist = gmres(A, b, M=P, atol=1e-8, rtol=1e-8)

#       println("GMRES iterations: ", hist.niter)

#       push!(table, (nk=nk, α=α, ρ0=ρ0, niter=hist.niter))
#     end
# # α = 1e-6
#     # println("\n--- nk=$nk α=$α ρ0=$ρ0 ---")

#     # A, b, P = assemble_precond(model; k=0, α=α, ρ0=ρ0)

#     # x, hist = gmres(A, b, M=P, atol=1e-8, rtol=1e-8)

#     # println("GMRES iterations: ", hist.niter)

#     push!(table, (nk=nk, α=α, ρ0=ρ0, niter=hist.niter))
#   end
# end

# println(table)

# end


module AdvectionReactionDiffusionPreconditionerTests

using Gridap
import Gridap: ∇
using Gridap.Algebra
using Printf
using Krylov
using LinearAlgebra
using LinearOperators
using DataFrames
using Gridap.Geometry: simplexify
using Arpack

ϕ_ex(x) = cos(2π*x[1])*cos(2π*x[2]) + exp(π*x[1]*x[2])
u_ex(x) = VectorValue(1e-1*cos(π*x[1])*sin(π*x[2]),
                       -1e-1*sin(π*x[1])*cos(π*x[2]))

# ---------------- Mesh ----------------
function generate_model2d(nk)
  domain = (0,1,0,1)
  n = 2^nk
  partition = (n,n)
  CartesianDiscreteModel(domain, partition) |> simplexify
end

function setup_model_labels_unit_square!(model)
  labels = get_face_labeling(model)
  add_tag!(labels, "Gamma_p", [6,2,3,4,8])
  add_tag!(labels, "Gamma_u", [1,2,3,5,7])
end

# ---------------- Linear Solver Wrapper ----------------
struct GridapLinearSolverPreconditioner{A,B}
  numerical_setup::A
  matrix::B
  function GridapLinearSolverPreconditioner(matrix; ls=LUSolver())
    ss = symbolic_setup(ls, matrix)
    ns = numerical_setup(ss, matrix)
    new{typeof(ns), typeof(matrix)}(ns, matrix)
  end
end

function LinearAlgebra.mul!(z, M::GridapLinearSolverPreconditioner, r)
  Gridap.solve!(z, M.numerical_setup, r)
end

function LinearOperators.LinearOperator(solver::GridapLinearSolverPreconditioner)
  function apply(res, x, α, β)
    tmp = similar(x)
    mul!(tmp, solver, x)
    if β ≈ 0.0
      res .= tmp
    else
      res .= β .* res .+ α .* tmp
    end
  end
  LinearOperator(Float64, size(solver.matrix,1), size(solver.matrix,2), true, true, apply)
end

# ---------------- Triangular Preconditioner ----------------
function TriangularPreconditioner(A11inv, A22inv, F, range1, range2, N)

  function apply(res, x, α, β)

    r1 = view(x, range1)
    r2 = view(x, range2)

    z = zeros(eltype(x), N)

    z2 = A22inv * r2
    r1_hat = r1 - F * z2
    z1 = A11inv * r1_hat

    z[range1] .= z1
    z[range2] .= z2

    if β ≈ 0.0
      res .= z
    else
      res .= β .* res .+ α .* z
    end
  end

  LinearOperator(Float64, N, N, true, true, apply)
end

# ---------------- Assembly ----------------
function assemble_precond(model; k=0, α=1e-3, ρ0=1.0)

ζ_ex(x) = α*∇(ϕ_ex)(x) - u_ex(x) * ϕ_ex(x)
m_ex(x) = (-1.0)*(∇⋅ζ_ex)(x) + ρ0*ϕ_ex(x)

  reffe_ζ = ReferenceFE(raviart_thomas, Float64, k)
  reffe_ϕ = ReferenceFE(lagrangian, Float64, k)

  Wh_ = TestFESpace(model, reffe_ζ, dirichlet_tags="Gamma_p", conformity=:HDiv)
  Ψh_ = TestFESpace(model, reffe_ϕ, conformity=:L2)

  Wh = TrialFESpace(Wh_, x->zero(VectorValue{2,Float64}))
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

  n_Λ = get_normal_vector(Λ)
  h_e_Γu = CellField(get_array(∫(1)dΓu), Γu)
  h_e_Γp = CellField(get_array(∫(1)dΓp), Γp)

  # Bilinear forms
  a(ζ, τ) = ∫((1.0/α)*(ζ⋅τ))*dΩ
  b(τ, ψ) = ∫((∇⋅τ)*ψ)*dΩ
  c(ϕ,ψ) = ∫((ρ0)*ϕ*ψ)*dΩ
  e(ϕ, τ) = ∫((1.0/α)*(u_ex⋅τ)*ϕ)*dΩ

  H(τ) = ∫(ϕ_ex*(τ⋅n_Γu))*dΓu
  M(ψ) = ∫(-(m_ex*ψ))*dΩ

  lhs((ζ, ϕ), (τ, ψ)) = a(ζ, τ) + b(τ, ϕ) + e(ϕ, τ) +
                        b(ζ, ψ) - c(ϕ, ψ)

  rhs((τ, ψ)) = H(τ) + M(ψ)

  op = AffineFEOperator(lhs, rhs, Xh, Yh)

  ufield = CellField(u_ex, Ω)
  hK = CellField(get_array(∫(1)dΩ), Ω)
  unorm = CellField(x -> norm(u_ex(x))^2, Ω)

  β = ufield⋅n_Λ
  βu = ufield⋅n_Γu
  βp = ufield⋅n_Γp

  τK = CellField(x -> (1.0 / sqrt((2*norm(u_ex(x))/hK(x))^2 + (4*α/(hK(x)^2))^2)), Ω)

  A = op.op.matrix
  bvec = op.op.vector

  Ns = [num_free_dofs(U) for U in op.trial]
  Np = cumsum([1; Ns])

  range1 = Np[1]:Np[2]-1
  range2 = Np[2]:Np[3]-1
  N = size(A,1)

  # A11
  a11(ζ, τ) = ∫((1.0/α + unorm)*(ζ⋅τ) + (∇⋅ζ)*(∇⋅τ))*dΩ
  A11_ = assemble_matrix(a11, Wh, Wh_)
  A11inv = inv(Array(A11_))

  # A22
  a22a_B2(ϕ, ψ) = ∫((1.0+ρ0)*ϕ*ψ)*dΩ

  a22b_B2(ϕ, ψ) = ∫((ρ0)*ϕ*ψ)*dΩ + ∫(α*(∇(ϕ))⋅∇(ψ))*dΩ +
                 ∫((α/h_e)*jump(ϕ)*jump(ψ))dΛ +
                 ∫((α/h_e_Γu)*ϕ*ψ)dΓu -
                 ∫(α*((∇(ϕ))⋅n_Γu)*ψ)dΓu - ∫(α*mean(∇(ϕ))⋅jump(ψ*n_Λ))dΛ -
                 ∫(α*((∇(ψ))⋅n_Γu)*ϕ)dΓu - ∫(α*mean(∇(ψ))⋅jump(ϕ*n_Λ))dΛ +
                 ∫((u_ex⋅∇(ϕ))*ψ)dΩ +
                 ∫((0.5*(β.⁻))*(mean(ϕ)*jump(ψ)))dΛ +
                 ∫((0.5*abs(β.⁺))*(jump(ϕ)*jump(ψ)))dΛ +
                 ∫((0.5*abs(βu))*(ϕ*ψ))*dΓu +
                 ∫((0.5*abs(βp))*(ϕ*ψ))*dΓp +
                 ∫(τK*(u_ex⋅(∇(ϕ)))*(u_ex⋅(∇(ψ))))dΩ

  A22a_ = assemble_matrix(a22a_B2, Ψh, Ψh_)
  A22b_ = assemble_matrix(a22b_B2, Ψh, Ψh_)

  A22ainv = inv(Array(A22a_))
  A22binv = inv(Array(A22b_))
  A22inv = A22binv #+ A22ainv

  # (1,2) block
  a12(ϕ, τ) = ∫((1.0/(α))*(u_ex⋅τ)*ϕ)*dΩ
  F = Array(assemble_matrix(a12, Ψh, Wh_))

  P = TriangularPreconditioner(A11inv, A22inv, F, range1, range2, N)

  return A, bvec, P
end
# ---------------- Preconditioned Operator ----------------
function build_preconditioned_operator(A, P)
  N = size(A,1)

  function apply(res, x, α, β)
    tmp = A * x             # Ax
    tmp2 = similar(x)
    mul!(tmp2, P, tmp)      # P^{-1}(Ax)

    if β ≈ 0.0
      res .= tmp2
    else
      res .= β .* res .+ α .* tmp2
    end
  end

  LinearOperator(Float64, N, N, true, true, apply)
end
# ---------------- Driver ----------------
table = DataFrame(nk=Int[], α=Float64[], ρ0=Float64[], niter=Int[])

for nk in (2,3,4)
  model = generate_model2d(nk)
  setup_model_labels_unit_square!(model)

  # for α in (1e-9, 1e-6, 1e-3, 1)
  α = 1e-8
    for ρ0 in (1e-6, 1)

      println("\n--- nk=$nk α=$α ρ0=$ρ0 ---")

      A, b, P = assemble_precond(model; k=0, α=α, ρ0=ρ0)

      x, hist = gmres(A, b, M=P, atol=1e-8, rtol=1e-8)

      println("GMRES iterations: ", hist.niter)
      # ---- Condition number estimation (robust: uses singular values) ----
using Arpack

Mop = build_preconditioned_operator(A, P)

try
  # Largest singular value of M
  σmax = svds(Mop, nsv=1, which=:LM)[1][1]

  # Smallest singular value of M
  σmin = svds(Mop, nsv=1, which=:SM)[1][1]

  κ_est = σmax / σmin

  println("Estimated κ₂(P^{-1}A) ≈ ", κ_est)
  println("σmax ≈ ", σmax, ", σmin ≈ ", σmin)
catch err
  println("SVD-based estimation failed: ", err)
end
      push!(table, (nk=nk, α=α, ρ0=ρ0, niter=hist.niter))
    end
  # end
end

println(table)

end