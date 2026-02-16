
function generate_model2d(nk)
  domain =(0,1,0,1)
  n      = 2^nk #+ 1
  # Discrete model
  partition = (n,n)
  CartesianDiscreteModel(domain, partition) |> simplexify
end

function generate_rectangle(nk)
  domain =(0,2,0,1)
  n      = 2^nk + 1
  # Discrete model
  partition = (2*n,n)
  CartesianDiscreteModel(domain, partition) |> simplexify
end

function setup_model_labels_unit_square!(model)
  labels = get_face_labeling(model)
  add_tag!(labels,"Gamma_p",[6,2,3,4,8]) #6top, right, and other 3 corners
  add_tag!(labels,"Gamma_u",[1,2,3,5,7]) # bottom and 7left and low-left corner
end  

function setup_model_labels_rectangle_channel!(model)
  labels = get_face_labeling(model)
  add_tag!(labels,"Gamma_out",[8]) #right 
  add_tag!(labels,"Gamma_in",[7,1,3]) # left and 2 corners
  add_tag!(labels,"Gamma_topbot",[1,2,3,4,5,6]) # corners bottom and top
end  

 function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(4-2)
  end


function assemble_darcy(model, k, κ, κF, FF, f_rhs, g_rhs, u_ex, p_ex)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)

  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags="Gamma_u",conformity=:HDiv)
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  
  Uh = TrialFESpace(Uh_,u_ex)
  Qh = TrialFESpace(Qh_)
  # Lh = TrialFESpace(Lh_)

  Yh = MultiFieldFESpace([Uh_,Qh_])
  Xh = MultiFieldFESpace([Uh,Qh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  dΓ = Measure(Γ,2*(k+2)-1)
  n_Γ = get_normal_vector(Γ)

  # a((u, p),(v, q)) = ∫( (1/κ)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ 
  # b((v, q)) = ∫((1/κ)*(u_ex⋅v))*dΩ + ∫(∇(p_ex)⋅v)*dΩ - ∫(q*(∇⋅u_ex))*dΩ
  a((u, p),(v, q)) = ∫(κF*(v⋅u))dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ  #+ (FF)*(4-2)*∫((x -> (normu(u_ex(x)))^(4-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ#+ ∫(FF*v⋅((normu∘(u))*u))*dΩ + ∫(FF*v⋅((normu∘(u))*u))*dΩ#
  b((v, q)) = ∫(f_rhs⋅v)*dΩ -∫(p_ex*(v⋅n_Γ))dΓ - ∫(q*g_rhs)*dΩ

  # Build affine FE operator
  op = AffineFEOperator(a,b,Xh,Yh)
 op,dΩ
end
function compute_errors_darcy(xh, dΩ, κ, u_ex, p_ex)
  uh, ph = xh
  euh = u_ex - uh
  # error_u = sqrt(∫((1.0/κ)*(∇⋅euh)*(∇⋅euh))*dΩ)
  error_u = (sum(∫((euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
  error_p = sqrt(sum(∫(κ*(p_ex-ph)*(p_ex-ph))*dΩ)) 
  error_u,error_p
end


grav = VectorValue(0.0,1.0)

function transport(ϕ)
    grav*(1.0+ϕ)
end

# function build_analytical_functions(κ, κF, FF, u_ex, p_ex, ζ_ex, ϕ_ex)
#   f_ex(x) = (1.0/κ)*u_ex(x) + ∇(p_ex)(x) + FF*u_ex(x)*(normu∘u_ex)(x) - (transport∘ϕ_ex)(x)
#   g_ex(x) = (∇⋅u_ex)(x)
#   m_ex(x) =  (1.0-g_ex(x))*ϕ_ex(x) - (∇⋅ζ_ex)(x)

#   return f_ex, g_ex, m_ex
# end

function assemble_df_ard(model, k, κ, κF, FF, f_ex, g_ex, m_ex, u_ex, p_ex, ζ_ex, ϕ_ex)
  # f_ex, g_ex, m_ex = build_analytical_functions(κ, κF, FF, u_ex, p_ex, ζ_ex, ϕ_ex)
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

    a1(u,v)   = ∫( (κF)*(u⋅v))*dΩ
    aF2(u,v) = (FF)*(4-2)*∫((x -> (normu(u_ex(x)))^(4-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ
    b1(v,q)   = ∫(-(∇⋅v)*q)*dΩ
    F0(v,ϕ)  = ∫((transport∘ϕ)⋅v)*dΩ
    F0_lin(v,ϕ) = ∫(1.0 * (grav⋅v) * ϕ)*dΩ

    F(v)     = ∫(f_ex⋅v)*dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫(g_ex*q)dΩ
    
    a2(ζ,τ) = ∫( (1.0)*(ζ⋅τ))*dΩ
    b2(τ,ψ) = ∫((∇⋅τ)*ψ)*dΩ
    c2(ϕ,ψ) = ∫((ϕ * ψ))*dΩ - ∫(g_ex*(ϕ * ψ))*dΩ
    e(u,ϕ,τ) = ∫((1.0)*(u⋅τ) * ϕ)*dΩ  
    M(τ) = ∫(ϕ_ex*(τ⋅n_Γu))dΓu
    H(ψ) = ∫(m_ex*ψ)dΩ

    resi((u,p,ζ,ϕ),(v,q,τ,ψ)) =  a1(u,v) + b1(v,p) +
                                 b1(u,q) + 
                                 a2(ζ,τ) + b2(τ,ϕ) +
                                  b2(ζ,ψ) - c2(ϕ,ψ)
                          

    jaco((v,q,τ,ψ)) = F(v) - G(q) + M(τ) - H(ψ) - ∫((1.0)*(u_ex⋅τ) * ϕ_ex)*dΩ + ∫((transport∘ϕ_ex)⋅v)*dΩ 

  op = AffineFEOperator(resi,jaco,Xh,Yh)
  op,dΩ
end

function compute_errors_df_ard(xh, dΩ, κ, u_ex, p_ex, ζ_ex, ϕ_ex)
    uh, ph, ζh, ϕh = xh
    euh = u_ex - uh
    eph = p_ex - ph
    eζh = ζ_ex - ζh
    eϕh = ϕ_ex - ϕh
    
    error_u = (sum(∫((euh⋅euh).^(0.5*3.0))dΩ))^(1.0/3.0) + sqrt(sum(∫((∇⋅euh)*(∇⋅euh))*dΩ))
    error_p = sqrt(sum(∫(eph*eph)*dΩ))
    error_ζ =  (sum(∫(((∇⋅eζh)*(∇⋅eζh)).^(0.5*6.0/5.0))dΩ))^(5.0/6.0) + sqrt(sum(∫(eζh⋅eζh)dΩ))
    error_ϕ = (sum(∫((eϕh*eϕh).^(0.5*6.0))dΩ))^(1.0/6.0)
    # error_ζ =  sqrt(sum(∫(eζh⋅eζh)dΩ) + sum(∫((∇⋅eζh)*(∇⋅eζh))*dΩ))
    # error_ϕ = sqrt(sum(∫(eϕh*eϕh)*dΩ))
    error_u,error_p,error_ζ,error_ϕ
end

function assemble_darcy_forchheimer_noExact(model, k, κ, FF, f_rhs, g_rhs)
  reffe_u = ReferenceFE(raviart_thomas,Float64,k)
  reffe_p = ReferenceFE(lagrangian,Float64,k)
function normu(v)
    (v⋅v>0)*(sqrt(v⋅v))^(3-2)
  end
  # FESpaces
  Uh_ = TestFESpace(model,reffe_u,dirichlet_tags = "Gamma_u",conformity=:HDiv)#,dirichlet_tags="boundary")
  Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  
  Uh = TrialFESpace(Uh_,VectorValue(0,0))
  Qh = TrialFESpace(Qh_)
  
  Yh = MultiFieldFESpace([Uh_,Qh_])
  Xh = MultiFieldFESpace([Uh,Qh])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+2))
  Γ = BoundaryTriangulation(model, tags = "Gamma_p")
  n_Γ = get_normal_vector(Γ)
  dΓ = Measure(Γ,2*(k+2)-1)


  a((u, p),(v, q)) = ∫( (1/κ + FF)*(u⋅v) )*dΩ - ∫(q*(∇⋅u))*dΩ - ∫((∇⋅v)*p)*dΩ #+ ∫(FF*v⋅((normu∘(u))*u))*dΩ
  b((v, q)) = ∫(f_rhs⋅v)*dΩ - ∫(q*g_rhs)*dΩ

  # Build affine FE operator
  op = AffineFEOperator(a,b,Xh,Yh)
 op#, dΩ
end


function assemble_darcy_forchheimer_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF; prec_variant=:prec_variant)

   X1,X2 = op.trial
   Y1,Y2 = op.test 


  @assert prec_variant in (:B1,:B2)
  if (prec_variant==:B1)
    a11_B1(u,v)=∫((1.0/κ + FF)*(u⋅v) + (1.0/κ + FF)*(∇⋅u)*(∇⋅v))dΩ #+ ∫(FF*v⋅((normu∘(u))*u))dΩ
    a22_B1(p,q)= ∫((1.0/(1.0/κ + FF))*p*q)dΩ
    
    A11,A22 = assemble_matrix(a11_B1,X1,Y1),assemble_matrix(a22_B1,X2,Y2)
    
  else
    # a11_B2(u,v) =∫((1.0/κ + FF)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ #+ ∫(FF*v⋅((normu∘(u))*u))dΩ
    #   a22a_B2(p,q)= ∫(p*q)dΩ
    #   a22b_B2(p,q)= ∫((1.0/(1.0/κ + FF))*(∇(p))⋅∇(q))dΩ + ∫(((1.0/(1.0/κ + FF))/h_e)*jump(p)*jump(q))dΛ + ∫(((1.0/(1.0/κ + FF))/h_e_Γ)*p*q)dΓ

      # a11_B2 is the bilinear form corresponding to the velocity blobk in the preconditioner (5.5).
    a11_B2(u,v) =∫((1.0/κ)*(u⋅v)+(∇⋅u)*(∇⋅v))dΩ + (FF)*∫((x -> (normu(u_ex(x)))^(r-2))*(v⋅u))dΩ + (FF)*(r-2)*∫((x -> (normu(u_ex(x)))^(r-4)) * (u_ex ⋅ u)  * (u_ex ⋅ v))dΩ
      # a22a_B2 is the bilinear gorm corresponding to the 1st part of the bilinear form : (I)^{-1}
    a22a_B2(p,q)= ∫(p*q)dΩ
      # a22_B2_1 corresponds to (-κΔ)^{-1}
    a22_B2_1(p,q) = ∫(κ*(∇(p))⋅∇(q))dΩ + ∫((κ/h_e)*jump(p)*jump(q))dΛ + ∫((κ/h_e_Γ)*p*q)dΓ
      # a22_B2_2 corresponds to (-F*norm(u_ex)^{r-2}Δ)^{-1}
    a22_B2_2(p,q) = (1/FF)*∫((x -> normu(u_ex(x))^(2-r))*(∇(p)⋅∇(q)))dΩ  + (1/FF)*∫(((x -> normu(u_ex(x))^(2-r))/h_e)*jump(p)*jump(q))dΛ  + (1/FF)*∫(((x -> normu(u_ex(x))^(2-r))/h_e_Γ)*p*q)dΓ
      # a22_B2_3 corresponds to (-[div(A∇)])^{-1}, where A = F*(r-2)*norm(u_ex)^{r-4}
    a22_B2_3(p,q) = (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((Minv⋅(∇(p))) ⋅ ∇(q)) )dΩ - (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((mean((Minv ⋅ ∇(p)))) ⋅ jump(q*n_Λ)) )dΛ + (1/FF)*(1/(r-2))*∫((x -> normu(u_ex(x))^(4-r))*((mean((Minv ⋅ ∇(q)))) ⋅ jump(p*n_Λ)) )dΛ + ∫(jump(p)*jump(q))dΛ #+ (1/FF)*(1/(r-2))*∫(jump(Minv ⋅ ∇(p) ⋅ n_Γ)*jump(Minv ⋅ ∇(q) ⋅n_Γ))dΓ
    
    a22b_B2(p,q) = a22_B2_1(p,q) + a22_B2_2(p,q) + a22_B2_3(p,q)
     
    A11,A22a,A22b = assemble_matrix(a11_B2,X1,Y1),assemble_matrix(a22a_B2,X2,Y2),assemble_matrix(a22b_B2,X2,Y2)
    return A11,A22a,A22b
  end
end

function compute_errors_robust_darcy_forchheimer(xh, op, dΩ, dΓ, dΛ, h_e, h_e_Γ, κ, FF, u_ex, p_ex)

  blocks=assemble_darcy_forchheimer_riesz_mapping_preconditioner_blocks(op, dΩ, dΓ, dΛ,
                                                         h_e, h_e_Γ,
                                                         κ, FF;
                                                         prec_variant=:B2)

  A11,A22a,A22b=blocks
  A11 = LinearOperator(A11)
  A22 = LinearOperator(A22a)+LinearOperator(A22b)

  A22_inv = LinearOperator(GridapLinearSolverPreconditioner(A22a))+
                LinearOperator(GridapLinearSolverPreconditioner(A22b))
  uh, ph = xh
  eu = u_ex-uh
  ep = p_ex-ph
  X1,X2 = op.trial
  euh=interpolate(eu,X1)
  eph=interpolate(ep,X2)

  freedofs = get_free_dof_values(eph)
  newfreedofs = zeros(length(freedofs))
  auxM = A22_inv
  auxr=copy(freedofs)
  rtol=1.0e-18

  function custom_stopping_condition(solver, A, b, r, tol)
    mul!(r, A, solver.x)
    r .-= b                       # r := b - Ax
    bool = norm(r) ≤ tol*norm(b)  # tolerance based on the 2-norm of the residual
    #@printf("ERR NORM: ||b-Ax||/||b||: %16.7e\n",norm(r)/norm(b))
    return bool
  end
  
  minres_callback(solver) = custom_stopping_condition(solver, auxM, freedofs, auxr, rtol)
  # Using minres here for simplicity, but one could 
  # use pcg as well, as A is SPD
  (newfreedofs, stats) = minres(auxM, freedofs, M=A22; itmax=2000, verbose=1,
                        callback=minres_callback,
                        atol=0.0,
                        rtol=0.0,
                        etol=0.0) 

  weighted_norm = sqrt(dot(newfreedofs,freedofs))

  return compute_operator_norm(A11,euh), weighted_norm #compute_operator_norm(A22,eph),

end
