# Appendix : Darcy--Forchheimer preconditioners


module LinearisedDarcyForchheimerPreconditionerTests
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

  function normu(v,r)
    (v⋅v>0)*(sqrt(v⋅v))^(r-2) 
  end

  function deri(v,r) 
    (v⋅v>0)*(r-2)*(sqrt(v⋅v))^(r-4)*v   
  end

  dim = 2; I = TensorValue(1,0,0,1)

  # NOTE: this is for the **linearised** problem around um. 
  # First we take only the linearisation F|u|^{r-2}u \approx F|um|^{r-2}u = FHm*u, 
  # where FHm=F*(normu∘um), since normu already contains the r-2 power. This still 
  # accommodates u in L3 if we keep the nonlinearity not in Linfty

  u_ex(x)  = VectorValue(0.1*cos(π*x[1])*sin(π*x[2]),
                       -0.1*sin(π*x[1])*cos(π*x[2]))
  p_ex(x)  = sin(π*x[1])*cos(π*x[2]) + exp(π*x[1]*x[2])
  g_ex(x) = (∇⋅u_ex)(x)

  function assemble_linearised_darcy_forchheimer_and_precond(model, u_ex, p_ex, g_ex; k=k, κ=κ, FF=FF, r=r, prec_variant=:prec_variant)
    
    um(x) = u_ex(x) 
    
    # in the simplest case:
    #FHum(x) = FF*I*(x-> normu(um(x),r))(x)
    # if FH has the full linearisation then:  
    FHum(x) = FF*(I*(x-> normu(um(x),r))(x) + (x-> deri(um(x),r))(x)⊗um(x)) 
    
    κinvpFHum(x) = (1/κ*I + FHum(x))
    gradp_ex(x) = ∇(p_ex)(x)
    f_ex(x) = (1.0/κ)*u_ex(x) + gradp_ex(x) + FHum(x)⋅u_ex(x)
    #rprime = 1/(r-1); 

    pweight(x) = (x->inv(FHum(x)))(x) 
    reffe_u = ReferenceFE(raviart_thomas,Float64,k)
    reffe_p = ReferenceFE(lagrangian,Float64,k)

    # FESpaces
    Uh_ = TestFESpace(model,reffe_u,dirichlet_tags = "Gamma_u",conformity=:HDiv)
    Qh_ = TestFESpace(model,reffe_p,conformity=:L2)
  
    Uh = TrialFESpace(Uh_,um)
    Qh = TrialFESpace(Qh_)
  
    Yh = MultiFieldFESpace([Uh_,Qh_])
    Xh = MultiFieldFESpace([Uh,Qh])

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*(k+2))

    Λ = SkeletonTriangulation(model)
    dΛ = Measure(Λ,2*(k+2)-1)
    h_e = CellField(get_array(∫(1)dΛ),Λ)
    
    Γp = BoundaryTriangulation(model, tags = "Gamma_p")
    dΓp = Measure(Γp,2*(k+2)-1)
    n_Γp = get_normal_vector(Γp)
    h_e_Γp = CellField(get_array(∫(1)dΓp),Γp)

    a(u,v)   = ∫( (1/κ)*(u⋅v))*dΩ
    d1(u,v) = ∫((FHum⋅u)⋅v)dΩ
    b(v,q)   = ∫(-(∇⋅v)*q)dΩ
    F(v)     = ∫(f_ex⋅v)dΩ -∫(p_ex*(v⋅n_Γp))dΓp
    G(q)     = ∫((-1.0)*(g_ex*q))dΩ

    
    lhs((u, p),(v, q)) = a(u,v) + d1(u,v) +  
                         b(v,p) + b(u,q)  
  
    rhs((v, q)) = F(v) + G(q)

    op = AffineFEOperator(lhs,rhs,Xh,Yh)
   
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

      A11_,A22a_,A22b_,A22c_ = assemble_matrix(a11_B1,Uh,Uh_),assemble_matrix(a22a_B1,Qh,Qh_),assemble_matrix(a22b_B1,Qh,Qh_),assemble_matrix(a22c_B1,Qh,Qh_)
      A22=LinearOperator(GridapLinearSolverPreconditioner(A22a_))+
             LinearOperator(GridapLinearSolverPreconditioner(A22b_)) +
             LinearOperator(GridapLinearSolverPreconditioner(A22c_))

      A22ainv=inv(Array(A22a_));
      A22binv=inv(Array(A22b_));  
      A22cinv=inv(Array(A22c_));  

      Pinv[range2,range2]=A22ainv+A22binv+A22cinv;         
    
    else #For B2 we take the simpler Riesz map preconditioner for p, 
      # which is spectrally equivalent to the more complex one in B1
      # We need to contract with I/dim in the divergence and pressure 
      # terms because the weight is tensorial

      a11_B2(u,v)=∫((κinvpFHum⋅u)⋅v + (x->κinvpFHum(x)⊙I/dim)*(∇⋅u)*(∇⋅v))dΩ
      a22_B2(p,q)= ∫((x->inv(κinvpFHum(x))⊙I/dim)*p*q)dΩ

      A11_,A22_= assemble_matrix(a11_B2,Uh,Uh_),assemble_matrix(a22_B2,Qh,Qh_)  
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

  # run now the tests 
  table = DataFrame(nk=Int[], r=Float64[], κ=Float64[], FF=Float64[], niter=Int[], cond_num=Float64[])
  
  for nk in (2,3,4)

    model = generate_model2d(nk)
    setup_model_labels_unit_square!(model)
    # for r in (3,3.5,4)
    r=3.0
      for κ in (1e-10, 1e-5, 1)
        for FF in (1e-10, 1, 1e5)
          println("\n--- Running for nk=$nk, r=$r, kappa=$κ, FF=$FF ---")

          oper,riesz,evals = assemble_linearised_darcy_forchheimer_and_precond(model, u_ex, p_ex, g_ex; k=0, κ=κ, FF=FF, r=r, prec_variant=:B2)
          
          A = oper.op.matrix
          b = oper.op.vector
          x0 = 0.001*zeros(num_free_dofs(oper.trial))
          x, hist = minres(A, b, M = riesz, itmax=1000, atol=1e-8, rtol=1e-8, verbose=0)
          @printf("MINRES converged in %d iterations\n", hist.niter)
          @printf("condition number of the preconditioned system: %1.3e\n", maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals)))
          push!(table,(r=r,κ=κ,FF=FF,nk=nk,niter=hist.niter,cond_num=maximum(broadcast(abs,evals))/minimum(broadcast(abs,evals))))
      
        end
      # end
    end  

  end

  @show(table) 
end
