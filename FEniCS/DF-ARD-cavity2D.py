'''


            
'''

from fenics import *
from numpy import cos , pi


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["allow_extrapolation"]= True




fileO = XDMFFile("out-2D-cavity.xdmf")
fileO.parameters["functions_share_mesh"] = True
fileO.parameters['rewrite_function_mesh'] = True
fileO.parameters["flush_output"] = True


PETScOptions.set("mat_mumps_icntl_4", 1)
PETScOptions.set("mat_mumps_icntl_14", 400)


# ********* Model coefficients and parameters ********* #

kappa = Constant(1.) # inverse permeability
alpha = Constant(1.)
F     = Constant(20.)
phitop = Constant(20.)

def ff(phi):
    return 10*phi*Constant((1,1))

def forch(u):
    return conditional(gt(dot(u,u),0),sqrt(dot(u,u)),0)

#mesh = UnitSquareMesh(60,60)

def mymesh(lx,ly, Nx,Ny):
    m = UnitSquareMesh(Nx, Ny,'crossed')
    x = m.coordinates()

    #Refine on top and bottom sides
    x[:] = (x - 0.5) * 2.
    x[:] = 0.5 * (cos(pi * (x - 1.) / 2.) + 1.)   

    #Scale
    x[:,0] = x[:,0]*lx
    x[:,1] = x[:,1]*ly

    return m

mesh = mymesh(1,1,64,64)


n = FacetNormal(mesh)

facet_mesh = MeshFunction('size_t', mesh, 1, 0)

top = 31
rest = 32
boundaries = {rest: CompiledSubDomain('near(x[0], 0) || near(x[0], 1) || near(x[1], 0)'),
              top: CompiledSubDomain('near(x[1], 1)')}

[subb.mark(facet_mesh, tag) for tag, subb in boundaries.items()]

ds = Measure('ds', domain=mesh, subdomain_data=facet_mesh)

    
# ********* Finite dimensional spaces ********* #
    
k = 0 
Vh = FiniteElement("RT", mesh.ufl_cell(), k+1)
Qh = FiniteElement("DG", mesh.ufl_cell(), k)

Wh = FiniteElement("RT", mesh.ufl_cell(), k+1)
Psih = FiniteElement("DG", mesh.ufl_cell(), k)
    
# Mixed space
Hh = FunctionSpace(mesh,MixedElement([Vh,Qh,Wh,Psih]))

# test and trial functions for product space
v, q, w, psi = TestFunctions(Hh)
sol = Function(Hh)
dsol = TrialFunction(Hh)
u, p, zeta, phi = split(sol)

print("....... DoF = ", Hh.dim())
    
# ******** Weak forms ************* #
# - rho0*phi*psi*dx

Resid = 1.0/kappa * dot(u,v) * dx \
    + F*forch(u)*dot(u,v) * dx \
    - p * div(v) * dx - q * div(u) * dx \
    - dot(ff(phi),v)* dx \
    + 1/alpha*dot(zeta,w)* dx \
    + phi*div(w)*dx + psi*div(zeta)*dx \
    - phitop*dot(w,n)*ds(top)

# ********* Boundaries and boundary conditions ********* #
bcU = DirichletBC(Hh.sub(0), Constant((0,0)), 'on_boundary')
bcs = [bcU]

# ********* Solving 
Tang = derivative(Resid,sol,dsol)
solve(Resid==0, sol, J = Tang, bcs=bcs, \
          solver_parameters={'newton_solver':{'linear_solver':'mumps',\
                                               'absolute_tolerance':1.0e-6,\
                                               'relative_tolerance':1.0e-6,\
                                               'maximum_iterations':20}})
        
u_h,p_h,zeta_h,phi_h = sol.split()
    
u_h.rename("u","u")
p_h.rename("p","p")
zeta_h.rename("zeta","zeta")
phi_h.rename("phi","phi")

fileO.write(phi_h, 0.)
fileO.write(zeta_h, 0.)
fileO.write(p_h, 0.)
fileO.write(u_h, 0.)    
