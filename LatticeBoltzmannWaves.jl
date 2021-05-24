using Plots, Einsum

# Simulation parameters
const Nx          = 401    # resolution x-dir
const Ny          = 401    # resolution y-dir
const rho0        = 1000    # average density
const Nt          = 600   # number of timesteps
const NL          = 9       # D2Q9 Lattice
const tau         = 0.6    # collision timescale

const cxs = [0, 0, 1, 1, 1, 0,-1,-1,-1]
const cys = [0, 1, 1, 0,-1,-1,-1, 0, 1]
const opp = [1,6,7,8,9,2,3,4,5] # bounce back array, opposite direction
const weights = [4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36] # sums to 1

##build the struct that will hold the state of the LB system##
mutable struct LatticeState <: Function
    rho::Array{Float32, 2}    # macroscale density
    ux::Array{Float32, 2}    # macroscale velocity, x component
    uy::Array{Float32, 2}    # macroscale velocity, y component
    F::Array{Float32, 3}    # particle distribution function
    Feq::Array{Float32, 3}    # forcing term for collisioin
    tmp::Array{Float32, 2}   # place-holder array

    function LatticeState(Nx, Ny) #constructor
        self = new()

        self.F = fill(1.f0, (Nx,Ny,NL))
        self.tmp = fill(1.f0, (Nx,Ny))
        self.ux = fill(0.f0, (Nx,Ny))
        self.uy = fill(0.f0, (Nx,Ny))
        self.rho = fill(0.f0, (Nx,Ny))

        sum!(self.rho, self.F)
        calcU!(self.ux, self.uy, self.F, self.rho)

        self.Feq = zeros(Nx,Ny,NL)
        return self
    end
end

@fastmath @inbounds function applyDrift!(tmp::Array{Float32},F::Array{Float32})
    for i in 1:size(F)[3]
        A = @view F[:,:,i]
        circshift!(tmp,A, (cxs[i],0) )
        circshift!(A,tmp, (0,cys[i]) )
    end
end

@fastmath @inbounds function calcFeq!(Feq::Array{Float32},rho::Array{Float32},ux::Array{Float32},uy::Array{Float32})
    @einsum Feq[i,j,k]= rho[i,j]*weights[k]*(1. +  3. * (cxs[k] * ux[i,j] +  cys[k] * uy[i,j]) + 4.5*(cxs[k] * ux[i,j] +  cys[k] * uy[i,j]).^2 - 1.5*(ux[i,j].^2 + uy[i,j].^2))
end

@fastmath @inbounds function applyBGKCollision!(F::Array{Float32},Feq::Array{Float32})
    @. F -= (1.0/tau) .* (F .- Feq)
end

@fastmath @inbounds function calcU!(ux::Array{Float32},uy::Array{Float32},F::Array{Float32},rho::Array{Float32})
    @einsum ux[i,j] = F[i,j,k] * cxs[k] / rho[i,j]
    @einsum uy[i,j] = F[i,j,k] * cys[k] / rho[i,j]
end

function iterateLB(f::LatticeState, Nt::Int64)
    for it in 1:Nt
        calcU!(f.ux, f.uy, f.F, f.rho)
        calcFeq!(f.Feq, f.rho, f.ux,f.uy) #calculate Feq then apply the collision step
        applyBGKCollision!(f.F, f.Feq)
        applyDrift!(f.tmp,f.F) #Apply the particle drift
        sum!(f.rho, f.F) # Calculate fluid variables
    end
end

problem = LatticeState(Nx,Ny)
problem.rho += [exp(-sqrt((x-Nx/2)^2 + (y-Ny/2)^2)) for x in 1:Nx, y in 1:Ny]; #Initial condition. Modify density with a pulse in the middle

iterateLB(problem,Nt) #run simulation and save output

Plots.heatmap(problem.rho, c=:viridis, size=(650,640), aspect_ratio=:equal)#plot the final density variation
