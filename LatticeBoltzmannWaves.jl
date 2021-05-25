using Plots, Einsum

# Simulation parameters
const Nx, Ny          = 401, 401    # grid points in x and y
const rho0        = 1000    # average density
const Nt          = 600   # number of timesteps
const NL          = 9       # D2Q9 Lattice
const tau         = 0.6    # collision timescale
const omega       = 1/tau # appears in the collision equation, precomputing saves a division.

const cxs = [0, 0, 1, 1, 1, 0,-1,-1,-1]
const cys = [0, 1, 1, 0,-1,-1,-1, 0, 1]
const opp = [1,6,7,8,9,2,3,4,5] # bounce back array, opposite direction. Not used but will be when boundaries are added.
const weights = [4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36] # sums to 1

##build the struct that will hold the state of the LB system##
mutable struct LatticeState
    rho::Array{Float32, 2}    # macroscale density
    ux::Array{Float32, 2}    # macroscale velocity, x component
    uy::Array{Float32, 2}    # macroscale velocity, y component
    F::Array{Float32, 3}    # particle distribution function
    Feq::Array{Float32, 3}    # equillibrium term for collision
    tmp::Array{Float32, 2}   # place-holder array

    function LatticeState(Nx, Ny) #constructor
        self = new()
        self.F = fill(1.f0, (Nx,Ny,NL))
        self.Feq = similar(self.F)
        self.ux = fill(0.f0, (Nx,Ny))
        self.uy = similar(self.ux)
        self.rho = similar(self.ux)
        self.tmp = similar(self.ux)
        sum!(self.rho, self.F)
        set_initial_condition!(self)
        return self
    end
end

@fastmath @inbounds function apply_drift!(tmp::Array{<:Real},F::Array{<:Real})
    for i in 1:size(F)[3]
        A = @view F[:,:,i]
        circshift!(tmp,A, (cxs[i],0) )
        circshift!(A,tmp, (0,cys[i]) )
    end
end

@fastmath @inbounds function feq_point(rho::Real,ux::Real,uy::Real,cx::Int,cy::Int,weight::Real)
    cu = (cx * ux +  cy * uy)
    return rho*weight*(1. +  3. *cu + 4.5*cu*cu - 1.5*(ux*ux + uy*uy))
end

@fastmath @inbounds function calculate_feq!(Feq::Array{<:Real},rho::Array{<:Real},ux::Array{<:Real},uy::Array{<:Real})
    @einsum Feq[i,j,k] = feq_point(rho[i,j],ux[i,j],uy[i,j],cxs[k],cys[k],weights[k])
end

@fastmath @inbounds function apply_collision!(F::Array{<:Real},Feq::Array{<:Real})
    @. F -= omega .* (F .- Feq)
end

@fastmath @inbounds function calculate_u!(ux::Array{<:Real},uy::Array{<:Real},F::Array{<:Real},rho::Array{<:Real})
    @einsum ux[i,j] = F[i,j,k] * cxs[k] / rho[i,j]
    @einsum uy[i,j] = F[i,j,k] * cys[k] / rho[i,j]
end

@fastmath @inbounds function correct_F!(F::Array{<:Real},rho::Array{<:Real})
    @einsum F[i,j,k] = F[i,j,k] * rho0 / rho[i,j]
end

function iterate_lb(f::LatticeState, Nt::Int)
    for it in 1:Nt
        calculate_u!(f.ux, f.uy, f.F, f.rho) # Calculate macroscopic velocity
        calculate_feq!(f.Feq, f.rho, f.ux,f.uy) # Calculate Feq then apply the collision step
        apply_collision!(f.F, f.Feq)
        apply_drift!(f.tmp,f.F) # Apply the particle drift
        sum!(f.rho, f.F) # Calculate fluid density
    end
end

@fastmath @inbounds function set_initial_condition!(f::LatticeState)
    @. f.rho += [exp(-sqrt((x-Nx/2)^2 + (y-Ny/2)^2)) for x in 1:Nx, y in 1:Ny]; # Initial condition. Modify density with a pulse in the middle
    correct_F!(f.F,f.rho) #correct F to the new density
    sum!(f.rho,f.F)
    calculate_u!(f.ux,f.uy,f.F,f.rho)
end

function run()
    problem = LatticeState(Nx,Ny)
    iterate_lb(problem,Nt) # Run simulation
    Plots.heatmap(problem.rho, c=:redsblues, size=(650,640), clims=(rho0-.2, rho0+0.2), aspect_ratio=:equal) # Plot the final density variation
end

run()