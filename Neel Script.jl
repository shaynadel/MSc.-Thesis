using SparseArrays, LinearAlgebra, LaTeXStrings, Arpack, ProgressMeter, PyPlot, Colors, Statistics, DelimitedFiles
rcParams = PyPlot.matplotlib["rcParams"]
rcParams.update(Dict(
    "text.usetex" => true,
    "font.family" => "Helvetica"
))


function is_inside_hexagon(kx, ky)
    vertices = [
        (4*pi/3, 0),
        (4*pi/3 * cos(π/3), 4*pi/3 * sin(π/3)),
        (4*pi/3 * cos(2π/3), 4*pi/3 * sin(2π/3)),
        (4*pi/3 * cos(π), 4*pi/3 * sin(π)),
        (4*pi/3 * cos(4π/3), 4*pi/3 * sin(4π/3)),
        (4*pi/3 * cos(5π/3), 4*pi/3 * sin(5π/3))
    ]
    count = 0
    for i in 1:length(vertices)
        v1 = vertices[i]
        v2 = vertices[mod1(i + 1, length(vertices))]
        # Check if point (kx, ky) is inside using cross products
        if (v2[1] - v1[1]) * (ky - v1[2]) - (v2[2] - v1[2]) * (kx - v1[1]) >= 0
            count += 1
        end
    end
    return count == length(vertices)  # All edges have positive cross product
end


function NNBond(l1::Int64,l2::Int64)

    if l1 == 1 
        if l2 == 2 # 12
            return [1.0,0.0]
        elseif l2 == 3 # 13
            return [-0.5,sqrt(3)/2]
        elseif l2 == 4 # 14
            return [0.5,sqrt(3)/2]
        end
    elseif l1 == 2 
        if l2 == 3 # 23
            return [0.5,sqrt(3)/2]
        elseif l2 == 4 # 24
            return [-0.5,sqrt(3)/2]
        end
    elseif l1 == 3
        if l2 == 4 # 34
            return [1.0,0.0]
        end
    else
        assert(false,"l1=$l1,l2=$l2 values are not valid")
        return [0.0,0.0]
    end
end

function NNNBond(l1::Int64,l2::Int64)

    if l1 == 1 
        if l2 == 2 # 12
            return sqrt(3)*[1.0,0.0]
        elseif l2 == 3 # 13
            return sqrt(3)*[sqrt(3)/2,0.5]
        elseif l2 == 4 # 14
            return sqrt(3)*[sqrt(3)/2,-0.5]
        end
    elseif l1 == 2 
        if l2 == 3 # 23
            return sqrt(3)*[sqrt(3)/2,-0.5]
        elseif l2 == 4 # 24
            return sqrt(3)*[sqrt(3)/2,0.5]
        end
    elseif l1 == 3
        if l2 == 4 # 34
            return sqrt(3)*[1.0,0.0]
        end

    else
        assert(false,"l1=$l1,l2=$l2 values are not valid")
        return [0,0]
    end
end


function HighSymmetryPath(;N_points = 1000)::Vector{Vector{Float64}}
    """
        Scanns BZ through the  symmetry path.
        returns
    """
    # Define -symmetry points in k-space
    # Γ = [0.0, 0.0]
    # K = [4*pi/3, 4*pi/(3*sqrt(3))]
    # M = [4*pi/3, 0.0]
    Γ = [0.0, 0.0]
    K = [4*pi/3, 0.0]
    M = 2*pi/(sqrt(3))*[cos(pi/6),-sin(pi/6)]

    # Linear interpolation between -symmetry points
    Γ_K = [Γ .* (1 - t) + K .* t for t in range(0, stop=1, length=N_points)]
    K_M = [K .* (1 - t) + M .* t for t in range(0, stop=1, length=N_points)]
    M_Γ = [M .* (1 - t) + Γ .* t for t in range(0, stop=1, length=N_points)]

    # Concatenate all path segments
    BZ_path = vcat(K_M, M_Γ,Γ_K)
    return BZ_path
end

function Omega_k(J::Float64,k::Vector{Float64},l1::Int64,l2::Int64)
    
    chi_nn = NNBond(l1,l2)
    chi_nnn = NNNBond(l1,l2)

    omega = 2*sqrt(
        ( 1 + J )^2 
        - ( cos(k'*chi_nn) + J*cos(k'*chi_nnn) )^2
    )
    return omega
end

function GSEnergy_per_k(J::Float64,k::Vector{Float64})
    """
        Returns the ground state energy per k
    """
    E = 0.0
    for l1 in 1:3
        for l2 in l1+1:4

            chi_nn = NNBond(l1,l2)
            chi_nnn = NNNBond(l1,l2)
            E += (
                Omega_k(J,k,l1,l2)- 2*( 1 + J )
            )
        end
    end
    return E
end

function GSEnergy(J::Float64;N=100)
    E = 0.0
    kx_values = range(-3π/2,3π/2,N)
    ky_values = range(-3π/2,3π/2,N)
    count = 0
    for kx in kx_values
        for ky in ky_values
            if is_inside_hexagon(kx, ky)
                k_vec = [kx, ky]
                count += 1
                E += GSEnergy_per_k(J,k_vec)
            end
        end
    end  
    return E/count

end