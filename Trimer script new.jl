using SparseArrays, LinearAlgebra, LaTeXStrings, Arpack, ProgressMeter, PyPlot, Colors, Statistics, DelimitedFiles
rcParams = PyPlot.matplotlib["rcParams"]
rcParams.update(Dict(
    "text.usetex" => true,
    "font.family" => "Helvetica"
))


"""
    This script contains 5 parts:
    1. General functions.
    2. Building the operators in the ireeducible representation basis.

    5. Vacuum state properties: energy, fluctuations and magnetization.
"""


## 1. General functions
"""
    This part contains general functions that are used in the rest of the code. 
    Mainly concatenating integers, printing sparse matrices, checking if a point is inside the BZ, and the high symmetry path for the plots.
"""
function Realize(num::Complex{Float64},tol::Float64)
    """
        Within an error tolerance tol, returns a real number. If out of the tolerance, throws an error.
        Arguments:
            num (Complex{Float64})- complex number to be realized
            tol (Float64) - tolerance for imaginary part
        Returns:
            Realized number or error if imaginary part is larger than tol.
    """
    if abs(num.im) < tol
        return num.re
    else
        error("Imaginary value is larger than $tol")
    end
end

function print_nonzero_entries(matrix::SparseMatrixCSC)
    """
        Gives a nice way to view the non-empty enteries of a sparse matrix.
        Arguments:
            matrix (SparseMatrixCSC) - sparse matrix to be printed.
    """
    for col in 1:size(matrix, 2)
        for row_ptr in matrix.colptr[col]:(matrix.colptr[col + 1] - 1)
            row = matrix.rowval[row_ptr]
            val = matrix.nzval[row_ptr]
            println("[$row, $col] $val")
        end
    end
end

function ConcInts(i::Int64,j::Int64,k::Int64)
    """"
        Concatenates 3 single-digit integers to 3-digit integer.
        Arguments:
            i,j,k (Int64) - integers to be concatenated.
    """
    
    str1, str2, str3 = [string(i),string(j), string(k)]
    concatenated_str = str1 * str2 * str3
    return parse(Int, concatenated_str)
end

function threshold_sparse!(matrix::SparseMatrixCSC, threshold::Float64)
    """
        Removes elemens that are less in absolute value than threshold and leaves them empty.
        Arguments:
            matrix (SparseMatrixCSC) - sparse matrix to be thresholded.
            threshold (Float64) - threshold value.
        Returns:
            matrix (SparseMatrixCSC) - thresholded sparse matrix.
    """
    matrix.nzval .= ifelse.(abs.(matrix.nzval) .< threshold, 0.0, matrix.nzval)
    dropzeros!(matrix)  # Removes entries exactly equal to zero
    return matrix
end

function PrintWholeSparse(matrix::SparseMatrixCSC;digits=4)
    """
        Prints the whole sparse matrix in a nice format.
        Arguments:
            matrix (SparseMatrixCSC) - sparse matrix to be printed.
            {keyword} digits (Int64) = 4 - number of digits to round to.
    """
    x = round.(matrix,digits=digits)
    Base.print_matrix(stdout, x)
end

function is_inside_hexagon(kx, ky)
    """
        Checks if the point (kx, ky) is inside a hexagon with vertices at (4π/3, 0), (4π/3 * cos(π/3), 4π/3 * sin(π/3)), etc.
        Arguments:
            kx (Float64) - x-coordinate of the point.
            ky (Float64) - y-coordinate of the point.
        Returns:
            Boolean - true if the point is inside the hexagon, false otherwise.
    """
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

function HighSymmetryPath(;N_points = 1000)::Vector{Vector{Float64}}
    """
        Scanns BZ through the high symmetry path.
        Used to plot the excitations.
        Arguments:
            {keyword} N_points (Int64) = 1000 - number of points in the path.
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



## 2. Building the operators
"""
    This part contains the functions that build the operators in the irreducible representation basis.
    Starting from the definition in the product basis, then creating a change of basis matrix, and finally the spin operators dictionary
    in the irrep. basis.
"""

# Product basis operators
function CreateProdDict()
    """
        Creating dictionary mapping product basis states.
        Corresponding to the product state dictionary in the notes.
        Returns:
            Dict{Int64,Int64} - dictionary with keys as concatenated integers and values as indices.
            Example:
                dict[123] = 1
                dict[321] = 2
    """
    dict = Dict{Int64,Int64}()
    # Distinct cyclic
    dict[123], dict[312], dict[231]= [1,2,3]

    # Distinct odff
    dict[213], dict[132], dict[321]= [4,5,6]
    
    # Unique
    dict[111], dict[222], dict[333] = [7,8,9]
    
    # Single+Pair
    dict[211], dict[121], dict[112] = [10,11,12]
    dict[311], dict[131], dict[113] = [13,14,15]
    dict[122], dict[212], dict[221] = [16,17,18]
    dict[133], dict[313], dict[331] = [19,20,21]
    dict[322], dict[232], dict[223] = [22,23,24]
    dict[233], dict[323], dict[332] = [25,26,27]

    # With flavor 4
    dict[423], dict[432], dict[342], dict[243], dict[234], dict[324] = [28,29,30,31,32,33]
    dict[431], dict[413], dict[143], dict[341], dict[314], dict[134] = [34,35,36,37,38,39]
    dict[412], dict[421], dict[241], dict[142], dict[124], dict[214] = [40,41,42,43,44,45]
    dict[444] = 46
    dict[144],dict[414],dict[441] = [47,48,49]
    dict[244],dict[424],dict[442] = [50,51,52]
    dict[344],dict[434],dict[443] = [53,54,55]
    dict[114],dict[141],dict[411] = [56,57,58]
    dict[224],dict[242],dict[422] = [59,60,61]
    dict[334],dict[343],dict[433] = [62,63,64]
    return dict
end


function CreateTMatrix(l::Int64,mu::Int64,nu::Int64, dict::Dict{Int64,Int64})
    """
        Creates T^{mu nu}_l matrix in the product state basis.
        Arguments:
            l (Int64) - spatial index of the trimer. takes values 1,2,3
            mu (Int64) - flavor index. takes values 1,2,3,4
            nu (Int64) - flavor index. takes values 1,2,3,4
            dict (Dict{Int64,Int64}) - dictionary from CreateProdDict().
        Returns:
            SparseMatrixCSC{Float64, Int64} - sparse matrix in product state basis.
                Containes 1 in enteries allowed by the spin operator.
            
    """
    S = spzeros(64,64)
    for a in 1:4
        for b in 1:4
            # State in product basis
            
            if l==1 # Rearrange w.r.t l
                int1 = ConcInts(mu,a,b)
                int2 = ConcInts(nu,a,b)
            elseif l==2
                int1 = ConcInts(a,mu,b)
                int2 = ConcInts(a,nu,b)
            elseif l==3
                int1 = ConcInts(a,b,mu)
                int2 = ConcInts(a,b,nu)
            else
                error("Invalid value for l. l must be between 1 and 3")
            end

            # Mapped value
            try 
                i = dict[int1]
                j = dict[int2]
                S[i,j] = 1           
            catch
                continue
            end
        end
    end
    return S 
end

# Representation theory operators 
function ChangeOfBasisMatrix()
    """
        Generates the Clebsh-Gordan matrix mapping the inital basis to the irrep. basis.
        Anomalous terms appear only on the 16-27 and 28-45 subspace. 
        The auxiliary subspace 46-64 is unchanged is turns to be irrelevant in the calculations.
        Returns:
            U (Matrix{Float64}) - matrix of size 64x64.
    """
    U = zeros(64,64)

    v1 = SparseVector(64, [1,2,3,4,5,6], 1/sqrt(6)*[1,1,1,-1,-1,-1])
    v2 = SparseVector(64, [1,2,3,4,5,6], 1/sqrt(6)*[1,1,1,1,1,1])
    v3 = SparseVector(64, [7], [1])
    v4 = SparseVector(64, [8], [1])
    v5 = SparseVector(64, [9], [1])
    v6 = SparseVector(64, [10,11,12], 1/sqrt(3)*[1,1,1])
    v7 = SparseVector(64, [13,14,15], 1/sqrt(3)*[1,1,1])
    v8 = SparseVector(64, [16,17,18], 1/sqrt(3)*[1,1,1])
    v9 = SparseVector(64, [19,20,21], 1/sqrt(3)*[1,1,1])
    v10 = SparseVector(64, [22,23,24], 1/sqrt(3)*[1,1,1])
    v11 = SparseVector(64, [25,26,27], 1/sqrt(3)*[1,1,1])

    v12 = SparseVector(64, [1,2], 1/sqrt(2)*[-1,1])
    v13 = SparseVector(64, [1,2,3], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v14 = SparseVector(64, [4,5], 1/sqrt(2)*[-1,1])
    v15 = SparseVector(64, [4,5,6], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])

    # 16-27 subspace 
    # Mixed rep.
    v16 = SparseVector(64, [10,11], 1/sqrt(2)*[-1,1])
    v17 = SparseVector(64, [10,11,12], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v18 = SparseVector(64, [13,14], 1/sqrt(2)*[-1,1])
    v19 = SparseVector(64, [13,14,15], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v20 = SparseVector(64, [16,17], 1/sqrt(2)*[-1,1])
    v21 = SparseVector(64, [16,17,18], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v22 = SparseVector(64, [19,20], 1/sqrt(2)*[-1,1])
    v23 = SparseVector(64, [19,20,21], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v24 = SparseVector(64, [22,23], 1/sqrt(2)*[-1,1])
    v25 = SparseVector(64, [22,23,24], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    v26 = SparseVector(64, [25,26], 1/sqrt(2)*[-1,1])
    v27 = SparseVector(64, [25,26,27], [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])

    # 28-45 subspace
    # Antisymmetric rep.
    u28 = SparseVector(64, [28,30,32,29,31,33], 1/(sqrt(6))*[1,1,1,-1,-1,-1]) # 1bar
    u29 = SparseVector(64, [34,36,38,35,37,39], 1/(sqrt(6))*[1,1,1,-1,-1,-1]) # 2bar
    u30 = SparseVector(64, [40,42,44,41,43,45], 1/(sqrt(6))*[1,1,1,-1,-1,-1]) # 3bar
    # Symmetric rep.
    u31 = SparseVector(64, [28,30,32,29,31,33], 1/(sqrt(6))*[1,1,1,1,1,1]) # 1bar sym
    u32 = SparseVector(64, [34,36,38,35,37,39], 1/(sqrt(6))*[1,1,1,1,1,1]) # 2bar sym
    u33 = SparseVector(64, [40,42,44,41,43,45], 1/(sqrt(6))*[1,1,1,1,1,1]) # 3bar sym
    # Mixed rep.
    u34 = SparseVector(64, [1,3].+27, 1/sqrt(2)*[-1,1])
    u35 = SparseVector(64, [2,4].+27, 1/sqrt(2)*[-1,1])
    u36 = SparseVector(64, [2,4,6].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    u37 = SparseVector(64, [1,3,5].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    u38 = SparseVector(64, [7,9].+27, 1/sqrt(2)*[-1,1])
    u39 = SparseVector(64, [8,10].+27, 1/sqrt(2)*[-1,1])
    u40 = SparseVector(64, [7,9,11].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    u41 = SparseVector(64, [8,10,12].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    u42 = SparseVector(64, [13,15].+27, 1/sqrt(2)*[-1,1])
    u43 = SparseVector(64, [14,16].+27, 1/sqrt(2)*[-1,1])
    u44 = SparseVector(64, [13,15,17].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])
    u45 = SparseVector(64, [14,16,18].+27, [-1/sqrt(6),-1/sqrt(6),sqrt(2/3)])

    # Auxiliary subapce 
    vectors = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,
                v21,v22,v23,v24,v25,v26,v27,
                u28,u29,u30,u31,u32,u33,
                u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45]  # List of sparse vectors

    for (i, v) in enumerate(vectors)
        U[i, :] = Vector(v)
    end
    
    # Rest of the space is not transformed.
    for i in 46:64
        U[i,i] =1.0
    end

    return U
end

function A_l_mu_nu(l::Int64,mu::Int64,nu::Int64,dict::Dict{Int64,Int64},U::Matrix{Float64})
    """
        Creates the trimer spin operator matrix elements in the irreducible representation basis.
        Arguments:
            l (Int64) - spatial index of the trimer. takes values 1,2,3
            mu (Int64) - flavor index. takes values 1,2,3,4
            nu (Int64) - flavor index. takes values 1,2,3,4
            dict (Dict{Int64,Int64}) - dictionary from CreateProdDict().
            U (Matrix{Float64}) - Clebsh-Gordan matrix from ChangeOfBasisMatrix()
        Returns:
            SparseMatrixCSC{Float64, Int64} - 64x64 sparse matrix.
    """
    # Create S in product basis
    S = CreateTMatrix(l,mu,nu,dict)
    A_l_mu_nu = U*S*transpose(U)
    return A_l_mu_nu
end

function CreateA_dict()
    """
        Creates a dictionary of the trimer spin operator matrix elements in the irreducible representation basis.
        Dictionary is indexed by concatenated integers - l mu nu.
        l = 1,2,3 - spatial index of the trimer.
        mu = 1,2,3,4 - flavor index.
        nu = 1,2,3,4 - flavor index.
        Returns:
            Dict{Int64,SparseMatrixCSC{Float64, Int64}}
            keys:
                A_dict[l mu nu] => A_l^{mu,nu} 
            
            Example:
                A_dict[123] => A_1^{23}
    """
    
    A_dict = Dict{Int64,SparseMatrixCSC{Float64, Int64}}()
    dict = CreateProdDict()
    U = ChangeOfBasisMatrix()
    for l in 1:3
        for mu in 1:4
            for nu in 1:4
                ind = ConcInts(l,mu,nu)
                A_dict[ind] = A_l_mu_nu(l,mu,nu,dict,U)
                threshold_sparse!(A_dict[ind],1e-16)
            end
        end
    end

    return A_dict
end

# Involves only BB bonds
function BB_bonds(l1::Int64,l2::Int64)
    """
        Creates the real space chi vector describing the bond. No intra-trimer bonds (chi=0)
        Arguments:
            l1 (Int64) - spatial index of the first trimer site. Takes values 1:3
            l2 (Int64) - spatial index of the second trimer site. Takes values 1:3
        Returns:
            nn (Vector{Vector{Float64}}) - nearest neighbor bonds. 1 element.
            nnn (Vector{Vector{Float64}}) - next nearest neighbor bonds. 2 elements.
    """
    # NN bonds
    if l1 == 1
        if l2 == 2
            # 12
            nn = [
                [-1.0,0.0]
            ]
            nnn = [
                [-0.5,-sqrt(3)/2],
                [-0.5, sqrt(3)/2]
            ]
        elseif l2 == 3
            # 13
            nn = [
                [-0.5, - sqrt(3)/2]
            ]
            nnn = [
                [0.5, - sqrt(3)/2],
                [-1.0,0]
            ]
        end
    elseif l1 ==2
        # 23
        if l2 == 3
            nn = [
                [0.5, - sqrt(3)/2]
            ]
            nnn = [
                [-0.5, - sqrt(3)/2],
                [1.0,0.0]
            ]
        end
    end
    return nn, nnn
end

function A_munu_A_numu(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}}, l1::Int64, l2::Int64, mu::Int64,nu::Int64)
    """
        Creates an assisting calculation to the interaction matrix of the trim-trimer interaction.
        Arguments:
            A_dict - A_l ^{mu nu} matrix. Created from CreateA_dict(). 
            l1 (Int64) - spatial index of the first A matrix. Takes values 1:3
            l2 (Int64) - spatial index of the second A matrix. Takes values 1:3
            mu (Int64) - flavor index. Takes values 1:4
            nu (Int64) - flavor index. Takes values 1:4
        Returns:
            normal1 (SparseMatrixCSC{Float64, Int64}) - b^{dagger} b matrix. -> A_A[1]
            normal2 (SparseMatrixCSC{Float64, Int64}) - b b^{dagger} matrix. -> A_A[2]
            anomalous1 (SparseMatrixCSC{Float64, Int64}) - b b matrix. -> A_A[3]
            anomalous2 (SparseMatrixCSC{Float64, Int64}) - b^{dagger} b^{dagger} matrix. -> A_A[4]
            A_loc (SparseMatrixCSC{Float64, Int64}) - local interaction matrix. -> A_A[5]
    """
    ind1 = ConcInts(l1,mu,nu)
    ind2 = ConcInts(l2,nu,mu)

    v1 = A_dict[ind1][1,:] # Row
    v2 = A_dict[ind1][:,1] # Column

    u1 = A_dict[ind2][1,:] # Row
    u2 = A_dict[ind2][:,1] # Column
    # Normal terms
    normal1 = v2 * u1'
    normal2 = v1 * u2'

    # Anomalous terms
    anomalous1 = v1 * u1'
    anomalous2 = v2 * u2'

    A_mat1 = A_dict[ind1]
    A_mat2 = A_dict[ind2]
    #TODO: Check if twice the negative term or not
    A_loc = A_mat1[1,1] * A_mat2 + A_mat1 * A_mat2[1,1] - 2*A_mat1[1,1]*A_mat2[1,1]*Diagonal(ones(64))

    return normal1, normal2, anomalous1, anomalous2, A_loc
end

function Al1_Al2(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}}, l1::Int64, l2::Int64;tol=1e-15)
    """
        Creates the A_A[s] matrices (from the notes). Uses A_munu_A_numu when contracting the flavor indeces
            sum_(mu nu).
        Arguments:
            A_dict - A_l ^{mu nu} matrix. Created from CreateA_dict().
            l1 - spatial index of first A matrix. Takes values 1:3
            l2 - spatial index of first A matrix. Takes values 1:3
        Returns:
            Dict{Int64, SparseMatrixCSC{Float64, Int64}}
            keys:
                1 => normal1 -> b^{dagger} b
                2 => normal2 -> b b^{dagger}
                3 => anomalous1 -> b b
                4 => anomalous2 -> b^{dagger} b^{dagger}
                5 => local -> b^{dagger} b
    """

    A_A = Dict{Int64, SparseMatrixCSC{Float64, Int64}}(
        1 => spzeros(64,64),
        2 => spzeros(64,64),
        3 => spzeros(64,64),
        4 => spzeros(64,64),
        5 => spzeros(64,64)
    )

    for mu in 1:4
        for nu in 1:4
            #TODO: Check if with the if condition or without.
            # if mu != nu 
                normal1, normal2, anomalous1, anomalous2, local1 = A_munu_A_numu(A_dict,l1,l2,mu,nu)
                A_A[1] += normal1
                A_A[2] += normal2
                A_A[3] += anomalous1
                A_A[4] += anomalous2
                A_A[5] += local1
            # end
        end
    end
    
    for i in 1:5
        A_A[i] = threshold_sparse!((A_A[i]), tol) # Throwing elements close to zero up to tol difference.
    end
    return A_A
end



## 4. 28-45 subspace
"""
    This part builds the 28-45 subspace matrix, then diagonalizes it using a Bogolioubov transformation.
"""
function AB_chi_bonds()
    """
       Creates dictionaries of the real space vectors of the AB bonds.
       Within the NN and NNN dictionaries, the keys are integers 1,2,3 representing the trimer site,
       Returns:
            nn_bonds (Dict{Int64,Vector{Vector{Float64}}}) - nearest neighbor bonds. 
            nnn_bonds (Dict{Int64,Vector{Vector{Float64}}}) - next nearest neighbor bonds.
    """
    nn_bonds = Dict{Int64,Vector{Vector{Float64}}}()
    nnn_bonds = Dict{Int64,Vector{Vector{Float64}}}()

    # 1->4 bonds
    nn_bonds[1] = [
       1/(sqrt(3))*[-sqrt(3)/2, 1/2],
       1/(sqrt(3))*[0.0,-1.0]
    ]
    nnn_bonds[1] = [
       1/(sqrt(3))*[sqrt(3)/2, 1/2],
       2/(sqrt(3))*[-sqrt(3)/2,-1/2]
    ]
    
    # 2->4 bonds
    nn_bonds[2] = [
       1/(sqrt(3))*[sqrt(3)/2, 1/2],
       1/(sqrt(3))*[0.0,-1.0]
    ]
    nnn_bonds[2] = [
       1/(sqrt(3))*[-sqrt(3)/2, 1/2],
       2/(sqrt(3))*[sqrt(3)/2,-1/2]
    ]
    
    # 3->4 bonds
    nn_bonds[3] = [
       1/(sqrt(3))*[sqrt(3)/2, 1/2],
       1/(sqrt(3))*[-sqrt(3)/2,1/2]
    ]
    nnn_bonds[3] = [
       2/(sqrt(3))*[0.0, 1.0],
       1/(sqrt(3))*[0.0,-1.0]
    ]


    return nn_bonds , nnn_bonds
end



## 5. Vacuum state properties: energy, fluctuations and magnetization.
"""
    Here I seperate the contributions to the vacuum state energy in the following way:
        1. Normal ordering: Constant terms on the way to get to the canonical form.
            a) BB: from the normal ordering of the BB interactions.
            b) AB: from the normal ordering of the AB interactions.
            c) Quadratic: from the quadratic terms arising from the BB interactions (S^2).

            The full normal ordering is the sum of the three contributions.

        2. Bogolioubov: Normal ordering after the dispersion is achieved.
        3. Magnetization and fulctuation functions.
"""


""" 
    New functions.
    Using the old functions:
    - CreateA_dict()
    - BB_bonds() - Gets the chi vectors for the BB bonds.
    - AB_chi_bonds() - Gets the chi vectors for the AB bonds.
    - Al1_Al2() - Creates the A_l^{mu nu}A_L^{nu mu} contraction matrices.


"""

function find_subspace_D(A::SparseMatrixCSC, N::SparseMatrixCSC)
    """
        Finds the subspace of nonzero indices in matrix A, plus ones that are connected via matrix N.
        Test:
            sparse1 = spzeros(4, 4)
            sparse1[2,3] = 1.0 
            sparse2 = spzeros(4, 4)
            sparse2[1,3] = 1.0 
            find_subspace_D(sparse1,sparse2) # = 1,2,3
    """
    # Check that matrices have the same size
    if size(A) != size(N)
        error("Matrices A and N must have the same size")
    end
    
    # Step 1: Find all indices where A has non-zero elements
    rows_A, cols_A = findnz(A)[1:2]  # Get row and column indices of non-zero elements
    initial_indices = Set(union(rows_A, cols_A))  # Combine row and column indices
    
    # Step 2: Iteratively expand the subspace using connections from N
    current_indices = copy(initial_indices)
    expanded = true
    
    while expanded
        expanded = false
        new_indices = Set{Int}()
        
        # For each index currently in our subspace
        for idx in current_indices
            # Check row idx of matrix N for connections to indices outside current subspace
            for col_idx in 1:size(N, 2)
                if N[idx, col_idx] != 0 && !(col_idx in current_indices)
                    push!(new_indices, col_idx)
                    expanded = true
                end
            end
            
            # Check column idx of matrix N for connections to indices outside current subspace
            for row_idx in 1:size(N, 1)
                if N[row_idx, idx] != 0 && !(row_idx in current_indices)
                    push!(new_indices, row_idx)
                    expanded = true
                end
            end
        end
        
        # Add newly found indices to our current set
        union!(current_indices, new_indices)
    end
    
    # Convert to sorted vector and return
    return sort(collect(current_indices))
end

function SU4Irrep(m::Int64)
    """
        Returns the SU(4) irrep of state m according to the dictionary (after erasing the 4 bar state in index 1).
        States between 45 and 64 are for now in an auxiliary space, thus doesn't have a defined irrep.
        States between 64 and 66 are in the fundamental representation.
    """
    if 1 <= m <= 10 || 30 <= m <= 32
        return "S"
    elseif 11 <= m <= 26 || 33 <= m <= 44
        return "M"
    elseif 27 <= m <= 29
        return "AS"
    elseif 45 <= m <= 66
        return "Undefined"
    else
        error("m=$m is out of range.")
    end
end

function OnTrimerInteraction()
    """
        Creates the on-trimer interaction matrix.
        The matrix is a diagonal matrix with specific values based on the SU(4) irrep. the state is in.
        The diagonal elements are:
            - 6 for S states
            - 3 for M states
            - 0 for AS and Undefined states
        Returns:
            matrix::Matrix{ComplexF64}: Diagonal matrix.
    """
    matrix = spzeros(ComplexF64, 66, 66)
    
    # Fill diagonal elements based on categorize_m function
    for m in 1:66
        category = SU4Irrep(m)
        
        if category == "S"
            matrix[m, m] = 6.0 + 0.0im
        elseif category == "M"
            matrix[m, m] = 3.0 + 0.0im
        elseif category == "AS" || category == "Undefined"
            matrix[m, m] = 0.0 + 0.0im
        end
    end
    
    return matrix
end

function CreateCanonicalForm(N::Matrix{ComplexF64}, A::Matrix{ComplexF64})
    """
        Creates the canonical form from the normal matrix N and anomalous matrix A.
        The canonical form is a block matrix of the form:
            [ N   A ]
            [ A^dagger  N^dagger ] # (N^dagger = N)
        Arguments:
            N::Matrix{ComplexF64}: Normal matrix
            A::Matrix{ComplexF64}: Anomalous matrix
        Returns:
            block_matrix::Matrix{ComplexF64}: Block matrix in canonical form
        
    """
    # Check if N and A are of the same size
    if size(N) != size(A)
        error("Matrices N and A must have the same dimensions")
    end
    
    block_matrix = [
        N              A;
        conj(transpose(A))     N
    ]
    
    return block_matrix
end

function BogoliubovDiagonalization(M:: Matrix{ComplexF64})
    """
        Takes a 2N x 2N matrix and diagonalizes it using a Bogolioubov transformation.
        Considers the matrix to be in the canonical form:
            [ N   A ]
            [ A^dagger  N^dagger ] # (N^dagger = N)
        where N is the normal matrix and A is the anomalous matrix.
        Arguments:
            M (Matrix{ComplexF64}) - 2N x 2N matrix to be diagonalized.
        Returns:
            sorted_eigenvalues (Vector{Float64}) - eigenvalues of the matrix.
            sorted_eigenvecs (Matrix{ComplexF64}) - eigenvectors of the matrix.
    """
    # Get size and check if it's 2N x 2N
    total_size = size(M, 1)
    if size(M, 1) != size(M, 2) || total_size % 2 != 0
        error("Matrix must be 2Nx2N")
    end
    
    N = div(total_size, 2)
    
    # Create the Gamma matrix (diagonal with 1s and -1s)
    negative_vector = [ones(Int, N); -ones(Int, N)]
    Gamma = Diagonal(negative_vector)
    
    # Multiply the matrix from the left with Gamma
    to_diagonalize = Gamma * M
    
    # Get eigenvalues and eigenvectors
    vals, vecs = eigen(to_diagonalize)
    
    # Make sure eigenvalues are real up to tolerance. Realize(..) raises an error if not.
    tol = 1e-5
    eigenvals = [Realize(ComplexF64(val), tol) for val in vals]
    
    # Find the N negative eigenvalues and multiply them by -1
    modified_eigenvals = copy(eigenvals)
    
    # Since we expect half positive and half negative eigenvalues,
    # we need to identify the negative ones
    for i in 1:length(eigenvals)
        if eigenvals[i] < 0
            modified_eigenvals[i] *= -1
        end
    end
    
    # Sort in ascending order
    sorted_indices = sortperm(modified_eigenvals)
    
    # Reorder eigenvalues and eigenvectors
    sorted_eigenvalues = modified_eigenvals[sorted_indices]
    sorted_eigenvecs = vecs[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvecs
end

# Creating the interaction matrices.

function CreateNormalBB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64,
     k::Vector{Float64};
     alpha = 1.0
    )
    """
        Creates the full BB interactions normal matrix.
        Adds explicitly the on-trimer interaction terms via the function OnTrimerInteraction() with the specific
        classification as in SU4Irrep(..).
    """

    inter_trimer = spzeros(ComplexF64,66,66)
    for l1 in 1:3
        for l2 in l1+1:3
            nn_chi, nnn_chi = BB_bonds(l1,l2)
            A_A = Al1_Al2(A_dict,l1,l2)

            for chi in nn_chi
                inter_trimer[1:63,1:63] += (
                    A_A[1]* exp(-im* (k'*chi)) 
                    +
                    transpose(A_A[1])* exp(im* (k'*chi))
                    +
                    A_A[5]
                )[2:64,2:64]
            end

            for chi in nnn_chi
                inter_trimer[1:63,1:63] += J*(
                    A_A[1]* exp(-im* (k'*chi)) 
                    +
                    transpose(A_A[1])* exp(im* (k'*chi))
                    +
                    A_A[5]
                )[2:64,2:64]
            end
        end
    end

    on_trimer = OnTrimerInteraction() # Adds the on trimer interaction terms
    N = alpha * inter_trimer + on_trimer
    return threshold_sparse!(N,1e-10)
end

function CreateAnomalousBB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64,
     k::Vector{Float64};
     alpha = 1.0
    )
    """
        Creates the full BB interactions anomalous matrix.
    """

    A = spzeros(ComplexF64,66,66)
    for l1 in 1:3
        for l2 in l1+1:3
            nn_chi, nnn_chi = BB_bonds(l1,l2)
            A_A = Al1_Al2(A_dict,l1,l2)

            for chi in nn_chi
                A[1:63,1:63] += (
                    A_A[3]* exp(-im* (k'*chi)) 
                    +
                    transpose(A_A[3])* exp(im* (k'*chi))
                )[2:64,2:64]
            end

            for chi in nnn_chi
                A[1:63,1:63] += J*(
                    A_A[3]* exp(-im* (k'*chi)) 
                    +
                    transpose(A_A[3])* exp(im* (k'*chi))
                )[2:64,2:64]
            end
        end
    end
    

    return threshold_sparse!((alpha*A),1e-10)
end

function CreateNormalAB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64,
     k::Vector{Float64};
     beta = 1.0
    )
    """
        Creates the full AB interactions normal matrix.
    """
    N = spzeros(ComplexF64,66,66)
    nn_chi_dict, nnn_chi_dict = AB_chi_bonds()
    for l in 1:3

        # bb
        N[1:63,1:63] += (
            2*(1+J)* # Number of bonds
            A_dict[ConcInts(l,4,4)][2:64,2:64]
        )

        
        for mu in 1:3 
            
        # aa
            for nu in 1:3

                element = (
                    2*(1+J)* # Number of bonds
                    A_dict[ConcInts(l,mu,nu)][1,1]
                )
                N[63+mu,63+nu] += element #ab     
            end

        # ab
            normal_vec = spzeros(ComplexF64, 63, 1)
            for chi in nn_chi_dict[l]
                normal_vec += (
                    A_dict[ConcInts(l,mu,4)][2:64,1] * exp(-im* (k'*chi))
                )
            end

            for chi in nnn_chi_dict[l]
                normal_vec += (
                    J* A_dict[ConcInts(l,mu,4)][2:64,1] * exp(-im* (k'*chi))
                )
            end
            N[1:63,63+mu] += normal_vec

            N[63+mu,1:63] += conj(normal_vec)
            
        end


    end
    return threshold_sparse!((beta*N),1e-10)
end

function CreateAnomalousAB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64,
     k::Vector{Float64};
     beta = 1.0
    )
    """
        Creates the full AB interactions anomalous matrix.
    """
    A = spzeros(ComplexF64,66,66)
    nn_chi_dict, nnn_chi_dict = AB_chi_bonds()
    for l in 1:3
        for mu in 1:3

            anomalous_vec = spzeros(ComplexF64, 63, 1)
            for chi in nn_chi_dict[l]
                anomalous_vec += (
                    A_dict[ConcInts(l,4,mu)][2:64,1] * exp(-im* (k'*chi))
                )
            end

            for chi in nnn_chi_dict[l]
               anomalous_vec += J*(
                    A_dict[ConcInts(l,4,mu)][2:64,1] * exp(-im* (k'*chi))
                )
            end
            A[1:63,63+mu] += anomalous_vec

            A[63+mu,1:63] += conj(anomalous_vec)
        end
    end

    return threshold_sparse!((beta*A),1e-10)
end

function CreateFullMatrix(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64,
     k::Vector{Float64};
     ind_vec = collect(1:66),
     alpha = 1.0,
     beta = 1.0
    )
    """
        Creates the full matrix for the Bogoliubov diagonalization.
        ind_vec is a set of indeces truncating the Hamiltonian if one wants only a specific subspace
    """
    N = Matrix(
        CreateNormalBB(A_dict, J, k, alpha=alpha)[ind_vec,ind_vec] + 
        CreateNormalAB(A_dict, J, k, beta=beta)[ind_vec,ind_vec]
    )
    A = Matrix(
        CreateAnomalousBB(A_dict, J, k, alpha=alpha)[ind_vec,ind_vec] +
        CreateAnomalousAB(A_dict, J, k,beta=beta)[ind_vec,ind_vec]
    )
    mat = CreateCanonicalForm(N, A)
    return mat
end


function NormalOrderingBB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64;
     ind_vec = collect(2:64),
     alpha = 1.0
    )
    """
        Calculates the vacuum state energy's contribution from the BB interactions.
        The expressions follows from the procedure of getting the Hamiltonian to the canonical form.
        ind_vec is a set of indices creating a specific subspace.
        Quadratic contribution comes from terms with S^2.
        Linear contribution comes from both inter-trimer interactions, and from the constants of the on-trimer interaction. 
        Arguments:
            A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}}: Dictionary of sparse matrices.
            J::Float64: Interaction strength.
            ind_vec::Vector{Int64}: Vector of indices after truncation.
        Returns:
            E_BB::Float64: The BB vacuum energy.
    """

    # Quadratic
    E_BB = alpha*( 1+ 2*J) -3
    
    dict_ind_vec = [idx+1 for idx in ind_vec if 1 <= idx <= 63] # match the indices to the original dictionary indices for A_A terms.
    # Inter trimer linear
    for l1 in 1:3
        for l2 in l1+1:3 
            A_A = Al1_Al2(A_dict,l1,l2)[5][dict_ind_vec,dict_ind_vec]
            E_BB += -alpha* 0.5 * ( 1 + 2*J ) * tr(A_A)
        end
    end
    # On trimer linear 
    # Counts how many states are at S and M rep. from the on-trimer constants.
    count_S = count(idx -> SU4Irrep(idx) == "S", ind_vec)
    count_M = count(idx -> SU4Irrep(idx) == "M", ind_vec)
    
    E_BB += (-6 * count_S -3 * count_M)/2
    

    return E_BB
end

function NormalOrderingAB(A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
     J::Float64;
     ind_vec = collect(1:66),
     beta = 1.0
    )
    """
        Calculates the vacuum state energy's contribution from the AB interactions.
        The expressions follows from the procedure of getting the Hamiltonian to the canonical form.
        ind_vec is a set of indices creating a specific subspace.
        Arguments:
            A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}}: Dictionary of sparse matrices.
            J::Float64: Interaction strength.
            ind_vec::Vector{Int64}: Vector of indices after truncation.
        Returns:
            E_AB::Float64: The AB vacuum energy.
        
    """
    E_AB = 0.0
    dict_ind_vec = [idx+1 for idx in ind_vec if 1 <= idx <= 63] # match the indices to the original dictionary indices for A terms.
    for l in 1:3
        ind = ConcInts(l,4,4)
        E_AB += -(1+J) * tr(A_dict[ind][dict_ind_vec,dict_ind_vec])
        
        # for mu in 1:3
        #     ind2 = ConcInts(l,mu,mu)
        #     E_AB += -(1+J) * A_dict[ind2][1,1]
        # end
        E_AB += -(1+J)
    end

    return (beta*E_AB)
end

function BogoliubovVacuumEnergy(
        A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
        J:: Float64;
        ind_vec = collect(1:66),
        N = 100,
        alpha = 1.0,
        beta = 1.0
    )
    """
        Calculates the contribution of the normal ordering of the dispersion.
        ind_vec is a set of indeces creating a specific subspace.
        N is the number of points in the BZ we average over.
        Arguments:
            A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}}: Dictionary of sparse matrices.
            J::Float64: Interaction strength.
            ind_vec::Vector{Int64}: Vector of indices after truncation.
            N::Int: Number of points in the BZ to average over.
        Returns:
            E_bog::Float64: The Bogoliubov vacuum energy.
    """

    kx_values = range(-3π/2, 3π/2, N)
    ky_values = range(-3π/2, 3π/2, N)
    
    Q_x = repeat(kx_values, 1, length(ky_values))
    Q_y = permutedims(repeat(ky_values, 1, length(kx_values)), (2, 1))

    # Create mask for points inside the hexagon
    mask = is_inside_hexagon.(Q_x, Q_y)

    # Get valid k points
    valid_kx = Q_x[mask]
    valid_ky = Q_y[mask]

    eigen_sum_vec = zeros(Float64, length(valid_kx))
    for m in eachindex(valid_kx)
        k_vec = [valid_kx[m], valid_ky[m]]
        M = CreateFullMatrix(A_dict, J, k_vec, ind_vec = ind_vec, alpha=alpha, beta=beta)
        eigenvals, _ = BogoliubovDiagonalization(M)
        eigen_sum_vec[m] = sum(eigenvals)/4 # Devided by half for twice the eigenvalues, and by another half from normal ordering
    end
    E_bog = mean(eigen_sum_vec)  

    return E_bog
end

"""
    Fluctuations and magnetization functions.
"""
function BogoliubovDiagonalizationForExp(M:: Matrix{ComplexF64})
    """
        Same as BogoliubovDiagonalization just does not re-aply Gamma matrix, by so we separate between alpha and alpha^dagger in this order.
        Used for the fluctuation calculation.
    """
    # Get size and check if it's 2N x 2N
    total_size = size(M, 1)
    if size(M, 1) != size(M, 2) || total_size % 2 != 0
        error("Matrix must be 2Nx2N")
    end
    
    N = div(total_size, 2)
    
    # Create the Gamma matrix (diagonal with 1s and -1s)
    negative_vector = [ones(Int, N); -ones(Int, N)]
    Gamma = Diagonal(negative_vector)
    
    # Multiply the matrix from the left with Gamma
    to_diagonalize = Gamma * M
    
    # Get eigenvalues and eigenvectors
    vals, vecs = eigen(to_diagonalize)
    
    # Make sure eigenvalues are real up to tolerance. Realize(..) raises an error if not.
    tol = 1e-5
    eigenvals = [Realize(ComplexF64(val), tol) for val in vals]
        
    # Sort in highest first -> corresponding to alpha -> V is in the correct place.
    sorted_indices = sortperm(eigenvals, rev=true) # 
    
    # Reorder eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvals[sorted_indices]
    sorted_eigenvecs = vecs[:, sorted_indices]
    
    return sorted_eigenvalues, sorted_eigenvecs
end


function ExpNum_per_k(
        A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
        J::Float64,
        k::Vector{Float64};
        ind_vec = collect(1:66),
        alpha = 1.0,
        beta = 1.0
    )
    """ 
        Calculates <a^dagger a> and <b^dagger b> per k.
        Transformation matrix is of the form:
            W = [ U  V ]
                [ V' U' ]
        connecting (b a b^dag a^ag) to ( alpha alpha^dag).
        Therefore, the fluctuations comes from norm(V)^2. To match to the two different sublattices, we track different indices.
        Last 3 indices of ind_vec needs to be 64,65,66 that corresponds to a operators.
        Arguments:
            A_dict (Dict{Int64, SparseMatrixCSC{Float64, Int64}}) - A_l ^{mu nu} matrix. Created from CreateA_dict().
            J (Float64) - J2/J1 coupling parameter.
            k (Vector{Float64}) - k vector.
            {keyword} ind_vec (Vector{Int64})= collect(1:66) - Vector of indices after truncation. Needs to have 64,65,66 at the end.
            {keyword} alpha (Float64) = 1.0 - Inter-trimer over on-trimer interaction ratio (unphysical parameter).
            {keyword} beta (Float64) = 1.0 - AB interaction over on-trimer interaction ratio (unphysical parameter).
        Returns:
            n_A (Float64) - <a^dagger a> expectation value on sublattice A per k.
            n_B (Float64) - <b^dagger b> expectation value on sublattice B per k.
    """
    N = length(ind_vec)
    BB_ind = 1:(N -3)
    AA_ind = (N-2):N
    shifted_indices = (N+1):2*N
    
    
    M = CreateFullMatrix(A_dict, J, k, ind_vec = ind_vec, alpha=alpha, beta=beta)
    eigenvals, eigenvecs =  BogoliubovDiagonalizationForExp(M) # Ordering so that alpha and alpha^dagger are separated.

    W = eigenvecs
    V_BB = W[BB_ind, shifted_indices]
    V_AA = W[AA_ind, shifted_indices]
    exp_B = norm(V_BB)^2
    exp_A = norm(V_AA)^2
    
    return exp_A, exp_B
end


function VacuumStateFluctuations(
    A_dict::Dict{Int64, SparseMatrixCSC{Float64, Int64}},
    J::Float64;
    ind_vec = collect(1:66),
    N = 30,
    alpha = 1.0,
    beta = 1.0
)
    # Creates BZ
    kx_values = range(-3π/2, 3π/2, N)
    ky_values = range(-3π/2, 3π/2, N)

    Q_x = repeat(kx_values, 1, length(ky_values))
    Q_y = permutedims(repeat(ky_values, 1, length(kx_values)), (2, 1))

    # Create mask for points inside the hexagon
    mask = is_inside_hexagon.(Q_x, Q_y)

    # Get valid k points
    valid_kx = Q_x[mask]
    valid_ky = Q_y[mask]

    n_A = zeros(Float64, length(valid_kx))
    n_B = zeros(Float64, length(valid_kx))
    for m in eachindex(valid_kx)
        k_vec = [valid_kx[m], valid_ky[m]]
        n_A[m], n_B[m] = ExpNum_per_k(
            A_dict,
            J,
            k_vec,
            ind_vec = ind_vec,
            alpha = alpha,
            beta = beta
        )
    end
    exp_A = mean(n_A)
    exp_B = mean(n_B)
    return exp_A, exp_B
end
