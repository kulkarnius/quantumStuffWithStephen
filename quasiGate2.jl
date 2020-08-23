using QuantumOptics
using PyPlot
using LinearAlgebra

"""States"""
sysbas = NLevelBasis(3)
cavbas = FockBasis(1)
function psy(i,j,k,l)
  a = tensor(nlevelstate(sysbas,i), nlevelstate(sysbas,j), fockstate(cavbas,k), fockstate(cavbas,l))
end

"""Initial State"""
psi0 = 0.5 * (psy(3,2,0,0) + psy(3,1,0,0) + psy(1,2,0,0) + psy(1,1,0,0))        #tensor(|upA> + |downA>, |upB> + |downB>, |0>_Broad, |0>_Narrow|)
rho0 = dm(psi0)

"""Final Expected State"""
psIdeal = 0.5 * (psy(3,2,0,0) - psy(3,1,0,0) + psy(1,2,0,0) + psy(1,1,0,0))
rhoIdeal = dm(psIdeal)


"""Global Parameters"""
wa = 0.0                    #Frequency separation b/w |down> and |e> for A. The transition |down> -> |e> is coupled to the cavity
wb = 0.0                   #Frequency separation b/w |down> and |e> for B. The transition |down> -> |e> is coupled to the cavity

wga = 0.0                  #Frequency separation b/w |up> and  |down> for A
wgb = 0.0                  #Frequency separation b/w |up> and  |down> for B

Delta = 600.
delta = 6.

wC1 = delta + Delta             #Resonant frequency for the broad plasmonic mode
wC2 = delta

"""Gate Time"""
tfinal = 315.722  

"""Cavity Cooperativity"""
C1A = 1200.
C2A = 1200.
C1B = 1200.
C2B = 1200.

"""Decay Rates"""
gammaA = 1e-4           #Decay Rate for System A
gammaStarA = 0.0            #Dephasing Rate for System A
gammaB = 1e-4           #Decay Rate for System B
gammaStarB = 0.0            #Dephasing Rate for System B

"""Mode Decay rates"""
k1 = 100.                    #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
k2 = 1.                     #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)

"""Coupling Rates"""
g1A_tilde = sqrt(C1A*k1*gammaA/4)               #Coupling rate for A to the Broad Mode
g2A_tilde = sqrt(C2A*k2*gammaA/4)              #Coupling rate for A to the Narrow Mode
g1B_tilde = sqrt(C1B*k1*gammaB/4)               #Coupling rate for B to the Broad Mode
g2B_tilde = sqrt(C2B*k2*gammaB/4)              #Coupling rates for B to the Narrow Mode

phi = -0.5 * pi
V = 2*sqrt(k1*k2/((wC1-wC2)^2+(k1+k2)^2))

Sab = V * exp(-im*phi)
Sba = V * exp(im*phi)
#Defining S Matrix
SMat = [[1,Sab] [Sba,1]] #[Saa, Sab, Sba, Sbb]

S_Sqrt = sqrt(SMat) #Taking Square root
S_InvSqrt = sqrt(inv(SMat)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
w1 = complex(w1_tilde, -k1)
w2 = complex(w2_tilde, -k2)


"""
Defining Operators: A similar tensor product between 4 systems. 
Here, the states in increasing order of energy for the systems are |up>, |down>, |e>
[System A, System B, Broad Mode, Narrow Mode]
"""

sgm21 = transition(sysbas, 1, 2)
sgm32 = transition(sysbas, 2, 3)

coupled = tensor(sysbas,sysbas, cavbas,cavbas)

downUpA  = embed(coupled, 1, sgm21)        #System A Annhilation operator for transition from down to up
eDownA  = embed(coupled, 1, sgm32)            #System A Annhilation operator for transition from down to excited
downUpB  = embed(coupled, 2, sgm21)          #System B Annhilation operator for transition from down to up
eDownB  = embed(coupled, 2, sgm32)              #System B Annhilation operator for transition from down to excited
a1  = embed(coupled, 3, destroy(cavbas))                  #Destruction operator for Broad mode
a2  = embed(coupled, 4, destroy(cavbas))                 #Destruction operator for Narrow mode


"""Defining Chi values"""
function X(i,j)
  X = w1 * S_InvSqrt[i,1] * S_Sqrt[1,j] + w2 * S_InvSqrt[i,2] * S_Sqrt[2,j]
  return X
end
  

"""Chi(X) matrix definition """

XPlus = [[(X(1,1) + conj(X(1,1)))/2, (X(2,1) + conj(X(1,2)))/2] [(X(1,2) + conj(X(2,1)))/2, (X(2,2) + conj(X(1,2)))/2]] #X+ = [Xaa, Xab, Xba, Xbb]
XMinus = [[0.5*im*(X(1,1) - conj(X(1,1))), 0.5*im*(X(2,1) - conj(X(1,2)))] [0.5*im*(X(1,2) - conj(X(2,1))),0.5*im*(X(2,2) - conj(X(2,2)))]] #X- = [Xaa, Xab, Xba, Xbb]

#Coupling Constants
ga1 = g1A_tilde * S_Sqrt[1,1] + g2A_tilde * S_Sqrt[2,1]
ga2 = g1A_tilde * S_Sqrt[1,2] + g2A_tilde * S_Sqrt[2,2]

gb1 = g1B_tilde * S_Sqrt[1,1] + g2B_tilde * S_Sqrt[2,1]
gb2 = g1B_tilde * S_Sqrt[1,2] + g2B_tilde * S_Sqrt[2,2]


"""Defining Sytem of qubits"""
HA = wa * dagger(eDownA) * eDownA - wga * dagger(downUpA) * downUpA
HB = wb * dagger(eDownB) * eDownB - wgb * dagger(downUpB) * downUpB

"""Defining System of Modes"""
H1 = XPlus[1,1] * dagger(a1) * a1
H2 = XPlus[2,2] * dagger(a2) * a2

"""Interaction Hamiltonian""" 
HOverlap = XPlus[1,2] * dagger(a1) * a2 + XPlus[2,1] * dagger(a2) * a1
HIA = ga1 * eDownA * dagger(a1) + conj(ga1) * dagger(eDownA) * a1 + ga2 * eDownA * dagger(a2) + conj(ga2) * dagger(eDownA) * a2
HIB = gb1 * eDownB * dagger(a1) + conj(gb1) * dagger(eDownB) * a1 + gb2 * eDownB * dagger(a2) + conj(gb2) * dagger(eDownB) * a2

"""Total Hamiltonian"""

H = HA + HB + H1 + H2 + HIA + HIB + HOverlap

"""Basis Change"""
la = eigvals(XMinus) #Solve Eigensystem

v = inv(eigvecs(XMinus))

vca = v[1,1] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2) #adding normalization manually
vcb = v[1,2] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2)
vda = v[2,1] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)
vdb = v[2,2] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)

l1 = la[1]
l2 = la[2] #Parse Eigenvalues 

c = vca*a1 + vcb*a2
d = vda*a1 + vdb*a2

"""Collapse Operators"""
J = [eDownA, eDownB, dagger(eDownA)*eDownA, dagger(eDownB)*eDownB, c, d]

rates = [gammaA, gammaB, gammaStarA, gammaStarB, l1, l2]

"""Defining Liouvillian"""
L = liouvillian(H, J; rates)

"""Matrix Exponential"""
op = exp(tfinal * L)

"""Density Matrix Calculations"""

vec_rho0 = vec(rho0)         #Vectorizing density matrix

vec_rhofinal = op * vec_rho0                #Applying superoperator to vectorized operator

rhofinal = reshape(vec_rhofinal, 36, 36) #Converting final vector into density matrix  

print(idelity(rhofinal,rhoIdeal))