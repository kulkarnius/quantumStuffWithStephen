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

"""Final Expected State"""
psIdeal = 0.5 * (psy(3,2,0,0) - psy(3,1,0,0) + psy(1,2,0,0) + psy(1,1,0,0))

rhoIdeal = tensor(psIdeal,dagger(psIdeal))

function myFidelity(t, p)
  fidelity(p, rhoIdeal)
end

"""Global Parameters"""
wa = 0.0                    #Frequency separation b/w |down> and |e> for A. The transition |down> -> |e> is coupled to the cavity
wb = 0.0                   #Frequency separation b/w |down> and |e> for B. The transition |down> -> |e> is coupled to the cavity

wga = 0.0                  #Frequency separation b/w |up> and  |down> for A
wgb = 0.0                  #Frequency separation b/w |up> and  |down> for B

w1_tilde = 1             #Resonant frequency for the broad plasmonic mode
w2_tilde = 1:1:1000 #Resonant frequency for the narrow Fabry-Perot mode

"""Coupling Rates"""
g1A_tilde = 0.1               #Coupling rate for A to the Broad Mode
g2A_tilde = 0.01              #Coupling rate for A to the Narrow Mode
g1B_tilde = 0.1               #Coupling rate for B to the Broad Mode
g2B_tilde = 0.01              #Coupling rates for B to the Narrow Mode

"""Decay Rates"""
gammaA = 0.000005           #Decay Rate for System A
gammaStarA = 0.0            #Dephasing Rate for System A
gammaB = 0.000005           #Decay Rate for System B
gammaStarB = 0.0            #Dephasing Rate for System B

"""Mode Decay rates"""
ka = 0.1                     #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
kb = 4.0                     #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)

"""Cavity Cooperativity"""
C1A = 4 * (g1A_tilde^2) / (ka * gammaA)
C2A =  4 * (g2A_tilde^2) / (ka * gammaA)
C1B = 4 * (g1B_tilde^2) / (kb * gammaB)
C2B = 4 * (g2B_tilde^2) / (kb * gammaB)

#Defining S Matrix
SMat = [[1,0.94] [0.94,1]] #[Saa, Sab, Sba, Sbb]

S_Sqrt = sqrt(SMat) #Taking Square root
S_InvSqrt = sqrt(inv(SMat)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
w1 = complex(w1_tilde, -ka)
#omega2 = [complex(Narrow, -kb) for Narrow=w2_tilde]


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

results = []

for omega in w2_tilde
  w2 = complex(omega, -kb)

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

  v = eigvecs(XMinus)

  vca = v[1,1] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2) #adding normalization manually
  vcb = v[1,2] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2)
  vda = v[2,1] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)
  vdb = v[2,2] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)

  l1 = la[1]
  l2 = la[2] #Parse Eigenvalues 

  c = vca*a1 + vcb*a2
  d = vda*a1 + vdb*a2

  """Collapse Operators"""
  J = []

  push!(J,sqrt(gammaA)*eDownA)
  push!(J,sqrt(gammaB)*eDownB)
  push!(J,sqrt(gammaStarA)*dagger(eDownA) *eDownA)
  push!(J,sqrt(gammaStarB)*dagger(eDownB) *eDownB)
  push!(J,sqrt(l1)*c)
  push!(J,sqrt(l2)*d)

  tfinal = 20000
  tlist = [0,tfinal]

  tout, fideliti = timeevolution.master(tlist, psi0, H, J; fout=myFidelity)


  push!(results, last(real(fideliti)))
end





    




plot(omega2, results, color="blue")
show()