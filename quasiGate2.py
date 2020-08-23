#Grabbing in necessary libraries
from qutip import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

"""States"""
def psy(i,j,k,l):
  a = tensor(basis(3,i), basis(3,j), basis(2,k), basis(2,l))
  return a

"""Initial State"""
psi0 = 0.5 * (psy(2,1,0,0) + psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))        #tensor(|upA> + |downA>, |upB> + |downB>, |0>_Broad, |0>_Narrow|)

"""Final Expected State"""
psIdeal = 0.5 * (psy(2,1,0,0) - psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))

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
g1A_tilde = np.sqrt(C1A*k1*gammaA/4)               #Coupling rate for A to the Broad Mode
g2A_tilde = np.sqrt(C2A*k2*gammaA/4)              #Coupling rate for A to the Narrow Mode
g1B_tilde = np.sqrt(C1B*k1*gammaB/4)               #Coupling rate for B to the Broad Mode
g2B_tilde = np.sqrt(C2B*k2*gammaB/4)              #Coupling rates for B to the Narrow Mode

opts = Options(nsteps=10000000)

phi = -0.5 * np.pi
V = 2*np.sqrt(k1*k2/((wC1-wC2)**2+(k1+k2)**2))

Sab = V * np.exp(np.complex(0, -phi))
Sba = V * np.exp(np.complex(0, phi))

#Defining S Matrix
Svalues = [1,Sab,Sba,1] #[Saa, Sab, Sba, Sbb]
SMatrix = np.matrix([[Svalues[0], Svalues[1]],[Svalues[2], Svalues[3]]])

S_Sqrt = sp.linalg.sqrtm(SMatrix) #Taking Square root
S_InvSqrt = sp.linalg.sqrtm(sp.linalg.inv(SMatrix)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
w1 = np.complex(wC1, -k1)
w2 = np.complex(wC2, -k2)

def main():
  """Defining Chi values"""
  def X(i,j):
    X = w1 * S_InvSqrt[i][0]*S_Sqrt[0][j] + w2 * S_InvSqrt[i][1]*S_Sqrt[1][j]
    return X
    
  def Xp(i,j):
    Xp = (X(i,j) + np.conjugate(X(j,i)))/2
    return Xp

  def Xm(i,j):
    Xm = np.complex(0,(X(i,j) - np.conjugate(X(j,i)))/2)
    return Xm


  """Chi matrices definition """
  XPlus = [[Xp(0,0), Xp(0,1)], [Xp(1,0),Xp(1,1)]] #X+ = [Xaa, Xab, Xba, Xbb]
  XPlusM = np.asmatrix(XPlus) #Define Chi(X) Positive matrix

  XMinus = [[Xm(0,0),Xm(0,1)],[Xm(1,0),Xm(1,1)]] #X- = [Xaa, Xab, Xba, Xbb]
  XMinusM = np.asmatrix(XMinus) #Define Chi(X) Minus matrix

  #Coupling Constants

  ga1 = g1A_tilde * S_Sqrt[0][0] + g2A_tilde * S_Sqrt[1][0]
  ga2 = g1A_tilde * S_Sqrt[0][1] + g2A_tilde * S_Sqrt[1][1]

  gb1 = g1B_tilde * S_Sqrt[0][0] + g2B_tilde * S_Sqrt[1][0]
  gb2 = g1B_tilde * S_Sqrt[0][1] + g2B_tilde * S_Sqrt[1][1]
      

  """Defining Sytem of qubits"""
  HA = wa * eDownA.dag() * eDownA - wga * downUpA.dag() * downUpA
  HB = wb * eDownB.dag() * eDownB - wgb * downUpB.dag() * downUpB

  """Defining System of Modes"""
  HC1 = XPlus[0][0] * a1.dag() * a1
  HC2 = XPlus[1][1] * a2.dag() * a2

  """Interaction Hamiltonian""" 
  HOverlap = XPlus[0][1] * a1.dag() * a2 + XPlus[1][0] * a2.dag() * a1
  HIa = ga1 * eDownA * a1.dag() + np.conjugate(ga1) * eDownA.dag() * a1 + ga2 * eDownA * a2.dag() + np.conjugate(ga2) * eDownA.dag() * a2
  HIb = gb1 * eDownB * a1.dag() + np.conjugate(gb1) * eDownB.dag() * a1 + gb2 * eDownB * a2.dag() + np.conjugate(gb2) * eDownB.dag() * a2


  """Total Hamiltonian"""

  H = HA + HB + HC1 + HC2 + HIa + HIb + HOverlap

  """Basis Change"""
  la, v = sp.linalg.eig(XMinusM) #Solve Eigensystem

  v = sp.linalg.inv(v)

  vca = v[0][0] 
  vcb = v[0][1] 
  vda = v[1][0] 
  vdb = v[1][1]


  l1, l2 = la #Parse Eigenvalues 

  c = vca*a1 + vcb*a2

  d = vda*a1 + vdb*a2


  """Collapse Operators"""
  collapseOperators = []
  # qubit relaxation
  collapseOperators.append(np.sqrt(gammaA) * eDownA)
  collapseOperators.append(np.sqrt(gammaB) * eDownB)

  # qubit dephasing
  collapseOperators.append(np.sqrt(gammaStarA) * eDownA.dag() * eDownA)
  collapseOperators.append(np.sqrt(gammaStarB) * eDownB.dag() * eDownB)

  # Weird Coupling Terms
  collapseOperators.append(np.sqrt(l1) * c)
  collapseOperators.append(np.sqrt(l2) * d)

  """Constructing the Liouvillian Superoperator"""
  L = liouvillian(H,c_ops=collapseOperators)

  """Matrix Exponential"""
  op = (tfinal * L).expm()

  """Density Matrix Calculations"""

  rho0 = ket2dm(psi0)                         #Defining Density Matrix

  vec_rho0 = operator_to_vector(rho0)         #Vectorizing density matrix

  vec_rhofinal = op * vec_rho0                #Applying superoperator to vectorized operator

  rhofinal = vector_to_operator(vec_rhofinal) #Converting final vector into density matrix  

  return fidelity(rhofinal, psIdeal)

print(main())

