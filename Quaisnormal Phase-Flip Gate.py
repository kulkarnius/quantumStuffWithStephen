"""
Gate from 2nd scheme in
"Cavity-assisted controlled phase-flip gates" from F. Kimiaee, S.C.Wein, and C.Simon

System: Hughes Quasinormal Mode Paper

Author: Atharva Kulkarni
"""
""" 
Notation:
Narrow and Broad -> 1,2
Tilde: Modes before symmetrization
Hat: Modes before symmetrization
Subscript (+/-): Diagonalized basis (+ is larger eigenvalue) 
Qubits -> System A and System B
"""


"""Importing Libraries"""
import numpy as np
import time
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from qutip import *

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

Delta = 7.42
delta = 0.0756

wC1 = delta + Delta             #Resonant frequency for the broad plasmonic mode
wC2 = delta  #Resonant frequency for the narrow Fabry-Perot mode

"""Coupling Rates"""
g1A_tilde = 0.1               #Coupling rate for A to the Broad Mode
g2A_tilde = 0.001              #Coupling rate for A to the Narrow Mode
g1B_tilde = 0.1               #Coupling rate for B to the Broad Mode
g2B_tilde = 0.001              #Coupling rates for B to the Narrow Mode

"""Decay Rates"""
gammaA = 10e-7           #Decay Rate for System A
gammaStarA = 0.0            #Dephasing Rate for System A
gammaB = 10e-7           #Decay Rate for System B
gammaStarB = 0.0            #Dephasing Rate for System B

"""Mode Decay rates"""
k1 = 1                     #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
k2 = 0.01                     #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)

"""Cavity Cooperativity"""
C1A = 4 * (g1A_tilde**2) / (k1 * gammaA)
C2A =  4 * (g2A_tilde**2) / (k1 * gammaA)
C1B = 4 * (g1B_tilde**2) / (k2 * gammaB)
C2B = 4 * (g2B_tilde**2) / (k2 * gammaB)

opts = Options(nsteps=10000000, atol=1e-05, rtol=1e-04)

phi = -0.5 * np.pi
V = 0.0267

Sab = V * np.exp(np.complex(0, phi))
Sba = V * np.exp(np.complex(0, -phi))

#Defining S Matrix
Svalues = [1,Sab,Sba,1] #[Saa, Sab, Sba, Sbb]
SMatrix = np.matrix([[Svalues[0], Svalues[1]],[Svalues[2], Svalues[3]]])

S_Sqrt = sp.linalg.sqrtm(SMatrix) #Taking Square root
S_InvSqrt = sp.linalg.sqrtm(sp.linalg.inv(SMatrix)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
w1 = np.complex(wC1, -k1)
w2 = np.complex(wC2, -k2)

"""
Defining Operators: A similar tensor product between 4 systems. 
Here, the states in increasing order of energy for the systems are |up>, |down>, |e>
[System A, System B, Broad Mode, Narrow Mode]
"""

downUpA  = tensor(qutrit_ops()[3], qeye(3), qeye(2), qeye(2))             #System A Annhilation operator for transition from down to up
eDownA  = tensor(qutrit_ops()[4], qeye(3), qeye(2), qeye(2))              #System A Annhilation operator for transition from down to excited
downUpB  = tensor(qeye(3),qutrit_ops()[3],  qeye(2), qeye(2))             #System B Annhilation operator for transition from down to up
eDownB  = tensor(qeye(3), qutrit_ops()[4], qeye(2), qeye(2))              #System B Annhilation operator for transition from down to excited
a1  = tensor(qeye(3), qeye(3), destroy(2), qeye(2))                       #Destruction operator for Broad mode
a2  = tensor(qeye(3), qeye(3), qeye(2), destroy(2))                       #Destruction operator for Narrow mode

def main(tfinal):

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


  """Chi(X) matrix definition """
  def XpValues():
    X = [Xp(0,0), Xp(0,1), Xp(1,0),Xp(1,1)] #X+ = [Xaa, Xab, Xba, Xbb]
    return X

  def XpMatrix(): 
    X = np.matrix([[XpValues()[0], XpValues()[1]],[XpValues()[2], XpValues()[3]]]) #Define Chi(X) Positive matrix
    return X

  def XmValues():
    X = [Xm(0,0),Xm(0,1),Xm(1,0),Xm(1,1)] #X- = [Xaa, Xab, Xba, Xbb]
    return X

  def XmMatrix():
    X = np.matrix([[XmValues()[0], XmValues()[1]],[XmValues()[2], XmValues()[3]]]) #Define Chi(X) Minus matrix
    return X

  #Coupling Constants

  ga1 = g1A_tilde * S_Sqrt[0][0] + g2A_tilde * S_Sqrt[1][0]
  ga2 = g1A_tilde * S_Sqrt[0][1] + g2A_tilde * S_Sqrt[1][1]

  gb1 = g1B_tilde * S_Sqrt[0][0] + g2B_tilde * S_Sqrt[1][0]
  gb2 = g1B_tilde * S_Sqrt[0][1] + g2B_tilde * S_Sqrt[1][1]
      

  """Defining Sytem of qubits"""
  HA = wa * eDownA.dag() * eDownA - wga * downUpA.dag() * downUpA
  HB = wb * eDownB.dag() * eDownB - wgb * downUpB.dag() * downUpB

  """Defining System of Modes"""
  HC1 = XpValues()[0] * a1.dag() * a1
  HC2 = XpValues()[3] * a2.dag() * a2

  """Interaction Hamiltonian""" 
  HOverlap = XpValues()[1] * a1.dag() * a2 + XpValues()[2] * a2.dag() * a1
  HIa = ga1 * eDownA * a1.dag() + np.conjugate(ga1) * eDownA.dag() * a1 + ga2 * eDownA * a2.dag() + np.conjugate(ga2) * eDownA.dag() * a2
  HIb = gb1 * eDownB * a1.dag() + np.conjugate(gb1) * eDownB.dag() * a1 + gb2 * eDownB * a2.dag() + np.conjugate(gb2) * eDownB.dag() * a2


  """Total Hamiltonian"""

  H = HA + HB + HC1 + HC2 + HIa + HIb + HOverlap

  """Basis Change"""
  la, v = sp.linalg.eig(XmMatrix()) #Solve Eigensystem

  vca = v[0][0] / np.sqrt(abs(v[0][0])**2 + abs(v[0][1])**2) #adding normalization manually
  vcb = v[0][1] / np.sqrt(abs(v[0][0])**2 + abs(v[0][1])**2)
  vda = v[1][0] / np.sqrt(abs(v[1][0])**2 + abs(v[1][1])**2)
  vdb = v[1][1] / np.sqrt(abs(v[1][0])**2 + abs(v[1][1])**2)

  #V = np.asmatrix(v)

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

  tlist = [0, tfinal]
  final_state = mesolve(H, psi0, tlist, collapseOperators, e_ops=[], args={}, options=opts).states[-1]

  fideliti = fidelity(final_state, psIdeal)

  print(fideliti)

  return fideliti

"""Calculating Fidelity"""

times = 118233

fidel = main(times)

"""Plotting the Results"""
fig, axes = plt.subplots()
#axes.plot(wC2, fidel)
#axes.set(xlabel="Cavity", ylabel="Fidelity", title='Fidelity as a function of wCNarrow')
#plt.show()

"""print(1 - np.pi / np.sqrt(CNarrowA) + 36* np.pi**2 / (32 * CNarrowA))
print(np.sqrt(CNarrowA)/2)
"""
