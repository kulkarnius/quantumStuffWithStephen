"""
Gate from 2nd scheme in
"Cavity-assisted controlled phase-flip gates" from F. Kimiaee, S.C.Wein, and C.Simon

System: Hughes Quasinormal Mode Paper

Author: Atharva Kulkarni
"""

"""Importing Libraries"""
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from qutip import *

"""Global Parameters"""
wa = 0       #Frequency separation b/w |down> and |e> for A. The transition |down> -> |e> is coupled to the cavity
wb = 0        #Frequency separation b/w |down> and |e> for B. The transition |down> -> |e> is coupled to the cavity

wga = 0                   #Frequency separation b/w |up> and  |down> for A
wgb = 0                   #Frequency separation b/w |up> and  |down> for B

wCBroad = 0.0   #Resonant frequency for the broad plasmonic mode
wCNarrow = np.linspace(0,1000,10000)  #Resonant frequency for the narrow Fabry-Perot mode

"""Coupling Rates"""
gBroadA = 0.0               #Coupling rate for A to the Broad Mode
gNarrowA = 0.1              #Coupling rate for A to the Narrow Mode
gBroadB = 0.0              #Coupling rate for B to the Broad Mode
gNarrowB = 0.1               #Coupling rates for B to the Narrow Mode

"""Decay Rates"""
gammaA = 0.000005                #Decay Rate for System A
gammaStarA = 0.0            #Dephasing Rate for System A
gammaB = 0.000005                #Decay Rate for System B
gammaStarB = 0.0            #Dephasing Rate for System B

"""Mode Decay rates"""
ka = 1                   #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
kb = 0.04                   #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)


#Defining S Matrix
Svalues = [1,0,0,1] #[Saa, Sab, Sba, Sbb]
SMatrix = np.matrix([[Svalues[0], Svalues[1]],[Svalues[2], Svalues[3]]])

S_Sqrt = sp.linalg.sqrtm(SMatrix) #Taking Square root
S_InvSqrt = sp.linalg.sqrtm(sp.linalg.inv(SMatrix)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
wBroad = np.complex(wCBroad, -ka)
wNarrow = [np.complex(Narrow, -kb) for Narrow in wCNarrow]

"""Defining Chi values"""
def X(i,j):
  X = wBroad * S_InvSqrt[i][0]*S_Sqrt[0][j] + wNarrow * S_InvSqrt[i][1]*S_Sqrt[1][j]
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
def Ga(i):
    if i == "Broad":
        g = gBroadA * S_Sqrt[0][0] + gNarrowA * S_Sqrt[1][0]
        return g
    else:
        g = gBroadA * S_Sqrt[0][1] + gNarrowA * S_Sqrt[1][1]
        return g

def Gb(i):
    if i == "Broad":
        g = gBroadB * S_Sqrt[0][0] + gNarrowB * S_Sqrt[1][0]
        return g
    else:
        g = gBroadB * S_Sqrt[0][1] + gNarrowB * S_Sqrt[1][1]
        return g


"""
Defining Operators: A similar tensor product between 4 systems. 
Here, the states in increasing order of energy for the systems are |up>, |down>, |e>
[System A, System B, Broad Mode, Narrow Mode]
"""

downUpA  = tensor(qutrit_ops()[3], qeye(3), qeye(2), qeye(2))             #System A Annhilation operator for transition from down to up
eDownA  = tensor(qutrit_ops()[4], qeye(3), qeye(2), qeye(2))              #System A Annhilation operator for transition from down to excited
downUpB  = tensor(qutrit_ops()[3], qeye(3), qeye(2), qeye(2))             #System B Annhilation operator for transition from down to up
eDownB  = tensor(qutrit_ops()[4], qeye(3), qeye(2), qeye(2))              #System B Annhilation operator for transition from down to excited
aBroad  = tensor(qeye(3), qeye(3), destroy(2), qeye(2))                   #Destruction operator for Broad mode
aNarrow  = tensor(qeye(3), qeye(3), qeye(2), destroy(2))                  #Destruction operator for Narrow mode

"""Defining Sytem of qubits"""
def Hk(k):
    if k == "A":
        H = wa * eDownA.dag() * eDownA + wga * downUpA.dag() * downUpA
        return H
    else:
        H = wb * eDownB.dag() * eDownB + wgb * downUpB.dag() * downUpB
        return H

"""Defining System of Modes"""
def HC(c):
    if c == "Broad":
        H = XpValues()[0] * aBroad.dag() * aBroad
        return H
    else:
        H = XpValues()[3] * aNarrow.dag() * aNarrow
        return H

"""Interaction Hamiltonian""" 
def HI(i):
    HOverlap = XpValues()[1] * aBroad.dag() * aNarrow + XpValues()[2] * aNarrow.dag() * aBroad
    if i == "a":
        H = Ga("Broad") * eDownA * aBroad.dag() + np.conjugate(Ga("Broad")) * eDownA.dag() * aBroad + Ga("Narrow") * eDownA * aNarrow.dag() + np.conjugate(Ga("Narrow")) * eDownA.dag() * aNarrow
        HI = H + HOverlap
        return HI
    else:
        H = Gb("Broad") * eDownB * aBroad.dag() + np.conjugate(Gb("Broad")) * eDownB.dag() * aBroad + Gb("Narrow") * eDownB * aNarrow.dag() + np.conjugate(Gb("Narrow")) * eDownB.dag() * aNarrow
        HI = H + HOverlap
        return HI

"""Total Hamiltonian"""
def H(): 
  H = Hk("A") + Hk("B") + HC("Broad") + HC("Narrow") + HI("a") + HI("b")
  return H

"""States"""
def psy(i,j,k,l):
  a = tensor(basis(3,i), basis(3,j), basis(2,k), basis(2,l))
  return a

"""Initial State"""
psi0 = 0.5 * (psy(2,1,0,0) + psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))   #tensor(|upA> + |downA>, |upB> + |downB>, |0>_Broad, |0>_Narrow)  )

"""Final Expected State"""
psIdeal = 0.5 * (-psy(2,1,0,0) + psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))


"""Basis Change"""
def eigensys(i):
  la, v = sp.linalg.eig(XmMatrix()) #Solve Eigensystem
  if i == "la":
    return la
  else:
    return v

def matrixfun():
  vca = eigensys("v")[0][0]
  vcb = eigensys("v")[0][1]
  vda = eigensys("v")[1][0]
  vdb = eigensys("v")[1][1]

  V = np.asmatrix(eigensys("v"))

  l1, l2 = eigensys("la") #Parse Eigenvalues 

  rootKappaAA = (1/(sp.linalg.det(V))) * (vca * vdb * np.sqrt(l1) - vcb * vda * np.sqrt(l2))
  rootKappaAB = (1/(sp.linalg.det(V))) * (vcb * vdb * (np.sqrt(l1) - np.sqrt(l2)))
  rootKappaBA = (1/(sp.linalg.det(V))) * (vca * vdb * np.sqrt(l2) - vcb * vda * np.sqrt(l1))
  rootKappaBB = (1/(sp.linalg.det(V))) * (vca * vda * (np.sqrt(l2) - np.sqrt(l1)))
  return rootKappaAA, rootKappaAB, rootKappaBA, rootKappaBB

def c():
  vca = eigensys("v")[0][0]
  vcb = eigensys("v")[0][1]
  c = vca*aBroad + vcb*aNarrow
  return c

def d():
  vda = eigensys("v")[1][0]
  vdb = eigensys("v")[1][1]
  d = vda*aBroad + vdb*aNarrow
  return d


"""Collapse Operators"""


def c_ops():
  collapseOperators = []
  # qubit relaxation
  collapseOperators.append(np.sqrt(gammaA) * eDownA)
  collapseOperators.append(np.sqrt(gammaB) * eDownB)

  # qubit dephasing
  collapseOperators.append(np.sqrt(gammaStarA) * eDownA.dag() * eDownA)
  collapseOperators.append(np.sqrt(gammaStarB) * eDownB.dag() * eDownB)

  # Weird Coupling Terms 
  collapseOperators.append(matrixfun()[0] * aBroad + matrixfun()[1] * aNarrow)
  collapseOperators.append(matrixfun()[2] * aBroad + matrixfun()[3] * aNarrow)
  return collapseOperators



"""Calculating Fidelity"""
tlist = np.linspace(0,500,1000)                             #Time Steps

fideliti = []

for wNarrow in wNarrow:
  sol = mesolve(H(), psi0, tlist, c_ops())          #Solver
  fidel = fidelity(sol.states[-1], psIdeal)  #Fidelity Calculation for each state at time steps defined
  fideliti.append(fidel)

"""Plotting the Results"""
fig, axes = plt.subplots()
axes.plot(wCNarrow, fideliti)
axes.set(xlabel="Cavity", ylabel="Fidelity", title='Fidelity as a function of wCNarrow')
plt.show()
