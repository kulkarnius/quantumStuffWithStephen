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
wa = 1.0 * 2 * np.pi        #Frequency separation b/w |down> and |e> for A. The transition |down> -> |e> is coupled to the cavity
wb = 1.0 * 2 * np.pi        #Frequency separation b/w |down> and |e> for B. The transition |down> -> |e> is coupled to the cavity

wga = 0.1                   #Frequency separation b/w |up> and  |down> for A
wgb = 0.1                   #Frequency separation b/w |up> and  |down> for B

wCBroad = 1.0 * 2 * np.pi   #Resonant frequency for the broad plasmonic mode
wCNarrow = 1.0 * 2 * np.pi  #Resonant frequency for the narrow Fabry-Perot mode

"""Coupling Rates"""
gBroadA = 0.04               #Coupling rate for A to the Broad Mode
gNarrowA = 0.04              #Coupling rate for A to the Narrow Mode
gBroadB = 0.04               #Coupling rate for B to the Broad Mode
gNarrowB = 0.04               #Coupling rates for B to the Narrow Mode

"""Decay Rates"""
gammaA = 0.1                #Decay Rate for System A
gammaStarA = 0.02           #Dephasing Rate for System A
gammaB = 0.1                #Decay Rate for System B
gammaStarB = 0.02           #Dephasing Rate for System B

"""Mode Decay rates"""
ka = 0.02                   #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
kb = 0.04                   #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)


#Defining S Matrix
Svalues = [1,0.94,0.94,1] #[Saa, Sab, Sba, Sbb]
SMatrix = np.matrix([[Svalues[0], Svalues[1]],[Svalues[2], Svalues[3]]])

S_Sqrt = sp.linalg.sqrtm(SMatrix) #Taking Square root
S_InvSqrt = sp.linalg.sqrtm(sp.linalg.inv(SMatrix)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
wBroad = np.complex(wCBroad, -ka)
wNarrow = np.complex(wCNarrow, -kb)

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
XpValues = [Xp(0,0), Xp(0,1), Xp(1,0),Xp(1,1)] #X+ = [Xaa, Xab, Xba, Xbb]
XpMatrix = np.matrix([[XpValues[0], XpValues[1]],[XpValues[2], XpValues[3]]]) #Define Chi(X) Positive matrix

XmValues = [Xm(0,0),Xm(0,1),Xm(1,0),Xm(1,1)] #X- = [Xaa, Xab, Xba, Xbb]
XmMatrix = np.matrix([[XmValues[0], XmValues[1]],[XmValues[2], XmValues[3]]]) #Define Chi(X) Minus matrix

#Coupling Constants
def g(a):
    if a == "a":
        g = gBroadA * S_Sqrt[0][0] + gNarrowA * S_Sqrt[1][0]
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
def H(k):
    if k == "A":
        H = wa * eDownA.dag() * eDownA + wga * downUpA.dag() * downUpA
        return H
    else:
        H = wb * eDownB.dag() * eDownB + wgb * downUpB.dag() * downUpB
        return H

"""Defining System of Modes"""
def HC(c):
    if c == "Broad":
        H = XpValues[0] * aBroad.dag() * aBroad
        return H
    else:
        H = XpValues[3] * aNarrow.dag() * aNarrow
        return H

"""Interaction Hamiltonian""" 
def HI(i):
    HOverlap = XpValues[1] * aBroad.dag() * aNarrow + XpValues[2] * aNarrow.dag() * aBroad
    if i == "a":
        H = g("a") * eDownA * aBroad.dag() + np.conjugate(g("a")) * eDownA.dag() * aBroad + g("b") * eDownA * aNarrow.dag() + np.conjugate(g("b")) * eDownA.dag() * aNarrow
        HI = H + HOverlap
        return HI
    else:
        H = g("a") * eDownB * aBroad.dag() + np.conjugate(g("a")) * eDownB.dag() * aBroad + g("b") * eDownB * aNarrow.dag() + np.conjugate(g("b")) * eDownB.dag() * aNarrow
        HI = H + HOverlap
        return HI

"""Total Hamiltonian"""
H = H("A") + H("B") + HC("Broad") + HC("Narrow") + HI("a") + HI("b")

"""States"""
def psy(i,j,k,l):
  a = tensor(basis(3,i), basis(3,j), basis(2,k), basis(2,l))
  return a

"""Initial State"""
psi0 = 0.5 * (psy(2,1,0,0) + psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))   #tensor(|upA> + |downA>, |upB> + |downB>, |0>_Broad, |0>_Narrow)  )

"""Final Expected State"""
psIdeal = 0.5 * (-psy(2,1,0,0) + psy(2,0,0,0) + psy(0,1,0,0) + psy(0,0,0,0))


"""Basis Change"""
la, v = sp.linalg.eig(XmMatrix) #Solve Eigensystem
vca = v[0][0]
vcb = v[0][1]
vda = v[1][0]
vdb = v[1][1]

V = np.asmatrix(v)
l1, l2 = la #Parse Eigenvalues 

rootKappaAA = (1/(sp.linalg.det(V))) * (vca * vdb * np.sqrt(l1) - vcb * vda * np.sqrt(l2))
rootKappaAB = (1/(sp.linalg.det(V))) * (vcb * vdb * (np.sqrt(l1) - np.sqrt(l2)))
rootKappaBA = (1/(sp.linalg.det(V))) * (vca * vdb * np.sqrt(l2) - vcb * vda * np.sqrt(l1))
rootKappaBB = (1/(sp.linalg.det(V))) * (vca * vda * (np.sqrt(l2) - np.sqrt(l1)))

def c():
  c = vca*aBroad + vcb*aNarrow
  return c

def d():
  d = vda*aBroad + vdb*aNarrow
  return d


"""Collapse Operators"""
collapseOperators = []

# qubit relaxation
collapseOperators.append(np.sqrt(gammaA) * eDownA)
collapseOperators.append(np.sqrt(gammaB) * eDownB)

# qubit dephasing
collapseOperators.append(np.sqrt(gammaStarA) * eDownA.dag() * eDownA)
collapseOperators.append(np.sqrt(gammaStarB) * eDownB.dag() * eDownB)

# Weird Coupling Terms 
collapseOperators.append(np.sqrt(Xm(1,1)) * aNarrow)
collapseOperators.append(np.sqrt(Xm(0,0)) * aBroad)
collapseOperators.append(rootKappaAA * aBroad + rootKappaAB * aNarrow)
collapseOperators.append(rootKappaBA * aBroad + rootKappaBB * aNarrow)



"""Calculating Fidelity"""
tlist = np.linspace(0,500, 1000)                            #Time Steps
sol = mesolve(H, psi0, tlist, collapseOperators)            #Solver
fidel = [fidelity(state, psIdeal) for state in sol.states]  #Fidelity Calculation for each state at time steps defined

"""Plotting the Results"""
fig, axes = plt.subplots()
axes.plot(tlist, fidel)
plt.show()
