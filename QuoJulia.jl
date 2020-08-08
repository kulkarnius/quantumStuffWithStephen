using QuantumOptics
using PyPlot
using LinearAlgebra

"""States"""
sysbas = NLevelBasis(3)
cavbas = FockBasis(2)
function psy(i,j,k,l)
  a = tensor(nlevelstate(sysbas,i), nlevelstate(sysbas,j), fockstate(cavbas,k), fockstate(cavbas,l))
end

"""Initial State"""
psi0 = 0.5 * (psy(2,2,1,1) + psy(2,1,1,1) + psy(3,2,1,1) + psy(3,1,1,1))        #tensor(|upA> + |downA>, |upB> + |downB>, |0>_Broad, |0>_Narrow|)

"""Final Expected State"""
psIdeal = 0.5 * (psy(2,2,1,1) + psy(2,1,1,1) + psy(3,2,1,1) - psy(3,1,1,1))

"""Global Parameters"""
wa = 0.0                    #Frequency separation b/w |down> and |e> for A. The transition |down> -> |e> is coupled to the cavity
wb = 0.0                   #Frequency separation b/w |down> and |e> for B. The transition |down> -> |e> is coupled to the cavity

wga = 0.0                  #Frequency separation b/w |up> and  |down> for A
wgb = 0.0                  #Frequency separation b/w |up> and  |down> for B

wCBroad = 0.0             #Resonant frequency for the broad plasmonic mode
wCNarrow = [1:1:100] #Resonant frequency for the narrow Fabry-Perot mode

"""Coupling Rates"""
gBroadA = 0.0               #Coupling rate for A to the Broad Mode
gNarrowA = 0.1              #Coupling rate for A to the Narrow Mode
gBroadB = 0.0               #Coupling rate for B to the Broad Mode
gNarrowB = 0.1              #Coupling rates for B to the Narrow Mode

"""Decay Rates"""
gammaA = 0.000005           #Decay Rate for System A
gammaStarA = 0.0            #Dephasing Rate for System A
gammaB = 0.000005           #Decay Rate for System B
gammaStarB = 0.0            #Dephasing Rate for System B

"""Mode Decay rates"""
ka = 1.0                     #Broad Mode decay rate (I have currently used "a" and Broad interchangably)
kb = 1.0                     #Narrow Mode decay rate (I have currently used "b" and Narrow interchangably)

"""Cavity Cooperativity"""
CBroadA = 4 * (gBroadA^2) / (ka * gammaA)
CNarrowA =  4 * (gNarrowA^2) / (ka * gammaA)
CBroadB = 4 * (gBroadB^2) / (kb * gammaB)
CNarrowB = 4 * (gNarrowB^2) / (kb * gammaB)

#Defining S Matrix
SMat = [[1,0] [0,1]] #[Saa, Sab, Sba, Sbb]

S_Sqrt = sqrt(SMat) #Taking Square root
S_InvSqrt = sqrt(inv(SMat)) #Negative sqrt by taking an inversion and then a square root

#Defining Complex Quasinormal mode frequencies
wBroad = wCBroad - im*ka
#wNarrow = [(Narrow - im*kb) for Narrow in wCNarrow]

"""
Defining Operators: A similar tensor product between 4 systems. 
Here, the states in increasing order of energy for the systems are |up>, |down>, |e>
[System A, System B, Broad Mode, Narrow Mode]
"""

t21 = transition(sysbas, 1, 2)
t32 = transition(sysbas, 2, 3)

coupled = tensor(sysbas,sysbas, cavbas,cavbas)


downUpA  = tensor(t21, identityoperator(sysbas), identityoperator(cavbas), identityoperator(cavbas))             #System A Annhilation operator for transition from down to up
eDownA  = tensor(t32, identityoperator(sysbas), identityoperator(cavbas), identityoperator(cavbas))              #System A Annhilation operator for transition from down to excited
downUpB  = tensor(identityoperator(sysbas), t21, identityoperator(cavbas), identityoperator(cavbas))            #System B Annhilation operator for transition from down to up
eDownB  = tensor(identityoperator(sysbas), t32, identityoperator(cavbas), identityoperator(cavbas))                #System B Annhilation operator for transition from down to excited
aBroad  = tensor(identityoperator(sysbas), identityoperator(sysbas), destroy(cavbas), identityoperator(cavbas))                   #Destruction operator for Broad mode
aNarrow  = tensor(identityoperator(sysbas), identityoperator(sysbas), identityoperator(cavbas), destroy(cavbas))                  #Destruction operator for Narrow mode




wN = complex(sqrt(CNarrowA) * 0.5, -kb)

"""Defining Chi values"""
function X(i,j)
  X = wBroad * S_InvSqrt[i,1] * S_Sqrt[1,j] + wN * S_InvSqrt[i,2] * S_Sqrt[2,j]
end
  
function Xp(i,j)
  Xp = (X(i,j) + conj(X(j,i)))/2
end

function Xm(i,j)
  Xm = 0.5*im*(X(i,j) - conj(X(j,i)))
end


"""Chi(X) matrix definition """

XPlus = [[Xp(1,1), Xp(2,1)] [Xp(2,1), Xp(2,2)]] #X+ = [Xaa, Xab, Xba, Xbb]
XMinus = [[Xm(1,1),Xm(2,1)] [Xm(1,2),Xm(2,2)]] #X- = [Xaa, Xab, Xba, Xbb]

#Coupling Constants
function Ga(i)
    if i == "Broad"
        g = gBroadA * S_Sqrt[1,1] + gNarrowA * S_Sqrt[2,1]
        return g
    else
        g = gBroadA * S_Sqrt[1,2] + gNarrowA * S_Sqrt[2,2]
        return g
    end
  end

function Gb(i)
    if i == "Broad"
          g = gBroadB * S_Sqrt[1,1] + gNarrowB * S_Sqrt[2,1]
        return g
    else
        g = gBroadB * S_Sqrt[1,2] + gNarrowB * S_Sqrt[2,2]
        return g
    end
  end

"""Defining Sytem of qubits"""
function Hk(k)
    if k == "A"
        H = wa * dagger(eDownA) * eDownA - wga * dagger(downUpA) * downUpA
        return H
    else
        H = wb * dagger(eDownB) * eDownB - wgb * dagger(downUpB) * downUpB
        return H
    end
  end

"""Defining System of Modes"""
function HC(c)
    if c == "Broad"
        H = XPlus[1,1] * dagger(aBroad) * aBroad
        return H
    else
        H = XPlus[2,2] * dagger(aNarrow) * aNarrow
        return H
    end
  end

"""Interaction Hamiltonian""" 
function HI(i)
    HOverlap = XPlus[1,2] * dagger(aBroad) * aNarrow + XPlus[2,1] * dagger(aNarrow) * aBroad
    if i == "a"
        H = Ga("Broad") * eDownA * dagger(aBroad) + conj(Ga("Broad")) * dagger(eDownA) * aBroad + Ga("Narrow") * eDownA * dagger(aNarrow) + conj(Ga("Narrow")) * dagger(eDownA) * aNarrow
        HI = H + HOverlap
        return HI
    else
        H = Gb("Broad") * eDownB * dagger(aBroad) + conj(Gb("Broad")) * dagger(eDownB) * aBroad + Gb("Narrow") * eDownB * dagger(aNarrow) + conj(Gb("Narrow")) * dagger(eDownB) * aNarrow
        HI = H
        return HI
    end
  end

"""Total Hamiltonian"""

H = Hk("A") + Hk("B") + HC("Broad") + HC("Narrow") + HI("a") + HI("b")

"""Basis Change"""
la = eigvals(XMinus) #Solve Eigensystem

v = eigvecs(XMinus)

vca = v[1,1] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2) #adding normalization manually
vcb = v[1,2] / sqrt(abs(v[1,1])^2 + abs(v[1,2])^2)
vda = v[2,1] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)
vdb = v[2,2] / sqrt(abs(v[2,1])^2 + abs(v[2,2])^2)

l1 = la[1]
l2 = la[2] #Parse Eigenvalues 

function c()
  c = vca*aBroad + vcb*aNarrow
end 

function d()
  d = vda*aBroad + vdb*aNarrow
end


"""Collapse Operators"""
J = [eDownA,eDownB, dagger(eDownA) * eDownA, dagger(eDownB) * eDownB, c(), d()]

rates = [gammaA,gammaB,gammaStarA,gammaStarB,l1,l2]
tlist = range(0,stop=(pi*real(wN)/(gNarrowA^2)), length=2)

pt = timeevolution.master(tlist, psi0, H, J; rates=rates)
println(pt[-1])



#function fidelity(a)
#    fide = tr(sqrt(real(sqrt(a) * (psIdeal * sqrt(a)))))
#end

#var = sqrt(CNarrowA) * 0.5
#println(main(10))

