using QuantumOptics
using PyPlot


#Parameters
N_cutoff = 10

wc = 0.1
wa = 0.1
g = 1.

#Bases
b_fock = FockBasis(N_cutoff)
b_spin = NLevelBasis(2)

b = tensor(b_fock, b_spin)

a = destroy(b_fock)
at = create(b_fock) 
n = number(b_fock)

sm = transition(b_spin, 1, 2)
sp = transition(b_spin, 2, 1)
sz = transition(b_spin, 1, 1)


#Hamiltonian
Hatom = wa*sz/2
Hfield = wc*n
Hint = g*(tensor(at,sm) + tensor(a,sp))
H = tensor(one(b_fock), Hatom) + tensor(Hfield, one(b_spin)) + Hint

#Initial state
a = 1.

psi0 = tensor(fockstate(b_fock, 1), nlevelstate(b_spin, 2))

#Integration Time
T = [0:0.1:20;]

γ = 0.5
J = [sqrt(γ)*identityoperator(b_fock) ⊗ sm]

# Master
tout, pt = timeevolution.master(T, psi0, H, J)

rhoInit = tensor(psi0, dagger(psi0))

print(fidelity(last(pt), rhoInit))