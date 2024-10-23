import numpy as np
import sys
import itertools
from numba import njit
import copy, time
#sys.path.append('/mnt/home/daviso53/Research/pyci/imsrg_ci/')
#import pyci_pairing_plus_ph as pyci

debug=False

def state2occupation(state):
    return np.asarray(list(zip(state[1],state[2]))).flatten()

@njit(debug=debug)#(cache=True)
def creator(phi, p):
    # if isinstance(phi,int):
    #     return 0
    empty = np.array([np.float64(x) for x in range(0)])

    idx = p+1
    phi = np.copy(phi)
    try:
        phase = (-1)**(np.sum(phi[1:idx]))
    except:
        return empty

    if phi[idx] == 0:
        phi[idx] = 1
        phi[0] = phi[0]*phase
        return phi
    else:
        return empty

@njit(debug=debug)#(cache=True)
def annihilator(phi, p):
    # if isinstance(phi,int):
    #     return 0
    empty = np.array([np.float64(x) for x in range(0)])

    idx = p+1
    phi = np.copy(phi)

    try:
        phase = (-1)**(np.sum(phi[1:idx]))
    except:
        return empty

    if phi[idx] == 1:
        phi[idx] = 0
        phi[0] = phi[0]*phase
        return phi
    else:
        return empty
@njit(debug=debug)#(cache=True)
def inner_product(phi_bra, phi_ket):

    # if isinstance(phi_bra, int) or isinstance(phi_ket, int):
    #     return 0

    try:
        temp1 = phi_bra[0]
        temp2 = phi_ket[0]
    except:
        return 0.0

    if np.array_equal(phi_bra[1::], phi_ket[1::]):
        return 1.0*phi_bra[0]*phi_ket[0]
    else:
        return 0.0

def gen_basis(n_holes, n_particles):

    config_base = np.append(np.ones(int(n_holes)),
                               np.zeros(int(n_particles)))
    config_str = ''.join([str(int(state)) for state in config_base])

    states = list(map("".join, itertools.permutations(config_str)))
    states = list(dict.fromkeys(states)) # remove duplicates
    states = [list(map(int, list(ref))) for ref in states]#.sort(key=number_of_pairs, reverse=True)

    # # COLLECT TOTAL SPIN=0 REFS
    states_s0 = []
    for state in states:
        S = 0
        for i in range(len(state)):
            if state[i] == 1:
                if i%2 == 0:
                    S += 1
                else:
                    S += -1
            else: continue
        if S == 0:
            states_s0.append(state)
            
    states = states_s0

    states.sort(key=number_of_pairs, reverse=True)        

    states = np.asarray(states)

    # append column of 1's for the phase bit
    states = np.hstack((np.ones((states.shape[0],1)), states))

    # AVEN'S ORIGINAL CODE FOR GENERATING BASIS
    # The problem here is that n_holes and n_particles must be even numbers.
    # -------------------------------------------------------------------------

    # up_config_base = np.append(np.ones(int(n_holes/2)),
    #                            np.zeros(int(n_particles/2)))
    # up_config_str = ''.join([str(int(state)) for state in up_config_base])

    # up_states = list(map("".join, itertools.permutations(up_config_str)))
    # up_states = list(dict.fromkeys(up_states)) # remove duplicates
    # up_states = [list(map(int, list(ref))) for ref in up_states]

    # dn_states = up_states

    # states = [] # [[phase, up_vector, dn_vector], ...]
    # for i in range(0, len(up_states)):
    #     for j in range(0, len(dn_states)):
    #         states.append([1, copy.deepcopy(up_states[i]), copy.deepcopy(dn_states[j])])

    # #sort by number of pairs
    # states.sort(key=number_of_pairs, reverse=True)

    # statesb = np.array([np.array(state2occupation(state)) for state in states])
    # statesb = np.hstack((np.ones((statesb.shape[0],1)), statesb))

    return states

# def number_of_pairs(state): # output number of pairs
#     pairs = 0
#     state_up = state[1]
#     state_down = state[-1]
#     for i in range(0,len(state_up)):
#         if state_up[i] == state_down[i] == 1:
#             pairs += 1
#     return pairs


def number_of_pairs(state):
    """Key for sorting permutations.

    Returns:
        
    pairs -- number of pairs in ref state config"""

    pairs = 0
    for i in range(0,len(state),2):
        if state[i] == 1 and state[i+1] == 1:
            pairs += 1
    return pairs

def density_1b(n_holes, n_particles, weights=None):
    
    basis = gen_basis(n_holes, n_particles)
    num_sp = n_holes+n_particles
    num_states = basis.shape[0]
    
    print("Basis: ",basis)

    if weights is None:
        weights = np.ones((num_states,1))
        weights = weights/np.sum(weights)

    try:
        test_iterator = iter(weights)
    except TypeError:
        print(weights, 'is not iterable')

    rho_in = np.zeros((num_sp, num_sp), dtype = np.float64)
    rho = _wrapper_density_1b(np.copy(rho_in), basis, weights, num_sp, num_states)

    return rho


def _wrapper_density_1b(rho, basis, weights, num_sp, num_states):
    for p in range(num_sp):
        for q in range(num_sp):

            for i in range(num_states):
                state_ket = basis[i,:]
                coeff_ket = weights[i]

                state_ket_action = creator(annihilator(state_ket,q),p)


                for j in range(num_states):
                    state_bra = basis[j, :]
                    coeff_bra = weights[j]

                    #print("Le Result: ", coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action))

                    rho[p,q] += (coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action))
                    
                    # if coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action) != 0.0:
                    #     print(p,q,i,j,coeff_bra*coeff_ket, state_bra, state_ket)
                    
    return rho

def density_2b(n_holes, n_particles, weights=None):
    basis = gen_basis(n_holes, n_particles)
    num_sp = n_holes + n_particles
    num_states = basis.shape[0]

    if weights is None:
        weights = np.ones((num_states, 1))
        weights = weights / np.sum(weights)

    # Ensure weights are iterable
    if not isinstance(weights, (list, np.ndarray)):
        raise ValueError(f"{weights} is not iterable")

    rho_in = np.zeros((num_sp, num_sp, num_sp, num_sp), dtype=np.float64)
    rho = _wrapper_density_2b_slow(np.copy(rho_in), basis, weights.flatten(), num_sp, num_states)
    return rho

@njit
def _wrapper_density_2b_slow(rho, basis, weights, num_sp, num_states):
    for p in range(num_sp):
        for q in range(num_sp):
            for r in range(num_sp):
                for s in range(num_sp):
                    for i in range(num_states):
                        state_ket = basis[i, :]
                        coeff_ket = weights[i]

                        state_ket_action = creator(creator(annihilator(annihilator(state_ket, r), s), q), p)
                            
                        for j in range(num_states):
                            state_bra = basis[j, :]
                            coeff_bra = weights[j]

                            result = coeff_bra * coeff_ket

                            rho[p, q, r, s] += result * inner_product(state_bra, state_ket_action)
                            # Uncomment these lines if needed
                            # rho[q, p, r, s] = -rho[p, q, r, s]
                            # rho[p, q, s, r] = -rho[p, q, r, s]
                            # rho[q, p, s, r] = rho[p, q, r, s]

    return rho


@njit(debug=debug)#(cache=True)
def _wrapper_density_2b(rho, basis, weights, num_sp, num_states):
    
    for q in range(num_sp):
        for p in range(q):
            for s in range(num_sp):
                for r in range(s):

                    for i in range(num_states):
                        state_ket = basis[i,:]
                        coeff_ket = weights[i]

                        state_ket_action = creator(creator(annihilator(annihilator(state_ket,r),s),q),p)
                            
                        for j in range(num_states):
                            state_bra = basis[j, :]
                            coeff_bra = weights[j]

                            rho[p,q,r,s] += coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action)
                            rho[q,p,r,s] = -rho[p,q,r,s]
                            rho[p,q,s,r] = -rho[p,q,r,s]
                            rho[q,p,s,r] = rho[p,q,r,s]

    
    return rho

def density_3b(n_holes, n_particles, weights=None):
    
    basis = gen_basis(n_holes, n_particles)
    num_sp = n_holes+n_particles
    num_states = basis.shape[0]
    
    if weights is None:
        weights = np.ones((num_states,1))
        weights = weights/np.sum(weights)

    try:
        test_iterator = iter(weights)
    except TypeError:
        print(weights, 'is not iterable')

    rho_in = np.zeros((num_sp, num_sp, num_sp, num_sp, num_sp, num_sp))
    rho = _wrapper_density_3b_fast(np.copy(rho_in), basis, weights, num_sp, num_states)
    # ti = time.time()
    # rho1 = _wrapper_density_3b_fast(np.copy(rho_in), basis, weights, num_sp, num_states)
    # tf = time.time()
    # print('{: .5f} seconds'.format(tf-ti))

    # ti =time.time()
    # rho = _wrapper_density_3b(np.copy(rho_in), basis, weights, num_sp, num_states)
    # tf = time.time()
    # print('{: .5f} seconds'.format(tf-ti))
    
    # print(np.array_equal(rho1, rho))

    return rho

@njit(debug=debug)#(cache=True)
def _wrapper_density_3b(rho, basis, weights, num_sp, num_states):
    # can speed up by manually anti-symmetrizing
    
    for p in range(num_sp):
        for q in range(num_sp):
            for r in range(num_sp):
                for s in range(num_sp):
                    for t in range(num_sp):
                        for u in range(num_sp):
                            for i in range(num_states):
                                state_ket = basis[i,:]
                                coeff_ket = weights[i]

                                state_ket_action = creator(creator(creator(annihilator(annihilator(annihilator(state_ket,s),t),u),r),q),p)
                            
                                for j in range(num_states):
                                    state_bra = basis[j, :]
                                    coeff_bra = weights[j]

                                    rho[p,q,r,s,t,u] += coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action)

    return rho

@njit(debug=debug)#(cache=True)
def _wrapper_density_3b_fast(rho, basis, weights, num_sp, num_states):
    # can speed up by manually anti-symmetrizing
    
    for r in range(num_sp):
        for q in range(r):
            for p in range(q):
                for u in range(num_sp):
                    for t in range(u):
                        for s in range(t):
                            for i in range(num_states):
                                state_ket = basis[i,:]
                                coeff_ket = weights[i]

                                state_ket_action = creator(creator(creator(annihilator(annihilator(annihilator(state_ket,s),t),u),r),q),p)
                            
                                for j in range(num_states):
                                    state_bra = basis[j, :]
                                    coeff_bra = weights[j]

                                    rho[p,q,r,s,t,u] += coeff_bra*coeff_ket*inner_product(state_bra,state_ket_action)
                                    rho[q,p,r,s,t,u] = -rho[p,q,r,s,t,u]
                                    rho[r,q,p,s,t,u] = -rho[p,q,r,s,t,u]
                                    rho[p,r,q,s,t,u] = -rho[p,q,r,s,t,u]
                                    rho[q,r,p,s,t,u] = rho[p,q,r,s,t,u]
                                    rho[r,p,q,s,t,u] = rho[p,q,r,s,t,u]

                                    rho[p,q,r,t,s,u] = -rho[p,q,r,s,t,u]
                                    rho[p,q,r,u,t,s] = -rho[p,q,r,s,t,u]
                                    rho[p,q,r,s,u,t] = -rho[p,q,r,s,t,u]
                                    rho[p,q,r,t,u,s] = rho[p,q,r,s,t,u]
                                    rho[p,q,r,u,s,t] = rho[p,q,r,s,t,u]

                                    rho[q,p,r,t,s,u] = rho[p,q,r,s,t,u]
                                    rho[q,p,r,u,t,s] = rho[p,q,r,s,t,u]
                                    rho[q,p,r,s,u,t] = rho[p,q,r,s,t,u]
                                    rho[q,p,r,t,u,s] = -rho[p,q,r,s,t,u]
                                    rho[q,p,r,u,s,t] = -rho[p,q,r,s,t,u]

                                    rho[r,q,p,t,s,u] = rho[p,q,r,s,t,u]
                                    rho[r,q,p,u,t,s] = rho[p,q,r,s,t,u]
                                    rho[r,q,p,s,u,t] = rho[p,q,r,s,t,u]
                                    rho[r,q,p,t,u,s] = -rho[p,q,r,s,t,u]
                                    rho[r,q,p,u,s,t] = -rho[p,q,r,s,t,u]

                                    rho[p,r,q,t,s,u] = rho[p,q,r,s,t,u]
                                    rho[p,r,q,u,t,s] = rho[p,q,r,s,t,u]
                                    rho[p,r,q,s,u,t] = rho[p,q,r,s,t,u]
                                    rho[p,r,q,t,u,s] = -rho[p,q,r,s,t,u]
                                    rho[p,r,q,u,s,t] = -rho[p,q,r,s,t,u]

                                    rho[q,r,p,t,s,u] = -rho[p,q,r,s,t,u]
                                    rho[q,r,p,u,t,s] = -rho[p,q,r,s,t,u]
                                    rho[q,r,p,s,u,t] = -rho[p,q,r,s,t,u]
                                    rho[q,r,p,t,u,s] = rho[p,q,r,s,t,u]
                                    rho[q,r,p,u,s,t] = rho[p,q,r,s,t,u]

                                    rho[r,p,q,t,s,u] = -rho[p,q,r,s,t,u]
                                    rho[r,p,q,u,t,s] = -rho[p,q,r,s,t,u]
                                    rho[r,p,q,s,u,t] = -rho[p,q,r,s,t,u]
                                    rho[r,p,q,t,u,s] = rho[p,q,r,s,t,u]
                                    rho[r,p,q,u,s,t] = rho[p,q,r,s,t,u]


    return 

def main():
    particles = 4
    holes = 4
    rho2 = density_2b(holes, particles)

    print(rho2)


    rho1 = density_1b(holes,particles)
main()