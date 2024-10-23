import numpy as np
import copy
from numba import jit
import matplotlib.pyplot as plt
import itertools
import density_matrix 
from joblib import Parallel, delayed
#%matplotlib inline #only if using in jupyter

# Will need to rewrite the code but instead of using spin as a DOF just make it 
# Another p/h state

            
def init_states(n_holes=4, n_particles=4):

    up_config_base = np.append(np.ones(int(n_holes)),
                               np.zeros(int(n_particles)))
    up_config_str = ''.join([str(int(state)) for state in up_config_base])

    up_states = list(map("".join, itertools.permutations(up_config_str)))
    up_states = list(dict.fromkeys(up_states)) # remove duplicates
    up_states = [list(map(int, list(ref))) for ref in up_states]


    # up_states = [[1,1,0,0],
    #              [1,0,1,0],
    #              [1,0,0,1],
    #              [0,1,1,0],
    #              [0,1,0,1],
    #              [0,0,1,1]]

    states = [] # [[phase, up_vector, dn_vector], ...]
    for i in range(0, len(up_states)):
        states.append([1, copy.deepcopy(up_states[i])])

    #sort by number of pairs
    states.sort(key=number_of_pairs, reverse=True)

    return states

def one_body_basis(n_holes=4, n_particles=4):

    up_config_base = np.append(np.ones(int(1)),
                               np.zeros(int(7)))
    up_config_str = ''.join([str(int(state)) for state in up_config_base])

    up_states = list(map("".join, itertools.permutations(up_config_str)))
    up_states = list(dict.fromkeys(up_states)) # remove duplicates
    up_states = [list(map(int, list(ref))) for ref in up_states]


    # up_states = [[1,1,0,0],
    #              [1,0,1,0],
    #              [1,0,0,1],
    #              [0,1,1,0],
    #              [0,1,0,1],
    #              [0,0,1,1]]
    states = [] # [[phase, up_vector, dn_vector], ...]
    for i in range(0, len(up_states)):
        states.append([1, copy.deepcopy(up_states[i])])

    #sort by number of pairs
    states.sort(key=number_of_pairs, reverse=True)

    return states
        

def two_body_basis(vacuum):
    basis = []
    for i in range(4):
      for j in range(4):
        temp = creator(i, vacuum)
        temp = creator(j, temp)
        """if temp[0] == -1:
           temp[0] = 1"""
        basis.append(copy.deepcopy(temp))

    for i in range(4):
      for j in range(4, 8):
        temp = creator(i, vacuum)
        temp = creator(j, temp)
        """if temp[0] == -1:
           temp[0] = 1"""
        basis.append(copy.deepcopy(temp))

    for i in range(4,8):
      for j in range(4):
        temp = creator(i, vacuum)
        temp = creator(j, temp)
        """if temp[0] == -1:
           temp[0] = 1"""
        basis.append(copy.deepcopy(temp))

    for i in range(4, 8):
      for j in range(4, 8):
        temp = creator(i, vacuum)
        temp = creator(j, temp)
        """if temp[0] == -1:
           temp[0] = 1"""
        basis.append(copy.deepcopy(temp))

    #basis.sort(key=number_of_pairs, reverse=True)
    return basis

        
def number_of_pairs(state): # output number of pairs
    pairs = 0
    state_up = state[1]
    state_down = state[-1]
    for i in range(0,len(state_up)):
        if state_up[i] == state_down[i] == 1:
            pairs += 1
    return pairs

def inner_product(state_1, state_2):
    if (state_1[1] == state_2[1]):
        IP = 1;
    else:
        IP = 0;
    return state_1[0]*state_2[0]*IP # phases * inner product
    

def creator(i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    if i > 7:
        vec[0] = 0
        return vec
    else:
        n = 0 # number of occupied states left of i
        for bit in vec[1][0:i]:
            if bit == 1: n += 1
                
        if vec[1][i] == 0: # create
            vec[1][i] = 1
            vec[0] *= (-1)**n # phase
            return vec
        else:
            vec[0] = 0
            return vec

 
def annihilator(i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    if i > 7:
        vec[0] = 0
        return vec
    else:
        n = 0 # number of occupied states left of i
        for bit in vec[1][0:i]:
            if bit == 1: n += 1
                
        if vec[1][i] == 1: # annihilate
            vec[1][i] = 0
            vec[0] *= (-1)**n # phase
            return vec
        else:
            vec[0] = 0
            return vec
    
def kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def first_term(i,j,state):
    if i % 2 == 1 and j % 2 == 0:
        state = creator(i-1,creator(j+1,annihilator(i,annihilator(j,state))))
        return state
    else:
        vec = copy.deepcopy(state)
        vec[0] = 0
        return vec
    
def second_term(i,j,state):
    if j % 2 == 1 and i % 2 == 0:
        state = creator(i+1,creator(j-1,annihilator(i,annihilator(j,state))))
        return state
    else:
        vec = copy.deepcopy(state)
        vec[0] = 0
        return vec
    

def raiser(i, state):
        
    #if a state is even it is already spin up 
        if i % 2 == 1:
            #print("THE GOOD ONE")
            state = annihilator(i, state)
            state = creator(i-1, state)
            return state
        else: 
            vec = copy.deepcopy(state)
            vec[0] = 0
            return vec
    
def lower(i, state):
    #if a state is odd it is already spin down 
        if i % 2 == 0 and i <8:
            state = annihilator(i, state)
            state = creator(i+1, state)
            return state
        else: 
            vec = copy.deepcopy(state)
            vec[0] = 0
            return vec
        
def sz(i, state):
    return state
    
        
def spin2B(S1B):
  holes     = [0,1,2,3]
  particles = [4,5,6,7]
  bas1B = range(8)
  bas2B = construct_basis_2B(holes, particles)
  idx2B = construct_index_2B(bas2B)

  spin2B = np.zeros([64,64])

  for (i, j) in bas2B:
      for (k, l) in bas2B:
        if (i, j) != (k,l):
            spin2B[idx2B[(i,j)],idx2B[(k,l)]] = S1B[i,j]*S1B[k,l]

  return spin2B

def construct_basis_2B(holes, particles):
  basis = []
  for i in holes:
    for j in holes:
      basis.append((i, j))

  for i in holes:
    for a in particles:
      basis.append((i, a))

  for a in particles:
    for i in holes:
      basis.append((a, i))

  for a in particles:
    for b in particles:
      basis.append((a, b))

  return basis

def construct_index_2B(bas2B):
  index = { }
  for i, state in enumerate(bas2B):
    index[state] = i

  return index
    

def delta_term(phi, phip, p, q,d):
    result = 0
    temp = annihilator(p, phip)
    temp = creator(q, temp)
    result += np.floor_divide(p, 2)*d*inner_product(phi,temp)
    #print("----D-result: ", result)
    return result


def g_term(phi, phip, p, q, r, s, g):
    result = 0
    temp = annihilator(p, phip)
    temp = annihilator(q, temp)
    temp = creator(r, temp)
    temp = creator(s, temp)
    if p % 2 == 0 and r%2 == 0 and q == p+1 and s == r+1:
        result += (g/2)*inner_product(phi,temp)
    #if result != 0:
    #    print(result, " ", phi, " ", phip, " ", p, " ", q, " ", r, " ", s)
    #print("----G-result: ", result)
    return result

def f_term(phi, phip, p, q, pp, f):
    if p == pp: 
        return 0
    else:
        result = 0
        temp = annihilator(1, q, phip)
        temp = annihilator(-1, q, temp)
        temp = creator(-1, pp, temp)
        temp = creator(1, p, temp)
        # if temp[0]!=0: 
        #     print("p=%i, pp=%i, q=%i"%(p,pp,q))
        #     print(temp)
        #     print(phi)
        #     print(inner_product(phi,temp))
        result += (f/2)*inner_product(phi,temp)

        temp = annihilator(1, pp, phip)
        temp = annihilator(-1, p, temp)
        temp = creator(-1, q, temp)
        temp = creator(1, q, temp)
        result += (f/2)*inner_product(phi,temp)
        # print("----F-result: ", result)
        return result
    

def matrix(states, d, g, f): # states is a list of form [[phase, up, down], [] ... []]
    H = np.zeros((len(states),len(states)))
    row, col = -1, -1 # increases to 0 at first run through
    length = len(states[0][1])
    print("Length ", length)

    for phi in states:
        col = -1 # reset col number to be increased to 0
        row += 1
        print("row: %i"%(row))
        for phip in states:
            col += 1
            for p in range(0,length):
                for q in range(0,length):
                    H[row, col] += delta_term(phi, phip, p, p, d)
                    for r in range(0,length):
                        for s in range(0, length):
                            H[row,col] += 1/4*g_term(phi, phip, p, q, r, s, g)
                        
    return H

def make_2B_spins():
    S2B = np.zeros((64,64))
    two_bod = two_body_basis([1,[0,0,0,0,0,0,0,0]])
    #making S2B
    #print("NEW ", new)
    row = 0 
    for phi in two_bod:
        column = 0 
        for phip in two_bod:
            for i in range(0,8):
                for j in range(0,8):
                    if i != j:
                        #first term in notes
                        term = inner_product(phi, (raiser(i,lower(j, phip))))*0.5
                        #if term > 0: 
                            #print(phi, "  ", phip, "  ", i, "  ", j, "  ", (raiser(i,lower(j, phip))))
                        #term = inner_product(phi, phip)
                        S2B[row, column] += term
                        #second term in notes
                        term = inner_product(phi, (lower(i,raiser(j, phip))))*0.5
                        S2B[row, column] += term 
                        #third term in notes
                        term = inner_product(phi, phip) * (phi[1][i])*(-1)**i * phi[1][j]*(-1)**j * 1/8
                        S2B[row, column] += term
            column += 1 
        row += 1
        
    np.set_printoptions(threshold=np.inf)

    S1B = np.diag([3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4, 3/4])
    #S1B = np.zeros((8,8))
    return S1B, S2B

def compute_spin_element(row, column, states):
    phi = states[row]
    phip = states[column]
    spin_element = 0.0

    for i in range(0, 8):
        spin_element += inner_product(phi,phip) * (3 / 4) * (phip[1][i])**2

        for j in range(0, 8):
            if i != j:
                #if (i%2 == 1 and j%2 == 0):
                term = -inner_product(phi, first_term(i, j, phip)) * 0.5 + kronecker_delta(i, j + 1) * inner_product(phi, creator(i - 1, annihilator(j, phip))) * 0.5
                spin_element += term

                #if (j%2 == 1 and i%2 == 0):
                term = -inner_product(phi, second_term(i, j, phip)) * 0.5 + kronecker_delta(i, j - 1) * inner_product(phi, creator(i + 1, annihilator(j, phip))) * 0.5
                spin_element += term

                #term = inner_product(phi, phip) * (phi[1][i])*(-1)**i * phi[1][j]*(-1)**j * 1/8
                term = -inner_product(phi, creator(j, creator(i, annihilator(j, annihilator(i, phip))))) * ((-1) ** j) * ((-1) ** i) \
                       + kronecker_delta(i, j) * inner_product(phi, creator(i, annihilator(j, phip)))
                spin_element += term * 1 / 8

    return row, column, spin_element

def make_spin_op(states):
    print("MAKING THE SPIN")
    
    # Initialize an empty Spin matrix
    Spin = np.zeros((len(states), len(states)))

    # Parallel execution of the matrix filling process
    results = Parallel(n_jobs=-1)(
        delayed(compute_spin_element)(row, column, states)
        for row in range(len(states))
        for column in range(len(states))
    )

    # Populate the Spin matrix with the computed elements
    for row, column, spin_value in results:
        Spin[row, column] = spin_value

    np.set_printoptions(threshold=np.inf)
    return Spin

def get_operator_from_y(y, dim1B, dim2B):
  
  # reshape the solution vector into 0B, 1B, 2B pieces
  ptr = 0
  zero_body = y[ptr]

  ptr += 1
  one_body = np.reshape(y[ptr:ptr+dim1B*dim1B], (dim1B, dim1B))

  ptr += dim1B*dim1B
  two_body = np.reshape(y[ptr:ptr+dim2B*dim2B], (dim2B, dim2B))

  return zero_body,one_body,two_body

def compute_row(row, states, H1B, H2B, idx2B):
        column_results = []
        for phip in states:
            column_value = 0
            for p in range(8):
                for q in range(8):
                    term1 = H1B[p, q] * inner_product(phip, creator(p, annihilator(q, states[row])))
                    column_value += term1
                    
                    for r in range(8):
                        for s in range(8):
                            term2 = (1/4) * H2B[idx2B[(p, q)], idx2B[(r, s)]] * inner_product(phip, creator(p, creator(q, annihilator(s, annihilator(r, states[row])))))
                            column_value += term2
            
            column_results.append(column_value)
        return column_results

def main():                
    two_bod = two_body_basis([1,[0,0,0,0,0,0,0,0]])


    print("two body ", len(two_bod), " ", two_bod)

    states = init_states(4,4)
                    

    holes     = [0,1,2,3]
    particles = [4,5,6,7]
    bas1B = range(8)
    bas2B = construct_basis_2B(holes, particles)
    idx2B = construct_index_2B(bas2B)

    H2B = np.zeros((64,64))

    H1B = np.zeros((8,8))

    

    row = 0 
    for phi in two_bod:
        column = 0 
        print("row: %i"%(row))
        for phip in two_bod:
            for p in range(0,8, 2):
                q = p +1
                for r in range(0,8,2):
                    s = r + 1
                    term = g_term(phi, phip, p, q, r, s, 0.5)
                    H2B[row, column] += term 
            column += 1 
        row += 1


    print(H2B)

    #make one body basis 

    one_bod = one_body_basis()

    row = 0 
    for phi in one_bod:
        column = 0 
        for phip in one_bod:
            for p in range(0,8):
            
                term = delta_term(phi,phip,p,p,1.0)
                H1B[row, column] += term 
            column += 1 
        row += 1

    print("H1B ", H1B)

    holes     = [0,1,2,3]
    particles = [4,5,6,7]
    bas1B = range(8)
    bas2B = construct_basis_2B(holes, particles)
    idx2B = construct_index_2B(bas2B)


    CI = np.zeros((70,70))



    states = init_states(4,4)

# Assuming H1B and H2B are numpy arrays, and inner_product, creator, and annihilator are defined elsewhere.

    # Initialize CI matrix
    num_states = len(states)
    CI = np.zeros((num_states, num_states))

    # Parallel computation
    results = Parallel(n_jobs=-1)(delayed(compute_row)(row, states, H1B, H2B, idx2B) for row in range(num_states))

    # Fill CI matrix
    for row, column_values in enumerate(results):
        print("Row ", row)
        CI[row, :] = column_values

    
    """row = 0
    for phi in states:
        print("row: %i"%(row))
        column = 0
        for phip in states:
            for p in range(8):
                for q in range(8):
                    for r in range(8):
                        for s in range(8):
                            CI[row,column] += H1B[p,q]*inner_product(phip,creator(p,annihilator(q, phi))) + 1/4*H2B[idx2B[(p, q)], idx2B[(r, s)]]* inner_product(phip,creator(p,creator(q,annihilator(s,annihilator(r, phi)))))
            column += 1
        row += 1"""


    eigvals, eigvecs = np.linalg.eigh(CI)

    print("eigvals!! ", eigvals)

    

    p = [1,[1,0,0,1,0,0,0,0]]
    new = (raiser(3,lower(0, p)))
    

    print("inner prod ", inner_product(new, [1,[0,1,1,0,0,0,0,0]]))

    S1B, S2B = make_2B_spins()


    print("S1B ", S1B)

    print("S2B ", S2B)

    print("H2B ", H2B)

    all_zero = np.all(H2B == 0)

    print(all_zero)

    #assert False

    """print("The initial states: ", states)
    hme = matrix(states, 1.0, 0.5, 0) # delta, g, f

    print("HME: ", hme)

    energies, states= np.linalg.eigh(hme)"""


    print("The states: ", states)

    #print("The energies: ", energies)
    


    spinny_boi = np.array([[1/2, 0, 0, 0, 0, 0, 0, 0],
                [0, -1/2, 0, 0, 0, 0, 0, 0],
                [0, 0, 1/2, 0, 0, 0, 0, 0],
                [0, 0, 0, -1/2, 0, 0, 0, 0],
                [0, 0, 0, 0, 1/2, 0, 0, 0],
                [0, 0, 0, 0, 0, -1/2, 0, 0],
                [0, 0, 0, 0, 0, 0, 1/2, 0],
                [0, 0, 0, 0, 0, 0, 0, -1/2]])


    spin_vec = np.diag(spinny_boi)


    #find the two-body part of the total spin squared operator

    two_body_spin_small = np.array([[0.0, -0.5], 
                                    [-0.5, 0.0]])

    two_body_spin_large = np.kron(np.eye(4), two_body_spin_small)

    two_body_spin = 0.5*spin2B(np.abs(spinny_boi)*(np.sqrt(3)/2))

    print("Two body spin ", two_body_spin)

    #total_two_body_spin = spinny_boi**2 + two_body_spin

    #ground_state = states[:,0]

    #print("Shape of Ground state ", np.shape(ground_state))

    #rho = np.outer(ground_state,ground_state.conj().T)

    #print("Shape of rho ", np.shape(rho))

    #exp_val = np.einsum("ij,ji -> ", spinny_boi, rho)

    #print(exp_val)

    #print(np.linalg.eigvalsh(hme))

    #print("Shape of final matrix ", np.shape(hme))

    np.set_printoptions(threshold=np.inf)

    #print(hme)

    weights_1b = np.zeros(8*8 ,dtype = np.float64)
    weights_1b[0] += 1.0

    weights_2b = np.zeros(64*64)
    weights_2b[0] += 1.0

    density1b = density_matrix.density_1b(4,4, weights_1b)
    density2b = density_matrix.density_2b(4,4, weights_2b)

    holes     = [0,1,2,3]
    particles = [4,5,6,7]
    bas1B = range(8)
    bas2B = construct_basis_2B(holes, particles)
    idx2B = construct_index_2B(bas2B)

    #print("SHAPE OF STATES", np.shape(states))

    spin_op = make_spin_op(states)

    useBen = False

    if useBen == True:

        density2B1 = np.einsum("pr, qs -> qrps", density1b, density1b)
        density2B2 = np.einsum("ps, qr -> qrps", density1b, density1b)
        density2B = density2B1 - density2B2
        density2B2d = np.zeros((64,64))

        for a in range(8):
            for b in range(8):
                for c in range(8):
                    for d in range(8):
                        density2B2d[idx2B[(a,d)], idx2B[(c,b)]] = density2B[a,b,c,d]

    else:
        density2B2d = np.zeros((64,64))

        for a in range(8):
            for b in range(8):
                for c in range(8):
                    for d in range(8):
                        density2B2d[idx2B[(a,d)], idx2B[(c,b)]] = density2b[a,c,b,d]

    print("1B denisty matrix ", density1b)


    print("Shape of desnity matrix 1 body ", np.shape(density1b))

    print("Shape of desnity matrix 2 body ", np.shape(density2b))

    projected_spin_exp_val = np.trace(spinny_boi@density1b)

    print("Projected spin operator expectation value ", projected_spin_exp_val)

    total_spin_sqrd_exp_val = np.trace((S1B)@density1b)

    total_spin_sqrd_exp_val_2B = np.trace(S2B@density2B2d)

    print("Total spin operator expectation value ", total_spin_sqrd_exp_val)

    print("Total spin operator expectation value 2B ", total_spin_sqrd_exp_val_2B)

    print("Expectation value of spin operator ", eigvecs[:,0].T@spin_op@eigvecs[:,0] )


main()
"""

main()
    # np.set_printoptions(threshold=np.nan, linewidth=300) # show full array
    #print(hme)


    # # large map of hme
    # fig = plt.figure(figsize=(8,8))
    # # plt.imshow(hme, cmap='summer', 
    # #                 interpolation='nearest',       
    # #                 vmin=np.amin(hme), 
    # #                 vmax=np.amax(hme)) 
    # # plt.imshow(hme, cmap='RdBu_r', 
    # #                 interpolation='nearest',       
    # #                 vmin=min(np.amin(hme), -np.amax(hme)), 
    # #                 vmax=max(np.amax(hme), np.abs(np.amin(hme)))) 
    # plt.imshow(hme, cmap='RdBu_r', 
    #                 interpolation='nearest',       
    #                 vmin=np.amin(hme), 
    #                 vmax=np.amax(hme)) 
    # plt.tick_params(axis='y', which='both', labelleft=False, labelright=False)
    # plt.tick_params(axis='x', which='both', labelbottom=False, labeltop=False)

    # # # Loop over data dimensions and create text annotations.
    # # for i in range(len(hme)):
    # #     for j in range(len(np.array(hme)[0])):
    # #         if np.abs(hme[i,j]) > 1.0e-5: # only if above threshold
    # #             plt.text(j, i, round(hme[i, j], 2), ha="center", va="center", color="black")

    # plt.show()
    """

