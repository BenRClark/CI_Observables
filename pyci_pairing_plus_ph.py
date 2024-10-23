import numpy as np
import copy
import matplotlib.pyplot as plt
import itertools
import density_matrix 
#%matplotlib inline #only if using in jupyter

            
def init_states(n_holes=4, n_particles=4):

    up_config_base = np.append(np.ones(int(n_holes/2)),
                               np.zeros(int(n_particles/2)))
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
    dn_states = up_states

    states = [] # [[phase, up_vector, dn_vector], ...]
    for i in range(0, len(up_states)):
        for j in range(0, len(dn_states)):
            states.append([1, copy.deepcopy(up_states[i]), copy.deepcopy(dn_states[j])])

    #sort by number of pairs
    states.sort(key=number_of_pairs, reverse=True)

    return states

def full_states(n_holes = 4, n_particles = 4):
    up_config_base = np.append(np.ones(int(n_holes)),
                               np.zeros(int(n_particles)))
    up_config_str = ''.join([str(int(state)) for state in up_config_base])

    up_states = list(map("".join, itertools.permutations(up_config_str)))
    up_states = list(dict.fromkeys(up_states)) # remove duplicates
    up_states = [list(map(int, list(ref))) for ref in up_states]

    dn_states = copy.deepcopy(up_states)
    new_up_states = copy.deepcopy(up_states)

    for i in range(0, len(up_states)):
        new_up_states[i] = up_states[i][:4]
        dn_states[i] = up_states[i][4:]


    states = [] # [[phase, up_vector, dn_vector], ...]
    for i in range(0, len(new_up_states)):
        
          states.append([1, copy.deepcopy(new_up_states[i]), copy.deepcopy(dn_states[i])])

    #sort by number of pairs
    states.sort(key=number_of_pairs, reverse=True)

    return states

def full_states(n_holes = 4, n_particles = 4):
    up_config_base = np.append(np.ones(int(n_holes)),
                               np.zeros(int(n_particles)))
    up_config_str = ''.join([str(int(state)) for state in up_config_base])

    up_states = list(map("".join, itertools.permutations(up_config_str)))
    up_states = list(dict.fromkeys(up_states)) # remove duplicates
    up_states = [list(map(int, list(ref))) for ref in up_states]


    dn_states = up_states

    states = [] # [[phase, up_vector, dn_vector], ...]
    for i in range(0, len(up_states)):
        for j in range(0, len(dn_states)):
            states.append([1, copy.deepcopy(up_states[i]), copy.deepcopy(dn_states[j])])

    #sort by number of pairs
    states.sort(key=number_of_pairs, reverse=True)

    return states
        
        
def number_of_pairs(state): # output number of pairs
    pairs = 0
    state_up = state[1]
    state_down = state[-1]
    for i in range(0,len(state_up)):
        if state_up[i] == state_down[i] == 1:
            pairs += 1
    return pairs

def inner_product(state_1, state_2):
    if (state_1[1] == state_2[1]) and (state_1[2] == state_2[2]):
        IP = 1;
    else:
        IP = 0;
    return state_1[0]*state_2[0]*IP # phases * inner product
    

def creator(spin, i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    n = 0 # number of occupied states left of i
    for bit in vec[spin][0:i]:
        if bit == 1: n += 1
            
    if vec[spin][i] == 0: # create
        vec[spin][i] = 1
        vec[0] *= (-1)**n # phase
        return vec
    else:
        vec[0] = 0
        return vec
    
def annihilator(spin, i, state): # spin: 1=up, -1=down | i=index to operate on
    vec = copy.deepcopy(state)
    n = 0 # number of occupied states left of i
    for bit in vec[spin][0:i]:
        if bit == 1: n += 1
            
    if vec[spin][i] == 1: # annihilate
        vec[spin][i] = 0
        vec[0] *= (-1)**n # phase
        return vec
    else:
        vec[0] = 0
        return vec
    
def raiser(i, state):
    temp = annihilator(-1, i, state)
    temp = creator(1, i, temp)
    return temp

def lower(i, state):
    temp = annihilator(1, i, state)
    temp = creator(-1, i, temp)
    return temp

def kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def first_term(i,j,state):
    state = creator(1,i,creator(-1,j,annihilator(-1,i,annihilator(1,j,state))))
    return state
    
def second_term(i,j,state):
    state = creator(-1,i,creator(1,j,annihilator(1,i,annihilator(-1,j,state))))
    return state
    
def make_spin_op(states, particles, holes):
    
    Spin = np.zeros((len(states), len(states)))
    #making S2B
    #print("NEW ", new)
    top = int((particles + holes)/2) 
    row = 0 
    for phi in states:
        column = 0 
        for phip in states:

            for i in range(0,top):
                for s in [-1,1]:
                    #one body term
                    Spin[row, column] += inner_product(phi,phip)*(3/4)*(phip[s][i])
                for j in range(0,top):
                    if i != j:
                        term = -0.5*inner_product(phi, first_term(i,j,phip))# + 0.5*kronecker_delta(i,j)*inner_product(phi, creator(1,i,annihilator(1,j,phip)))
                        #if term > 0:  
                        #    print(phi, "  ", phip, "  ", i, "  ", j, "  ", (raiser(i,lower(j, phip))))
                        #term = inner_product(phi, phip)
                        Spin[row, column] += term
                        term = -0.5*inner_product(phi, second_term(i,j,phip))# + 0.5*kronecker_delta(i,j)*inner_product(phi, creator(-1,i,annihilator(-1,j,phip)))
                        Spin[row, column] += term 
                        for s in [-1,1]:
                            for s2 in [-1,1]:
                                #if s != s2:
                                term = inner_product(phi, creator(s,i,creator(s2,j,annihilator(s,i,annihilator(s2,j,phip))))) #+ kronecker_delta(i,j)*kronecker_delta(s,s2)* inner_product(phi,creator(s,i,annihilator(s2,j,phip)))
                                Spin[row, column] += -(1/8)*term
            column += 1 
        row += 1
        
    np.set_printoptions(threshold=np.inf)
    return Spin

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
    

def delta_term(phi, phip, p, d):
    result = 0
    temp = annihilator(1, p, phip)
    temp = creator(1, p, temp)
    result += (p)*d*inner_product(phi,temp)
    
    temp = annihilator(-1, p, phip)
    temp = creator(-1, p, temp)
    result += (p)*d*inner_product(phi,temp)
    #print("----D-result: ", result)
    return result


def g_term(phi, phip, p, q, g):
    result = 0
    temp = annihilator(1, q, phip)
    temp = annihilator(-1, q, temp)
    temp = creator(-1, p, temp)
    temp = creator(1, p, temp)
    result += (-g/2)*inner_product(phi,temp)
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

    for phi in states:
        col = -1 # reset col number to be increased to 0
        row += 1
        print("row: %i"%(row))
        for phip in states:
            col += 1
            for p in range(0,length):
                H[row, col] += delta_term(phi, phip, p, d)
                for q in range(0,length):
                    H[row,col] += g_term(phi, phip, p, q, g)
                    #for pp in range(0,length):
                        #print("[{},{}] p:{}, q:{}, pp:{}".format(row,col,p,q,pp))
                        #print(phi, phip)
                    #    H[row,col] += f_term(phi, phip, p, q, pp, f)
                        
    return H

def get_gs():
    states = init_states(4,4)
    hme = matrix(states, 1.0, 0.5, 0)
    _, eigstates= np.linalg.eigh(hme)
    ground_state = eigstates[:,0]

    return ground_state


                        
holes = 4
particles = 4
sum = int(holes + particles)

states = init_states(holes,particles)

spin_op = make_spin_op(states, particles, holes)
#print(spin_op)
hme = matrix(states, 1.0, 0.5, 0) # delta, g, f

_, eigstates= np.linalg.eigh(hme)

np.set_printoptions(threshold=np.inf)

#print("States ", eigstates)

spinny_boi = np.array([[1/2, 0, 0, 0, 0, 0, 0, 0],
            [0, -1/2, 0, 0, 0, 0, 0, 0],
            [0, 0, 1/2, 0, 0, 0, 0, 0],
            [0, 0, 0, -1/2, 0, 0, 0, 0],
            [0, 0, 0, 0, 1/2, 0, 0, 0],
            [0, 0, 0, 0, 0, -1/2, 0, 0],
            [0, 0, 0, 0, 0, 0, 1/2, 0],
            [0, 0, 0, 0, 0, 0, 0, -1/2]])


#spin_vec = np.diag(spinny_boi)


#find the two-body part of the total spin squared operator

#two_body_spin_small = np.array([[0.0, -0.5], 
#                                [-0.5, 0.0]])

#two_body_spin_large = np.kron(np.eye(4), two_body_spin_small)

#two_body_spin = 0.5*spin2B(np.abs(spinny_boi)*(np.sqrt(3)/2))

#print("Two body spin ", two_body_spin)

#total_two_body_spin = spinny_boi**2 + two_body_spin

ground_state = eigstates[:,0]

print("Shape of Ground state ", np.shape(ground_state))

rho = np.outer(ground_state,ground_state.conj().T)

print("Shape of rho ", np.shape(rho))

#exp_val = np.einsum("ij,ji -> ", spinny_boi, rho)

#print(exp_val)

print(np.linalg.eigvalsh(hme))

print("Shape of final matrix ", np.shape(hme))

np.set_printoptions(threshold=np.inf)

print(hme)

density1b = density_matrix.density_1b(particles,holes)
density2b = density_matrix.density_2b(particles,holes)


hole_list = []
particle_list = [] 
for i in range(sum):
    if i < holes:
        hole_list.append(i)
    else:
        particle_list.append(i)


bas1B = range(6)
bas2B = construct_basis_2B(hole_list, particle_list)
idx2B = construct_index_2B(bas2B)

useBen = False

if useBen == True:
    density_op = 1
    i = 0 
    j = 0 
    for phi in states:
        for phip in states: 
            for r in range(4):
                for rspin in [-1,1]:
                    for s in range(4):
                        for sspin in [-1,1]:
                            for p in range(4):
                                for pspin in [-1,1]:
                                    for q in range(4):
                                        for qspin in [-1,1]:
                                            new_state = creator(creator(annihilator(annihilator(phip, r, rspin), s, sspin), q, qspin),p, pspin)

                    
                            

else:
    top = int(particles+holes)
    density2B2d = np.zeros((top*top,top*top))

    for a in range(top):
        for b in range(top):
            for c in range(top):
                for d in range(top):
                    density2B2d[idx2B[(a,d)], idx2B[(c,b)]] = density2b[a,c,b,d]

print("1B denisty matrix ", density1b)

psi = states[0]

print("first state ", (annihilator(1,0,annihilator(1,1, psi))  ))
print("second state ", (annihilator(1,1,annihilator(1,0, psi))  ))

print("Ground state: ", ground_state)
print("Total spin expectation value ", ground_state.T@spin_op@ground_state)

print("Shape of desnity matrix 1 body ", np.shape(density1b))

print("Shape of desnity matrix 2 body ", np.shape(density2b))



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