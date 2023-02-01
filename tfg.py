import matplotlib.pyplot as plt
import qsimov as qs
import numpy as np

def notQCorrelated(nParties, nQubits):
    size = (nParties+1)*nQubits
    gate = qs.QGate(size,0,"not Q-Correlated")
    for i in range(nQubits,size): #aplicar hadamard a todos los grupos menos al primero, particulas tipo 0 y mitad tipo 1
        gate.add_operation("H", targets=i)
    for i in range(nQubits): #entrelazar los qubits del segundo grupo con los del primero, particulas tipo 1
        gate.add_operation("X", targets=i, controls=i+nQubits)
    return gate    

def qCorrelated(nParties, nQubits):
    size = (nParties+1)*nQubits
    gate = qs.QGate(size,0,"Q-Correlated")
    for i in range(nQubits): # aplicamos hadamard a los primeros n qubits
        gate.add_operation("H", targets=i)
    for i in range(1,nParties+1): # para todos los gemerales menos el primer grupo del comandante
        i_bin = f"{i:0{nQubits}b}" # transformamos el numero en binario
        for j in range(nQubits): # iteramos por cada digito del numero en binario
            if i_bin[j] == '1': # si es 1 aplicamos not, i nos dice sobre que general y j nos dice sobre que qubit de ese general aplicamos la puerta
                gate.add_operation("X", targets=i*nQubits+j)
    for i in range(nQubits, size): # entrelazamos los qubits del primer grupo del comandante con los qubits del resto de grupos
        gate.add_operation("X", targets=i, controls=i%nQubits)
    return gate

def generacionCircuitos(nParties, nQubits):
    size = (nParties+1)*nQubits # qubits totales del circuito

    qCircuit = qs.QCircuit(size,size,"Q-Correlated Circuit") # generamos circuito de Q-correlacionados
    qCircuit.add_operation(qCorrelated(nParties,nQubits))

    notQCircuit = qs.QCircuit(size,size,"Not Q-Correlated Circuit") # generamos circuito de no Q-correlacionados
    notQCircuit.add_operation(notQCorrelated(nParties,nQubits))

    for i in range(size): # medimos todos los qubits de ambos circuitos
        qCircuit.add_operation("MEASURE", targets=i, outputs=i)
        notQCircuit.add_operation("MEASURE", targets=i, outputs=i)

    return (notQCircuit, qCircuit)   
    
def generacionListas(nParties, size):
    qCorrelados = np.random.randint(2, size=size) # list with q-correlated positions randomly generated
    
    nQubits = int(np.ceil(np.log2(nParties+1))) # qubits necesarios para codificar el valor de w
    w = 2**nQubits # nº of possible orders
    
    circuits = generacionCircuitos(nParties, nQubits)

    exec = qs.Drewom() # creamos el ejecutor para las simulaciones
    S = [[] for i in range(nParties+1)] # creamos el estado general del sistema distribuido (en una implementación real esto no es conocido por nadie)

    for isQCorr in qCorrelados:
        res = exec.execute(circuits[isQCorr])[0] # ejecutamos el circuito para no q-correlacionados si isQcorr == 0 o el circuito para qcorrelacionados si esQcorr == 1
        Li = [int("".join(str(int(x)) for x in res[i*nQubits:i*nQubits+nQubits]), 2) for i in range(nParties+1)] # traducimos los resultados de las medidas (agrupadas por el general que las realiza) de binario a decimal
        for i in range(nParties+1): # añadimos los resultados traducidos al estado general
            S[i].append(Li[i])
        
    return S, w

# generacionListas(7, 10)


def consistent(v,L):
    if v in L:
        return False # if the order is in the list, then is not consistent
    else:
        return True # if the order is not in the list, then is consistent

def send(pvl, Qlist, V, i, _, _2):
    print('------------')
    print('Commander communicates with party nº:',i)
    print('Sending:',pvl)
    print('------------')

    L = pvl[2].copy()
    aux = set()
    for p in pvl[0]:
        aux.add(Qlist[p])
    L.update(aux) # adds to L the values at position p in the Qlist
    print('L',i,'=',L)

    if consistent(pvl[1],L): # if (v, L) is consistent then
        V.append(pvl[1]) # adds to V the order v
    print('V',i,'=',V)

    return [pvl[0], pvl[1], L]# return (P, (v, L)) where P and v will be the same but L is updated

def sendm(pvl, Qlist, V, i, isQCorr, S):
    v = i
    P = {x for x in isQCorr if S[0][x] == v or S[1][x] == v} # set of correlated positions in commander list in which v appears

    print('------------')
    print('Bad Commander communicates with party nº:',i)
    print('Sending:',P,v,pvl[2])
    print('------------')

    L = pvl[2].copy()
    aux = set()
    for p in P:
        aux.add(Qlist[p])
    L.update(aux) # adds to L the values at position p in the Qlist
    print('L',i,'=',L)

    if consistent(v,L): # if (v, L) is consistent then
        V.append(v) # adds to V the order v
    print('V',i,'=',V)

    return [P, v, L] # return (P, (v, L)) where P and v will be the same but L is updated

def send2(pvl, Qlist, V, i, j, rounds):
    print('------------')
    print('Round:',rounds,'Party nº:',i,'communicates with party nº:',j)
    print('Sending:',pvl)
    print('------------')

    L = pvl[2].copy()
    aux = set()
    for p in pvl[0]:
        aux.add(Qlist[p])
    L.update(aux) # adds to L the values at position p in the Qlist
    print('L',j,'=',L)

    if consistent(pvl[1],L) and pvl[1] not in V and len(L) == rounds+1: # if (v, L) is consistent then
        print('entro')
        V.append(pvl[1]) # adds to V the order v
    print('V',j,'=',V)

    return [pvl[0], pvl[1], L] # return (P, (v, L)) where P and v will be the same but L is updated

def send2m(pvl, Qlist, V, i, j, rounds):
    pvl[1] = 3

    print('------------')
    print('Round:',rounds,'Bad Party nº:',i,'communicates with party nº:',j)
    print('Sending:',pvl)
    print('------------')

    L = pvl[2].copy()
    aux = set()
    for p in pvl[0]:
        aux.add(Qlist[p])
    L.update(aux) # adds to L the values at position p in the Qlist
    print('L',j,'=',L)

    if consistent(pvl[1],L) and pvl[1] not in V and len(L) == rounds+1: # if (v, L) is consistent then
        V.append(pvl[1]) # adds to V the order v
    print('V',j,'=',V)

    return [pvl[0], pvl[1], L] # return (P, (v, L)) where P and v will be the same but L is updated

def QBA(nParties, size, m):

    dishonest = np.random.choice(np.arange(1, nParties+1), m, replace=False)
    print(dishonest)
    #dishonest = [1]

    # Step 1
    # Generation, preparation and distribution of particles
    S, w = generacionListas(nParties, size) # S is a list containing the lists of each general and w the nº of possible orders
    print('S =',S)
    print('w =',w)
    #...

    # Step 2
    # Set up for the commander to send the first message to all the parties
    v = np.random.randint(w) # order transmitted by the commander, random value
    print('v =',v)

    isQCorrList = {i for i in range(size) if S[0][i] != S[1][i]}
    P = {i for i in isQCorrList if S[0][i] == v or S[1][i] == v} # set of correlated positions in commander list in which v appears
    print('P =',P)# podemos utilizar un subconjunto de P

    # Step 3
    pvls = [[P,v,set()] for i in range(nParties+1)]
    temp = [[P,v,set()] for i in range(nParties+1)] # pvls temporales para sustituir threads
    print('pvl =',pvls[1])

    V = [[] for i in range(nParties+1)] # orders followed

    # Step 3(a)
    sendf = send

    if 1 in dishonest: # calls other function in case that the commander is dishonest
        sendf = sendm

    for i in range(2,nParties+1): # all parties except the commander
        pvls[i] = sendf(pvls[1], S[i], V[i], i, isQCorrList, S)

    # Step 3(b)
    sendfs = [None] + [send2 if i not in dishonest else send2m for i in range(1, nParties+1)] # this is only in case that dishonest parties exist

    rounds = 1
    while rounds <= m+1:
        for i in range(2,nParties+1): # all parties except the commander
            for j in range(2,nParties+1): # all parties except the comander
                if i != j: # except itself
                    temp[j] = sendfs[i](pvls[i], S[j], V[j], i, j, rounds)
        rounds += 1
        for i in range(2,nParties+1): # all parties except the commander
            pvls[i] = temp[i]

    # Step 3(c)
    print('------------')
    print('Conciliation')
    print('------------')

    print(V)

    check = 0

    for i in range(2,nParties+1): # only generals orders, not commander
        V[i].sort()
        for j in range(2,nParties+1):
            V[j].sort()
            if i!=j and V[i] == V[j]:
                check+=1
 
    if check == (nParties-1)*(nParties-2): # this is if all the combinations are correct
        print('An agreement has been reach:',V[2])

QBA(4, 100, 1)