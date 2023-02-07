import itertools
import matplotlib.pyplot as plt
import numpy as np
import qsimov as qs
import sys

from mpi4py import MPI


def mpi_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def notQCorrelated(nParties, nQubits):
    size = (nParties+1)*nQubits
    gate = qs.QGate(size,0,"not Q-Correlated")
    for i in range(nQubits,size): #apply hadamard to all the groups but the first, particles of type 0 y half of type 1
        gate.add_operation("H", targets=i)
    for i in range(nQubits): #entangle the qubits of the second group with the first one, type 1 particles
        gate.add_operation("X", targets=i, controls=i+nQubits)
    return gate    


def qCorrelated(nParties, nQubits):
    size = (nParties+1)*nQubits
    gate = qs.QGate(size,0,"Q-Correlated")
    for i in range(nQubits): # apply hadamard to the firtst n qubits
        gate.add_operation("H", targets=i)
    rands = np.arange(1, nParties+1)
    np.random.shuffle(rands)
    # mpi_print("randy", rands)
    for i in range(1,nParties+1): # for all the generals but the first group of the commander
        i_bin = f"{rands[i-1]:0{nQubits}b}" # transform the number to binary
        for j in range(nQubits): # iterate for each digite of the binary number
            if i_bin[j] == "1": # if 1 apply not, i tells us over which general and j tells us over which qubit of that general apply the gate
                gate.add_operation("X", targets=i*nQubits+j)
    for i in range(nQubits, size): # entangle the qubits of the first group of the commander with the qubits of the rest of the groups
        gate.add_operation("X", targets=i, controls=i%nQubits)
    return gate


def genQCorrCircuit(nParties, nQubits):
    size = (nParties+1)*nQubits # total qubits of the circuit
    
    qCircuit = qs.QCircuit(size,size,"Q-Correlated Circuit") # generate Q-Correlated circuit
    qCircuit.add_operation(qCorrelated(nParties,nQubits))
    
    for i in range(size): #  we measure all the qubits of the circuit
        qCircuit.add_operation("MEASURE", targets=i, outputs=i)
    
    return qCircuit
    


def genNQCorrCircuit(nParties, nQubits):
    size = (nParties+1)*nQubits # total qubits of the circuit
    
    notQCircuit = qs.QCircuit(size,size,"Not Q-Correlated Circuit") # generate not Q-Correlated circuit
    notQCircuit.add_operation(notQCorrelated(nParties,nQubits))
    
    for i in range(size): # we measure all the qubits of the circuit
        notQCircuit.add_operation("MEASURE", targets=i, outputs=i)
    
    return notQCircuit


def generacionListas(nParties, size, nQubits, w):
    qCorrelados = np.random.randint(2, size=size) # list with q-correlated positions randomly generated
    # mpi_print("Q-Correlated positions:", qCorrelados)
    
    nqCirc = genNQCorrCircuit(nParties, nQubits)
    
    circuit_gen = [(lambda: nqCirc), (lambda: genQCorrCircuit(nParties, nQubits))]

    exec = qs.Drewom() # creation of the executor for the simulations
    raw = [[] for i in range(nParties+1)]

    for isQCorr in qCorrelados:
        res = exec.execute(circuit_gen[isQCorr]())[0] # execute the circuit for not q-correlated if isQcorr == 0 or the circuit for qcorrelated if esQcorr == 1
        for i in range(nParties+1):
            raw[i] += res[i*nQubits:i*nQubits+nQubits]
        
    return np.array(raw, dtype=int)


def consistent(v, L, w):
    # Cond 1: All the lists in 𝓛 must have the same size to be consistent
    it = iter(L)  # As L is a set, we cannot index it. We must use an iterator.
    the_len = len(next(it))
    if not all(len(subL) == the_len for subL in it):
        return False
    # Cond 2: All the lists in 𝓛 are created by elements in W - {v}
    if not all(all(0 <= x <= w and x != v for x in subL) for subL in L):
        return False
    # Cond 3: For each possible pair of lists, the elements at position k are different
    pairs = itertools.combinations(L, 2)
    return all(all(Li[k] != Lj[k] for k in range(the_len)) for Li, Lj in pairs)


def dishonest_comm(comm, rank, nParties, nDishonest):
    im_dishonest = False
    if rank == 0:  # Code to execute by the QSD
        # Deciding who is dishonest. This is not done by the QSD, but it is the only thing available that can act as a "master" node.
        dishonest_ids = np.random.choice(np.arange(1, nParties+1), nDishonest, replace=False)
        
        reqs = [None for i in range(1, nParties+1)]
        for i in range(1, nParties+1):
            buffer = np.array(i in dishonest_ids, dtype=int)
            reqs[i-1] = comm.Isend([buffer, MPI.INT], dest=i)  # Send to each general a message telling if is dishonest
            reqs[i-1].Test()  # Start sending. If Test or Wait are not called, Isend does not do anything (for this implementation of MPI)
        for i in range(nParties):
            if not reqs[i].Test():
                reqs[i].Wait()  # Wait until the sending is completed
        return dishonest_ids
    else:
        buffer = np.empty(1, dtype=int)
        req = comm.Irecv([buffer, MPI.INT], source=0)
        req.Wait()
        im_dishonest = bool(buffer[0])
        dis = ""
        if im_dishonest:
            dis = "dis"
        mpi_print(f"[{rank}] I'm {dis}honest")
    return im_dishonest


def measure_to_ints(raw, sizeL, nQubits):
    return [int("".join(str(x) for x in raw[i*nQubits:i*nQubits+nQubits]), 2) for i in range(sizeL)]


def particle_comm(comm, rank, nParties, nQubits, w, sizeL):
    Li = None
    Lc = None
    if rank == 0:  # Generate particles, send them and each general measures them
        mpi_print("|W| =", w)
        # Step 1
        # Generation, preparation and distribution of particles
        rawS = generacionListas(nParties, sizeL, nQubits, w) # S is a list containing the lists of each general and w the nº of possible orders
        
        reqs = [None for i in range(nParties+1)]
        reqs[0] = comm.Isend([rawS[0], MPI.INT], dest=1)  # Send the extra one to the commander
        reqs[0].Test()
        for i in range(1, nParties+1):
            reqs[i] = comm.Isend([rawS[i], MPI.INT], dest=i)  # Send particles to each general
            reqs[i].Test()
        for i in range(nParties + 1):
            if not reqs[i].Test():
                reqs[i].Wait()  # Wait until all receive
    else:
        buffer = np.empty(nQubits * sizeL, dtype=int)
        req = comm.Irecv([buffer, MPI.INT], source=0)
        req.Test()
        if rank == 1:  # The commander receives two
            bufferC = np.empty(nQubits * sizeL, dtype=int)
            req2 = comm.Irecv([bufferC, MPI.INT], source=0)
            req2.Wait()
            Lc = measure_to_ints(bufferC, sizeL, nQubits) # translate the results of the measurements from binary to decimal
            mpi_print(f"[{rank}]: Lc =", Lc)
        req.Wait()
        Li = measure_to_ints(buffer, sizeL, nQubits) # translate the results of the measurements from binary to decimal
        mpi_print(f"[{rank}]: L{rank} =", Li)
    return Li, Lc


def comm_broadcast(comm, rank, nParties, w, v, Vi, Li, isQCorr, Lc, is_biz):
    if rank == 1:
        # Step 2
        bad = ""
        if is_biz:
            bad = "B"
            v1 = np.random.randint(w)
            v2 = np.random.randint(w)
            while v2 == v1:
                v2 = np.random.randint(w)
        for i in range(2, nParties+1):
            if is_biz:
                if i <= int((nParties+1)/2):
                    v = v1
                else:
                    v = v2
            P = {x for x in isQCorr if Lc[x] == v}
            # mpi_print(f"[{bad}{rank} -> {i}]", (P, v, set()))
            send_pvl(comm, rank, i, P, v, set(), is_biz)
    elif rank != 0:
        # Step 3 a
        P, v, L = recv_pvl(comm, rank, 1)
        # Step 3 a i
        L.add(tuple(Li[j] for j in P))  # A set of lists is not possible because they are not hashables. A set of tuples yes (they are inmutables)
        mpi_print(f"[{rank}] L = {L}")
        # Step 3 a ii
        if consistent(v, L, w): # if (v, L) is consistent then
            # Step 3 a ii A
            Vi.add(v) # adds to V the order v
            # Step 3 a ii B
            lieu_broadcast(comm, rank, nParties, P, v, L, is_biz)


def send_pvl(comm, rank, dest, P, v, L, is_biz):
    bad = ""
    if is_biz:
        bad = "B"
    mpi_print(f"[{bad}{rank} -> {dest}] Sending", (P, (v, L)))
    reqs = [None for i in range(4 + len(L) * 2)]
    buff = np.array(len(P), dtype=int)
    reqs[0] = comm.Isend([buff, MPI.INT], dest=dest, tag=1)
    reqs[0].Test()
    # mpi_print(f"[{rank} -> {dest}] |P| = {buff}")
    reqs[1] = comm.Isend([np.array(list(P), dtype=int), MPI.INT], dest=dest, tag=2)
    reqs[1].Test()
    # mpi_print(f"[{rank} -> {dest}] P = {list(P)}")
    reqs[2] = comm.Isend([np.array(v, dtype=int), MPI.INT], dest=dest, tag=3)
    reqs[2].Test()
    # mpi_print(f"[{rank} -> {dest}] v = {v}")
    reqs[3] = comm.Isend([np.array(len(L), dtype=int), MPI.INT], dest=dest, tag=4)
    reqs[3].Test()
    # mpi_print(f"[{rank} -> {dest}] |L| = {v}")
    i = 5
    for subL in L:
        reqs[i-1] = comm.Isend([np.array(len(subL), dtype=int), MPI.INT], dest=dest, tag=i)
        reqs[i-1].Test()
        # mpi_print(f"[{rank} -> {dest}] |L{i-5//2}| = {len(subL)}")
        reqs[i] = comm.Isend([np.array(subL, dtype=int), MPI.INT], dest=dest, tag=i+1)
        reqs[i].Test()
        # mpi_print(f"[{rank} -> {dest}] L{i-5//2} = {subL}")
        i += 2
    for req in reqs:
        req.Wait()
    mpi_print(f"[{bad}{rank} -> {dest}] Sent packet!")


def recv_pvl(comm, rank, src):
    buff = np.empty(1, dtype=int)
    req = comm.Irecv([buff, MPI.INT], source=src, tag=1)
    req.Wait()
    # mpi_print(f"[{rank} <- {src}] |P| = {buff[0]}")
    buff = np.empty(buff[0], dtype=int)
    req = comm.Irecv([buff, MPI.INT], source=src, tag=2)
    req.Wait()
    P = set(buff)
    # mpi_print(f"[{rank} <- {src}] P = {P}")
    buff = np.empty(1, dtype=int)
    req = comm.Irecv([buff, MPI.INT], source=src, tag=3)
    req.Wait()
    v = buff[0]
    # mpi_print(f"[{rank} <- {src}] v = {v}")
    req = comm.Irecv([buff, MPI.INT], source=src, tag=4)
    req.Wait()
    L = set()
    # mpi_print(f"[{rank} <- {src}] |L| = {buff[0]}")
    for i in range(buff[0]):
        buff = np.empty(1, dtype=int)
        req = comm.Irecv([buff, MPI.INT], source=src, tag=i*2+5)
        req.Wait()
        # mpi_print(f"[{rank} <- {src}] |L{i}| = {buff[0]}")
        buff = np.empty(buff[0], dtype=int)
        req = comm.Irecv([buff, MPI.INT], source=src, tag=i*2+6)
        req.Wait()
        # mpi_print(f"[{rank} <- {src}] L{i} = {tuple(buff)}")
        L.add(tuple(buff))
    # mpi_print(f"[{rank}] Received full packet!")
    # mpi_print(f"[{rank} <- {src}] (P, (v, L)) =", (P, (v, L)))
    return P, v, L


def lieu_broadcast(comm, rank, nParties, P, v, L, is_biz):
    for i in range(2, nParties+1):
        if i == rank:
            continue
        sent = 1
        if is_biz: # in case the general is dishonest he can do the following things
            dis_action = np.random.randint(4) # he will do a random action between 4 possible options
            if dis_action == 0:
                sent = np.random.randint(2)  # sometimes sends information sometimes does not do anything
                mpi_print(f"The action for general {rank} is: maybe not sending inf {sent}")
            elif dis_action == 1:
                v = np.random.randint(nParties+1) # a random order is sent
                mpi_print(f"The action for general {rank} is: sending order {v}")
            elif dis_action == 2:
                P.clear() # deletes all the information in P
                mpi_print(f"The action for general {rank} is: empty P {P}")
            else:
                L.clear() # deletes all the information in L
                mpi_print(f"The action for general {rank} is: empy L {L}")
        if sent:
                send_pvl(comm, rank, i, P, v, L, is_biz)


def lieu_receive(comm, rank, P, v, L, nParties, w, round, Vi, Li, is_biz, num_biz):
    # Step 3 b i
    L.add(tuple(Li[j] for j in P))  # A set of lists is not possible because they are not hashables. A set of tuples yes (they are inmutables)
    # mpi_print(f"[{rank}] L = {L}")
    # Paso 3 b ii
    if consistent(v, L, w) and v not in Vi and len(L) == round + 1: # if (v, L) is consistent then
        # Step 3 b ii A
        Vi.add(v) # adds to V the order v
        # Step 3 b ii B
        if round <= num_biz:
            lieu_broadcast(comm, rank, nParties, P, v, L, is_biz)
    # mpi_print(f"[{rank}] V{rank} = {Vi}")


def decide_order(Vi, v, is_comm):
    if is_comm:
        return v
    return min(Vi)


def QBA(sizeL, nDishonest):
    comm = MPI.COMM_WORLD;  # The communicator
    size = comm.Get_size()  # How many nodes we have, counting the particle generator QSD
    rank = comm.Get_rank()  # Which node is this
    
    nParties = size - 1  # Number of generals, counting the commander
    
    # All must know how many qubits correspond to each one and the value of w.
    nQubits = int(np.ceil(np.log2(nParties+1))) # qubits needed to codify the w value
    w = 2**nQubits # nº of possible orders
    
    im_dishonest = dishonest_comm(comm, rank, nParties, nDishonest)  # If this node is dishonest or not
    # Step 1 a
    Li, Lc = particle_comm(comm, rank, nParties, nQubits, w, sizeL)  # Obtaining the list Li (and list Lc for the commander)
    isQCorrList = None
    v = None
    if rank == 1:
        # Step 1 b
        isQCorrList = {i for i in range(sizeL) if Li[i] != Lc[i]}
        mpi_print("isQCorr = ", isQCorrList)
        v = np.random.randint(w) # order transmitted by the commander, random value
        mpi_print("v =", v)
    
    Vi = set()
    # Steps 2 and 3 a
    comm_broadcast(comm, rank, nParties, w, v, Vi, Li, isQCorrList, Lc, im_dishonest)
    comm.Barrier()  # After sending all synchronize
    # Step 3 b
    for round in range(1, nDishonest + 2):
        if rank > 1:
            status = MPI.Status()
            PvLs = []
            while comm.Iprobe(source=MPI.ANY_SOURCE, status=status):  # If there is a PvL pending of reception
                src = status.Get_source()
                P, v, L = recv_pvl(comm, rank, src)
                PvLs.append((P, (v, L)))
            # Step 3 b (if)
            for P, (v, L) in PvLs:
                lieu_receive(comm, rank, P, v, L, nParties, w, round, Vi, Li, im_dishonest, nDishonest)
        comm.Barrier()  # After send all synchronize
    if rank > 1 and not im_dishonest:
        mpi_print(f"[{rank}] V{rank} = {Vi}")
    result = np.empty(nParties, dtype=int)
    if rank == 0:
        for src in range(1, nParties+1):
            buff = np.array(1)
            comm.Recv([buff, MPI.INT], source=src)
            result[src-1] = buff.item()
    else:
        comm.Send([np.array(decide_order(Vi, v, rank==1)), MPI.INT], dest=0)
    if rank == 0:
        mpi_print("Decisions:", result)
        mpi_print("Dishonests:", im_dishonest)
        filtered = {result[i] for i in range(nParties) if i+1 not in im_dishonest}
        mpi_print("Success:", len(filtered) == 1)


if __name__ == "__main__":
    QBA(int(sys.argv[1]), int(sys.argv[2]))
