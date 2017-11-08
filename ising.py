import numpy as np
import pickle
from state import State

config = {
    'N': 4,
    'J_min': 0.0,
    'J_max': 1.0,
    'J_scale': 1.0,
    'B_min': 0.0,
    'B_max': 2.0,
    'B_scale': 1.0
}

class IsingModel(object):
    def __init__(self, config=config):
        self.config = { k: config[k] for k in config }
        N = config['N']
        J_min = config['J_min']
        J_max = config['J_max']
        J_sc = config['J_scale']
        B_min = config['B_min']
        B_max = config['B_max']
        B_sc = config['B_scale']
        # J & B matrices
        J = np.random.uniform(low=J_min, high=J_max, size=N*N)
        J = J.reshape((N,N))
        B = np.random.uniform(low=B_min, high=B_max, size=N)
        # Symmetrize
        J = J + J.T
        J = 0.5*J
        # Zero diagonals
        np.fill_diagonal(J, 0.0)
        # Scale
        J = J_sc*J
        B = B_sc*B
        self.J = J
        self.B = B
        return
    def spawn(self):
        S = IsingState(self.config)
        return S
    def energy(self, state):
        return -0.5*state.S.dot(self.J).dot(state.S) - self.B.dot(state.S)
    def energy_split(self, state):
        return -self.B.dot(state.S), -0.5*state.S.dot(self.J).dot(state.S)


class IsingEnsemble(object):
    def __init__(self, model, integrator):
        self.model = model
        self.integrator = integrator
        self.states = []
    def append(self, S, E):
        self.states.append(
            IsingState(config={}, S=S, E=E)
        )
        return
    def exportState(self):
        # Embed index, label
        for idx, s in enumerate(self.states):
            s.info['idx'] = idx
            s.info['label'] = 'ISING%06d' % idx
        # Descriptor matrix
        n_samples = len(self.states)
        n_dim = self.model.config['N']
        IX = np.zeros((n_samples, n_dim), dtype='float64')
        for i in range(n_samples):
            IX[i,:] = np.copy(self.states[i].S)
        # Setup state
        state = State(
            ising_J=self.model.J,
            ising_B=self.model.B
        )
        state.register("generate_ising", self.model.config)
        state["configs"] = self.states
        state["labels"] = [ s.info for s in self.states ]
        state["IX"] = IX
        state["n_samples"] = n_samples
        state["n_dim"] = n_dim
        return state
    def pickle(self, pfile='kernel.svmbox.pstr'):
        pstr = pickle.dumps(self)
        with open(pfile, 'w') as f:
            f.write(pstr)
        return




class IsingState(object):
    def __init__(self, config={}, S=np.array([]), E=None):
        if S.shape[0] > 0:
            self.S = np.copy(S)
            self.N = self.S.shape[0]
            self.E = E
        else:
            method = config['spawn']
            self.config = { k: config[k] for k in config }
            N = config['N']
            if method == 'random':
                S = np.random.randint(low=0, high=2, size=N)
            elif method == 'zeros':
                S = np.zeros((N,))
            S = (S-0.5)*2
            self.N = N
            self.S = S
            self.E = E
        self.info = { "E": self.E, "N": self.N }
        return

class IsingIntegrator(object):
    def __init__(self, config):
        self.kT = config['kT']
    def integrate(self,
            model,
            state,
            n_steps,
            anneal=np.array([]),
            sample_every=1,
            sample_start=0,
            verbose=True):
        ensemble = IsingEnsemble(model, self)
        # Energy of initial configuration
        e_B, e_J = model.energy_split(state)
        E0 = e_J+e_B
        if verbose: print "Step= %4d    Energy=%+1.7e [e_B=%+1.4e e_J=%+1.4e]" % (0, E0, e_B, e_J)
        # Annealing
        if anneal.shape[0] == 0:
            anneal = [ self.kT for n in range(n_steps) ]
        else:
            assert anneal.shape[0] == n_steps
        # Integrate
        for n in range(n_steps):
            kT = anneal[n]
            T = -1*state.S
            for i in range(state.N):
                Jij_Sj = model.J[i].dot(state.S)
                e0 = - state.S[i] * (Jij_Sj + model.B[i])
                e1 = - T[i] * (Jij_Sj + model.B[i])
                accept = False
                if e1 < e0:
                    accept = True
                else:
                    p_acc = np.exp(-(e1-e0)/    kT)
                    p = np.random.uniform()
                    #print p_acc
                    if p < p_acc:
                        accept = True
                if accept:
                    #print "Accept"
                    state.S[i] = T[i]
            e_B, e_J = model.energy_split(state)
            E = e_J+e_B
            if verbose: print "Step= %4d    Energy=%+1.4e [e_B=%+1.4e e_J=%+1.4e]   T=%+1.3e" % (n, E, e_B, e_J, kT)
            if n >= sample_start and n % sample_every == 0:
                ensemble.append(state.S, E)
        return ensemble













