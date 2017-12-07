#! /usr/bin/env python
import librmt as rmt
import numpy as np

def generate_bits(config):
    gen_type = config["generate"]["type"]
    dim = config["generate"]["d"]
    if gen_type == "binary":
        s = 2.**0.5*np.random.randint(0, 2, size=dim) # NOTE factor 2 => std-dev = 1.0
    elif gen_type == "spins":
        s = np.random.randint(0, 2, size=dim)
        s = 2.0*(s - 0.5)
    else: raise NotImplementedError(gen_type)
    return s

class HamiltonianHopkins(object):
    def __init__(self, config):
        self.tag = "hopkins"
        self.d = config['generate']['d']
        J_min = config['hopkins']['J_min']
        J_max = config['hopkins']['J_max']
        J_sc  = config['hopkins']['J_scale']
        B_min = config['hopkins']['B_min']
        B_max = config['hopkins']['B_max']
        B_sc  = config['hopkins']['B_scale']
        self.s_h = 2.*(np.random.randint(0, 2, size=self.d)-0.5)
        J = np.outer(self.s_h, self.s_h)
        J = J_sc*J
        B = np.random.uniform(low=B_min, high=B_max, size=self.d)
        B = B_sc*B
        self.J = J
        self.B = B
        return
    def evaluate(self, s):
        e1 = - self.B.dot(s)/(4*self.d)**0.5
        e2 = -0.5*s.dot(self.J).dot(s)/self.d 
        return e1+e2, e1, e2

class HamiltonianIsing(object):
    def __init__(self, config):
        self.tag = "ising"
        self.d = config['generate']['d']
        J_min = config['ising']['J_min']
        J_max = config['ising']['J_max']
        J_sc  = config['ising']['J_scale']
        B_min = config['ising']['B_min']
        B_max = config['ising']['B_max']
        B_sc  = config['ising']['B_scale']
        # J & B matrices
        J = np.random.uniform(low=J_min, high=J_max, size=(self.d,self.d))
        B = np.random.uniform(low=B_min, high=B_max, size=self.d)
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
    def evaluate(self, s):
        e1 = - self.B.dot(s)/(4*self.d)**0.5
        e2 = -0.5*s.dot(self.J).dot(s)/self.d 
        return e1+e2, e1, e2

class HamiltonianPolynomial(object):
    def __init__(self, config):
        self.d = config["generate"]["d"]
        self.n_modes = config["poly"]["n_modes"]
        self.degree = config["poly"]["degree"]
        E_min = config["poly"]["E_min"]
        E_max = config["poly"]["E_max"]
        E_sc = config["poly"]["E_scale"]
        # MODES
        V = np.random.randint(0, 2, size=(self.n_modes,self.d))
        V = 2.*(V-0.5)
        norm = np.sum(V**2, axis=1)**0.5
        V = (V.T/norm).T
        self.V = V
        # POLYOMIAL PROJECTIONS & COEFFS
        self.prjs = []
        self.Es = []
        n_coeffs = 0
        for n in range(1, self.degree+1):
            n_comps = int(self.n_modes**(1./n)+0.5)
            assert n_comps > 0
            idcs_n = []
            for nn in range(n):
                a = np.arange(self.n_modes)
                np.random.shuffle(a)
                idcs = sorted(a[0:n_comps])
                idcs_n.append(idcs)
            self.prjs.append(idcs_n)
            n_coeffs += n_comps**n
            self.Es.append(np.random.uniform(low=E_min, high=E_max, size=n_comps**n))
        return
    def evaluate(self, s):
        es = []
        u = self.V.dot(s)
        for n in range(1, self.degree+1):
            idcs_n = self.prjs[n-1]
            poly_u = u[idcs_n[0]]
            for nn in range(1, n):
                poly_u = np.outer(poly_u, u[idcs_n[nn]]).flatten()
            es.append(self.Es[n-1].dot(poly_u))
        return np.sum(es), np.sum(es[0:-1]), np.sum(es[-1])

class HamiltonianGaussian(object):
    def __init__(self, config):
        self.tag = "gaussian"
        self.d = config["generate"]["d"]
        self.n_wells = config["gaussian"]["n_wells"]
        E_min = config['gaussian']['E_min']
        E_max = config['gaussian']['E_max']
        E_sc  = config['gaussian']['E_scale']
        S_min = config['gaussian']['S_min']
        S_max = config['gaussian']['S_max']
        E = E_sc*np.random.uniform(low=E_min, high=E_max, size=self.n_wells)
        V = np.random.randint(0, 2, size=(self.n_wells,self.d))
        V = 2.*(V-0.5)
        norm = np.sum(V**2, axis=1)**0.5
        V = (V.T/norm).T
        S = np.random.uniform(low=S_min, high=S_max, size=self.n_wells)
        self.E = E
        self.V = V
        self.S = S
    def evaluate(self, s):
        s_norm = s/(s.dot(s))**0.5

        e_sum = np.sum(
            self.E*np.exp((self.V.dot(s_norm)-1.)**2/(2.*self.S**2))
        )
        return e_sum, 0.0, 0.0

class HamiltonianEnsemble(object):
    def __init__(self, models, config):
        self.models = [ m(config) for m in models ]
        self.tag = '+'.join([ m.tag for m in self.models ])
        self.generator = self.models[0]
    def evaluate(self, s):
        e = 0.0
        e1 = 0.0
        e2 = 0.0
        for m in self.models:
            de, de1, de2 = m.evaluate(s)
            e += de
            e1 += de1
            e2 += de2
        return e, e1, e2

def generate_samples(config, generator_fct, model):
    n_snaps = config["N"]
    samples = []
    for i in range(n_snaps):
        s = generator_fct(config)
        e, e1, e2 = model.evaluate(s)
        samples.append([s, e, e1, e2])
    samples = sorted(samples, key=lambda s: -s[1])
    # Export as state
    state = rmt.State(
        model=model,
        configs=[],
        IX=[],
        T=[])
    for idx, s in enumerate(samples):
        config = rmt.State(target=s[1], e1=s[2], e2=s[3], idx=idx, label="L%d" % idx)
        state["configs"].append(config)
        state["IX"].append(s[0])
        state["T"].append(s[1])
        print idx, s[0][0:5], s[1], s[2]/s[1], s[3]/s[1]
    state["IX"] = np.array(state["IX"])
    state["T"] = np.array(state["T"])
    state["has_T"] = True
    return state

def write_spectrum(state):
    ofs = open('samples.txt', 'w')
    for idx, c in enumerate(state["configs"]):
        ofs.write('%d %+1.7e %+1.7e %+1.7e\n' % (idx, state["T"][idx], c["e1"], c["e2"]))
    ofs.close()
    hist, bin_edges = np.histogram(a=state["T"], bins=100, density=True)
    db = bin_edges[1]-bin_edges[0]
    bin_centres = (np.array(bin_edges)+db)[:-1]
    hist = np.array([list(bin_centres), list(hist)], dtype='float64').T
    np.savetxt('samples-hist.txt', hist)
    hist, bin_edges = np.histogram(a=[c["e1"] for c in state["configs"]], bins=100, density=True)
    db = bin_edges[1]-bin_edges[0]
    bin_centres = (np.array(bin_edges)+db)[:-1]
    hist = np.array([list(bin_centres), list(hist)], dtype='float64').T
    np.savetxt('samples-hist-e1.txt', hist)
    hist, bin_edges = np.histogram(a=[c["e2"] for c in state["configs"]], bins=100, density=True)
    db = bin_edges[1]-bin_edges[0]
    bin_centres = (np.array(bin_edges)+db)[:-1]
    hist = np.array([list(bin_centres), list(hist)], dtype='float64').T
    np.savetxt('samples-hist-e2.txt', hist)
    return

Hamiltonian = {
    "hopkins": HamiltonianHopkins,
    "ising": HamiltonianIsing
}

if __name__ == "__main__":
    config = {
        "N": 10000,
        "generate": { 
            "d": 100, "type": "spins"
        },
        "hopkins": {
            "J_min": -1.0, "J_max": +1.0, "J_scale": 1.0, "B_min": -1.0, "B_max": +1.0, "B_scale": 1.0
        },
        "ising": {
            "J_min": -1.0, "J_max": +1.0, "J_scale": 1.0, "B_min": -1.0, "B_max": +1.0, "B_scale": 1.0
        },
        "gaussian": {
            "n_wells": None, "E_min": -1.0, "E_max": -0.1, "E_scale": 1.0, "S_min": 0.8, "S_max": 1.2
        },
        "poly": {
            "n_modes": 125, "degree": None, "E_min": -1.0, "E_max": -0.1, "E_scale": 1.0, "S_min": 0.8, "S_max": 1.2
        }
    }

    config = {
        "N": 10000,
        "generate": { 
            "d": None, "type": "binary" # "spins"
        },
        "hopkins": {
            "J_min": -1.0, "J_max": +1.0, "J_scale": 1.0, "B_min": -1.0, "B_max": +1.0, "B_scale": 0.0
        }
    }
    for dim in [128]:
        #for n in [20,1,3,5,10,20]:
        for n in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20]:
            htype = "hopkins"
            config["generate"]["d"] = dim
            h_model = HamiltonianEnsemble([ HamiltonianHopkins for i in range(n) ], config)
            state = generate_samples(config, generate_bits, h_model)
            state.pickle('state_%s-%02d_%s_%04d.jar' % (htype, n, config["generate"]["type"], config["generate"]["d"]))
            #write_spectrum(state)
            #raw_input('...')
        
