import os
import sys; sys.path.append( "/home/xiaozy/Global/lib/hafqmc-master/hafqmc" )
import scipy.signal
import hafqmc.train
import hafqmc.config

cfg = hafqmc.config.example()

# cfg.molecule = dict(atom="N 0 0 0; N 0 0 2.4;", basis="ccpvdz", unit='B')
cfg.restart.hamiltonian = "AFQMC_hamiltonian.pkl"

cfg.ansatz.wfn_spinmix = False
cfg.ansatz.propagators[0].max_nhs = 100
cfg.ansatz.propagators[0].aux_network = None
cfg.ansatz.propagators[0].init_tsteps = [0.01] * 1
cfg.ansatz.propagators[0].sqrt_tsvpar = True
cfg.ansatz.propagators[0].init_random = 0.1
cfg.ansatz.propagators[0].hermite_ops = False
#
# cfg.ansatz.propagators[0].mf_subtract = True
cfg.ansatz.propagators[0].mf_subtract = False
#
cfg.ansatz.propagators[0].spin_mixing = True

cfg.ansatz.propagators[0].expm_option = ["scan", 2, 1]

cfg.optim.optimizer = "adabelief"
cfg.optim.grad_clip = 1.
cfg.optim.iteration = 15000
cfg.optim.lr.start = 3e-4
cfg.optim.lr.delay = 5e3
cfg.optim.lr.decay = 1.

cfg.sample.batch = 1000
cfg.sample.sampler = {"name": "hmc", "dt": 0.1, "length": 1.}
cfg.sample.burn_in = 100

cfg.loss.sign_factor = 1.0
cfg.loss.sign_target = 0.7

cfg.seed = 1
cfg.log.level = "info"

if __name__ == "__main__": 
    hafqmc.train.train(cfg)
