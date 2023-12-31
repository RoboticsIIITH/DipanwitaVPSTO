import numpy as np
import cma
import threading
import concurrent.futures

from .obf import OBF

np.seterr(all="ignore") # ignore sqrt warning as it is handled inside the code

class VPSTOOptions():
    def __init__(self, ndof):
        # Initialize with default parameters
        self.ndof = ndof
        self.vel_lim = 0.1 * np.ones(ndof)  # -vel_lim < dq < vel_lim (element-wise)
        self.acc_lim = 1.0 * np.ones(ndof)  # -acc_lim < ddq < acc_lim (element-wise)
        self.N_eval = 100                   # length of trajectories in cost function
        self.N_via = 5                      # number of via-points
        self.pop_size = 25                  # number of trajectories per population
        self.sigma_init = 0.5               # initial variance of via-points for CMA-ES algo
        self.max_iter = 1000                # maximum number of vpsto iterations
        self.CMA_diagonal = False           # Set to True for faster, less accurate optimization (linear complexity)
        self.multithreading = False         # Set to True for concurrently executing the cost evaluation
        
class VPSTOSolution():
    def __init__(self, ndof):
        self.ndof = ndof
        self.candidates = dict()
        self.candidates['pos'] = None
        self.candidates['vel'] = None
        self.candidates['acc'] = None
        self.candidates['T'] = 0.0
        self.w_best = None
        self.T_best = 0.0
        self.p_next = None
        self.loss_list = []
        
    def get_trajectory(self, t):
        if self.w_best is None:
            print('No solution available. Run optimization first.')
            return [], [], []
            
        N_via = int(len(self.w_best) / self.ndof) - 3
        obf = OBF(self.ndof)
        obf.setup_task(self.T_best * np.ones(N_via) / N_via)
        Phi = obf.get_Phi(t)
        dPhi = obf.get_dPhi(t)
        ddPhi = obf.get_ddPhi(t)
        return (Phi@self.w_best).reshape(-1,self.ndof), (dPhi@self.w_best).reshape(-1,self.ndof), (ddPhi@self.w_best).reshape(-1,self.ndof)
        
    def shift_solution_forward(self, delta_t, N_next=0):
        if self.w_best is None:
            print('No solution available. Run optimization first.')
            return
        if N_next == 0:
            N_next = self.opt.N_via
        obf = OBF(self.ndof)
        obf_via.setup_task(self.T_best * np.ones(N_via) / N_via)
        T_next = self.T_best - delta_t
        t_via = np.linspace(0, T_next, N_next+1) + dt
        self.p_next = (obf_via.get_Phi(t_via) @ self.w_best).reshape(-1, self.ndof)[1:].flatten()

class VPSTO():
    def __init__(self, ndof):
        self.opt = VPSTOOptions(ndof)
        self.sol = VPSTOSolution(ndof)
        
    def setup_basis(self, fix_qT=False):
        s_eval = np.linspace(0., 1., self.opt.N_eval)
        obf = OBF(self.opt.ndof)
        obf.setup_task(np.ones(self.opt.N_via)/self.opt.N_via)
        
        self.Phi = obf.get_Phi(s_eval)
        self.dPhi = obf.get_dPhi(s_eval)
        self.ddPhi = obf.get_ddPhi(s_eval)
        
        if fix_qT is False:
            self.dPhi_p = self.dPhi[:, self.opt.ndof:-2*self.opt.ndof]
            self.dPhi_b = np.concatenate((self.dPhi[:, :self.opt.ndof],
                                          self.dPhi[:, -2*self.opt.ndof:]), axis=1)
            self.ddPhi_p = self.ddPhi[:, self.opt.ndof:-2*self.opt.ndof]
            self.ddPhi_b = np.concatenate((self.ddPhi[:, :self.opt.ndof],
                                           self.ddPhi[:, -2*self.opt.ndof:]), axis=1)
        else:
            self.dPhi_p = self.dPhi[:, self.opt.ndof:-3*self.opt.ndof]
            self.dPhi_b = np.concatenate((self.dPhi[:, :self.opt.ndof],
                                          self.dPhi[:, -3*self.opt.ndof:]), axis=1)
            self.ddPhi_p = self.ddPhi[:, self.opt.ndof:-3*self.opt.ndof]
            self.ddPhi_b = np.concatenate((self.ddPhi[:, :self.opt.ndof],
                                           self.ddPhi[:, -3*self.opt.ndof:]), axis=1)
        
        sigma_p = np.linalg.inv(self.ddPhi_p.T @ self.ddPhi_p)
        self.ddPhi_p_pinv = sigma_p @ self.ddPhi_p.T
        eigv, V = np.linalg.eigh(sigma_p)
        D = np.diag(eigv / np.max(eigv))
        sigma_via = V @ D @ V.T
        self.sigma_p_chol = np.linalg.cholesky(sigma_via)
        self.sigma_p_chol_inv = np.linalg.inv(self.sigma_p_chol)

    def get_duration(self, q_via_list, dq_bound):
        pop_size = len(q_via_list)
        dq_q = (q_via_list @ self.dPhi[:,:-2*self.opt.ndof].T).reshape(pop_size, -1, self.opt.ndof)
        dq_dq = (self.dPhi[:,-2*self.opt.ndof:] @ dq_bound).reshape(-1, self.opt.ndof)
        T_dq = np.maximum(np.max(dq_q / (self.opt.vel_lim - dq_dq), axis=(1, 2)),
                          np.max(- dq_q / (self.opt.vel_lim + dq_dq), axis=(1, 2)))
        
        ddq_q = (q_via_list @ self.ddPhi[:,:-2*self.opt.ndof].T).reshape(pop_size, -1, self.opt.ndof)
        ddq_dq = (self.ddPhi[:,-2*self.opt.ndof:] @ dq_bound).reshape(-1, self.opt.ndof)
        T_p = ddq_dq / (2. * self.opt.acc_lim)
        T_ddq = np.maximum(np.max(T_p + np.nan_to_num(np.sqrt(T_p**2 + ddq_q / self.opt.acc_lim), nan=-np.inf), axis=(1, 2)),
                           np.max(-T_p + np.nan_to_num(np.sqrt(T_p**2 - ddq_q / self.opt.acc_lim), nan=-np.inf), axis=(1, 2)))
        return np.maximum(T_dq, T_ddq)
    
    def compute_phenotype_candidates(self, p_list, q_0, q_T, dq_bound):
        pop_size = len(p_list)
        # Compute duration of each movement and roll out
        if q_T is None:
            q_via_list = np.concatenate((np.tile(q_0, (pop_size, 1)), p_list), axis=1)
        else:
            q_via_list = np.concatenate((np.tile(q_0, (pop_size, 1)), 
                                         p_list, 
                                         np.tile(q_T, (pop_size, 1))), axis=1)
        T_list = self.get_duration(q_via_list, dq_bound)
        w_list = np.concatenate((q_via_list, 
                                 np.diag(T_list) @ np.tile(dq_bound, (pop_size, 1))), axis=1)
        self.sol.candidates['pos'] = (w_list @ self.Phi.T).reshape(pop_size, -1, self.opt.ndof)
        self.sol.candidates['vel'] = (np.diag(1/T_list) @ (w_list @ self.dPhi.T)).reshape(pop_size, -1, self.opt.ndof)
        self.sol.candidates['acc'] = (np.diag(1/T_list**2) @ (w_list @ self.ddPhi.T)).reshape(pop_size, -1, self.opt.ndof)
        self.sol.candidates['T'] = T_list
        
    def call_loss_multithreading(self, loss, candidate, costs, idx):
        costs[idx] = loss(candidate)
        
    def loss_multithread(self, loss):
        costs = np.empty(self.opt.pop_size)
        candidates = []
        for i in range(self.opt.pop_size):
            candidates.append({'pos': self.sol.candidates['pos'][i],
                               'vel': self.sol.candidates['vel'][i],
                               'acc': self.sol.candidates['acc'][i],
                               'T': self.sol.candidates['T'][i]})
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.opt.pop_size) as executor:
          futures = []
          for i in range(self.opt.pop_size):
            futures.append(executor.submit(self.call_loss_multithreading, loss, candidates[i], costs, i))
          for future in concurrent.futures.as_completed(futures):
            future.result()
        return costs
        
    def minimize(self, loss, q_0, q_T=None, dq_bound=None):
        if dq_bound is None:
            dq_bound = np.zeros(2*self.opt.ndof)
        if q_T is None:
            dim_x = self.opt.ndof * self.opt.N_via
            self.setup_basis(fix_qT = False)
            mu_p = - self.ddPhi_p_pinv @ self.ddPhi_b @ np.concatenate((q_0, dq_bound))
        else:
            dim_x = self.opt.ndof * (self.opt.N_via - 1)
            self.setup_basis(fix_qT = True)
            mu_p = - self.ddPhi_p_pinv @ self.ddPhi_b @ np.concatenate((q_0, q_T, dq_bound))
    
        if self.sol.p_next is None:
            x_init = np.zeros(dim_x)
        else:
            x_init = self.sigma_p_chol_inv @ (self.sol.p_next - mu_p)
            self.sol.p_next = None
        
        cmaes = cma.CMAEvolutionStrategy(x_init, self.opt.sigma_init, 
                                         {'CMA_diagonal': self.opt.CMA_diagonal, 
                                          'verbose': -1,
                                          'CMA_active': True,
                                          'popsize': self.opt.pop_size,
                                          'tolfun': 1e-9})
        self.sol.loss_list = []
        i = 0
        while not cmaes.stop() and i < self.opt.max_iter:
            x_samples = np.array(cmaes.ask())
            p_samples = mu_p+(self.sigma_p_chol@x_samples.T).T
            self.compute_phenotype_candidates(p_samples, q_0, q_T, dq_bound)
            if self.opt.multithreading is False:
                costs = loss(self.sol.candidates)
            else:
                costs = self.loss_multithread(loss)
            cmaes.tell(x_samples, costs)
            self.sol.loss_list.append(np.min(costs))
            print('# VP-STO iteration: ', i, 'cost: ',np.min(costs), end='\r')
            i += 1
        
        x_best = cmaes.result.xfavorite
        p_best = mu_p+self.sigma_p_chol@x_best
        if q_T is None:
            q_via_list = np.concatenate((q_0, p_best))
        else:
            q_via_list = np.concatenate((q_0, p_best, q_T))
        self.sol.w_best = np.concatenate((q_via_list, dq_bound))
        self.sol.T_best = self.get_duration(q_via_list.reshape(1,-1), dq_bound)[0]
        
        print('VP-STO finished after', i, 'iterations with a final loss of', self.sol.loss_list[-1])
        
        return self.sol
