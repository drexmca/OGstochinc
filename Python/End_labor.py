import numpy as np
from scipy.optimize import fsolve, root, brentq
from quantecon import markov
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as plt
from ellipse import estimation
import time
import datetime

class OG(object):
    """OG object that solves overlapping generations problems.
    
    ---Methods---
    update_Psi: Updates the policy functions
    jac: Calculates the jacobian at a specific point
    obj: Calculates Euler errors for capital and labor
    update_B_and_N: Updates the total capital and labor objects
    set_state: resets state variables
    calc_SS: iterates until convergence is reached
    """

    def __init__(self, household_params, firm_params):
        """Instantiate the state parameters of the OG model.

        ---Attributes---
        ### Household params ###
        Num: int, number of agents in smallest ability group
        S: int, number of time periods in agent life
        J: int, number of ability groups
        beta: float, discount factor
        sigma: float, constant of relative risk aversion
        Pi: (J,J) array, Probability of movement between ability groups
        e: (J,) array, effective labor units given ability type
        lambda_bar: (J,) array, steady state ergodic distribution ability of agents
        agents: (J,) array, number of agents in each ability group
        M: int, total number of agents
        shocks: (S,M) agents, ability life path of all agents
        theta: float, agent elliptical utility labor param

        ### Firm Params ###
        A: float, firm production multiplier
        alpha: float, cobb-douglas production parameter
        delta: float, depreciation

        ### State Params ###
        B: (S,M) array, Savings decision for M agents, S periods
        N: (S,M) array, Labor decisions for M agents, S periods
        r: float, interest rate
        w: float, wage rate
        grid_size: int, number of points in policy function grid
        b0min: float, min endpoint for policy function interpolation
        b0max: float, max endpoint for policy function interpolation
        Grid: (S,J,grid_size) array, policy grids for period s, ability j
        Psi_b: (S,J,grid_size) array, Capital policy function for period s, ability j
        Psi_n: (S,J,grid_size) array, Labor policy function for period s, ability j
        ellip_up: float, estimated upsilon from elliptical utility package
        ellip_b: float, estimated b from elliptical utility package
        """
        self.start = time.time()
        # Make household parameters as attributes.
        (self.Num, self.S, self.J, beta_annual, self.sigma,
         self.Pi, self.e, self.theta) = household_params
        # Convert beta_annual to the beta for a period.
        self.beta = beta_annual**(80/self.S)
        
        # Create the markov chain object so we can simulate the shock paths.
        MC = markov.core.MarkovChain(self.Pi)
        # Find the ergodic distribution.
        self.lambda_bar = MC.stationary_distributions
        # Determine the number of agents initially in each ability type.
        weights = (self.lambda_bar/(self.lambda_bar.min())*self.Num).astype('int')
        self.M = np.sum(weights)
        initial_e = np.zeros(self.M)
        for agent in np.cumsum(weights[0][1:]):
            initial_e[agent:] += 1
        # Simulate M shock paths.
        self.shocks = MC.simulate(self.S, initial_e, random_state=1).T
        self.abilities = self.e[self.shocks]
        
        # Set the firm parameters as attributes.
        (self.A, self.alpha, delta_annual) = firm_params
        # Convert delta_annual to the delta corresponding to the period length.
        self.delta = 1-(1-delta_annual)**(80/self.S)
        
        # Initialize the state.
        self.B = np.empty((self.S, self.M))
        self.B[0] = 0.0
        self.N = np.ones((self.S, self.M))*.9
        # Set initial guesses for r and w.
        self.r, self.w = 0.312381791072, 0.637945371028
        # Initialize
        self.grid_size = 100
        self.b0min = 0.
        self.b0max = 10.
        self.Grid = np.empty((self.S, self.J, self.grid_size))
        for j in range(self.J):
            self.Grid[-1,j] = np.linspace(self.b0min, self.b0max, self.grid_size)
        self.Psi_b = np.empty((self.S, self.J, self.grid_size))
        self.Psi_n = np.empty((self.S, self.J, self.grid_size))
        self.Psi_b[-1] = 0.0
        self.ellip_b, self.ellip_up = estimation(self.theta,1.)

        
                
    def update_Psi(self):
        """Update the policy functions.
        
        Calculates S labor decision policy functions, and S-1 capital savings policy 
        functions. Also calculates upper and lower bounds for a feasible region 
        for each ability and age in the model. 
        """

        # Make Psi_n_S.
        for j in range(self.J):
            for i in range(self.grid_size):
                grid = self.Grid[-1,j,i]
                def get_n_S(n):
                    return (self.w*self.e[j]*(self.w*self.e[j]*n+grid*(1+self.r))**-sigma
                           -self.ellip_b*n**(self.ellip_up-1)*(1-n**self.ellip_up)**(1/self.ellip_up-1))
                n_S, r = brentq(get_n_S, 0, 1, full_output=1)
                if r.converged!=1:
                    print 'n_S did not converge!'
                self.Psi_n[-1,j,i] = n_S        
        
        for s in range(self.S-2,-1,-1):
            for j in range(self.J):
                # Make Psi_n_S
                            
                lb = -999                        
                for j_ in range(self.J):
                    psi_b = UnivariateSpline(self.Grid[s+1,j_], self.Psi_b[s+1,j_])                
                    psi_n = UnivariateSpline(self.Grid[s+1,j_], self.Psi_n[s+1,j_])                
                    def get_lb(b1):
                        b2 = psi_b(b1)
                        n1 = psi_n(b1)
                        c1 = b1*(1+self.r)+self.w*self.e[j_]*n1-b2
                        return c1
                    guess = (-self.w*self.e[j_]*1.+psi_b(0))/(1+self.r)
                    lb_, info, ier, mesg = fsolve(get_lb, guess, full_output=1)
                    lb = np.max([lb,lb_])
                    if ier!=1:
                        print s, j, j_, 'The lower bound wasn\'t calculated correctly.'
                self.lb = lb
                self.b0min = (lb-self.w*self.e[j]*0.)/(1+self.r)+1e-5
                self.Grid[s,j] = np.linspace(self.b0min, self.b0max, self.grid_size)
                ub = self.Grid[s,j]*(1+self.r)+self.w*self.e[j]*1.
                self.ub = ub
                for i in range(self.grid_size):
                    obj = lambda x: self.obj(x, self.Grid[s,j,i], s, j, i)
                    jac = lambda x: self.jac(x, self.Grid[s,j,i], s, j, i)
#                     print obj(((lb+ub[i])/2., self.Psi_n[s+1,j,i]))
#                     print jac(((lb+ub[i])/2., self.Psi_n[s+1,j,i]))
#                     print (lb+ub[i])/2, self.lb
                    #TODO Fix feasible region
                    #TODO Check root method for jacobian evaluation
                    sol = root(obj, ((lb+ub[i])/2., self.Psi_n[s+1,j,i]), jac=jac)
                    print sol.fun, sol.x, self.Psi_n[s+1,j,i]
                    (self.Psi_b[s,j,i], self.Psi_n[s,j,i]), ier = sol.x, sol.success
                    # self.Psi[s,j,i], info, ier, mesg = fsolve(obj, (lb+ub[i])/2., full_output=1)
                    if ier!=1:
                        sol = root(obj, ((lb+ub[i])/2., .9999999), jac=jac)
                        print sol.fun, sol.x, self.Psi_n[s+1,j,i]
                        (self.Psi_b[s,j,i], self.Psi_n[s,j,i]), ier = sol.x, sol.success
                        print s, j, i, sol.message
                        #print s, j, i, 'no converge'
                        
    def jac(self, b1n0, b0, s, j, i):
        b1, n0 = b1n0
        if b1<self.lb:
            print 1111111
            return np.zeros((2,2))
        if b1>self.ub[i]:
            print 2222222
            return np.zeros((2,2))
        if n0<0:
            print 33333333
            return np.zeros((2,2))
        if n0>1:
            print 4444444
            return np.zeros((2,2))
                
        h = 1e-15
        
        b1n0_11 = (b1+h, n0)
        b1n0_12 = (b1, n0+h)
        b1n0_21 = (b1-h, n0)
        b1n0_22 = (b1, n0-h)
        
        Dcen1 = (self.obj(b1n0_11, b0, s, j, i)-self.obj(b1n0_21, b0, s, j, i))/(2.*h)
        Dcen2 = (self.obj(b1n0_12, b0, s, j, i)-self.obj(b1n0_22, b0, s, j, i))/(2.*h)
        return np.array([Dcen1, Dcen2])
    
    def obj(self, b1n0, b0, s, j, i):

        b1, n0 = b1n0
        psi_n = UnivariateSpline(self.Grid[s+1,j], self.Psi_n[s+1,j])
        n1 = psi_n(b1)
        b2 = np.array([UnivariateSpline(self.Grid[s+1,j_], self.Psi_b[s+1,j_])(b1) for j_ in xrange(self.J)])
          
            
        c0 = b0*(1+self.r)+self.w*self.e[j]*n0-b1
        #TODO n1 needs to be an expected value
        c1 = b1*(1+self.r)+self.w*self.e*n1-b2
        err_n = self.w*self.e[j]*c0**-self.sigma-self.ellip_b*n0**(self.ellip_up-1)*(1-n0**self.ellip_up)**(1/self.ellip_up-1)
        err_c = c0**-self.sigma-self.beta*(1+self.r)*np.sum(self.Pi[j]*(c1**-self.sigma))
        err = np.array([err_c, err_n])
        return err

    
    def update_B_and_N(self):
        for s in range(self.S-1):
            for j in range(self.J):
                psi_b = UnivariateSpline(self.Grid[s,j], self.Psi_b[s,j])
                psi_n = UnivariateSpline(self.Grid[s,j], self.Psi_n[s,j])
                self.B[s+1, self.shocks[s]==j] = psi_b(self.B[s, self.shocks[s]==j])
                self.N[s, self.shocks[s]==j] = psi_n(self.B[s, self.shocks[s]==j])
        #TODO Check and see if last period labor decision was updated correctly
        for j in range(self.J):
            psi_n = UnivariateSpline(self.Grid[S-1,j], self.Psi_n[S-1,j])
            self.N[S-1, self.shocks[S-1]==j] = psi_n(self.B[S-1, self.shocks[S-1]==j])

    def set_state(self):
        self.L = np.sum(self.N)
        self.K = np.sum(self.B)

        self.r = self.alpha*self.A*((self.L/self.K)**(1-self.alpha))-self.delta
        self.w = (1-self.alpha)*self.A*((self.K/self.L)**self.alpha)
        
        self.b0min = -self.w*(2+self.r)/(1+self.r)**2*np.min(self.e)*np.min(self.N)+1e-5
        #TODO This may or may not be necessary
        for j in range(self.J):
            self.Grid[-1,j] = np.linspace(self.b0min, self.b0max, self.grid_size)

    
    def calc_SS(self, tol=1e-10, maxiter=100):
        self.start = time.time()
        diff = 1
        count = 0
        while diff>tol and count<maxiter:
            r0, w0 = self.r, self.w
#             B0 = self.B
#             N0 = self.N
            self.update_Psi()
            self.update_B_and_N()
            self.set_state()
            print 'r and w', self.r, self.w
            print 'max of B', np.max(self.B)
            count += 1
            diff = max(np.abs(self.r-r0), np.abs(self.w-w0))
            self.r, self.w = .2*self.r+.8*r0, .2*self.w+.8*w0
            print count, diff
        self.write_output()

    def write_output(self):
        """
        Writes ouputfile to 'Output/output.txt'

        Example:
        Date Time
        Total SS computation time:
        Total Labor:
        Total Capital:
        SS interest rate:
        SS wage:
        """
        self.date = time.strftime("%m/%d/%Y")
        self.time = time.strftime("%I:%M:%S %p")
        total_time = str(datetime.timedelta(seconds=time.time() - self.start))
        output = open('Output/output.txt', 'w')
        output.write('{} {}\n'.format(self.date, self.time))
        output.write('Total SS computation time: {}\n'.format(total_time))
        output.write('Total Labor supply: {}\nTotal Capital supply: {}\nSteady State interest rate: {}\nSteady State Wage: {}\n'.format(self.L, self.K, self.r, self.w))
        output.close()





Num = 100
S = 4
J = 2
beta_annual = .96
sigma = 3.0
Pi = np.array([[0.4, 0.6],
               [0.6, 0.4]])
e = np.array([0.8, 1.2])
theta = 2.

household_params = (Num, S, J, beta_annual, sigma, Pi, e, theta)

A = 1.0
alpha = .35
delta_annual = .05
firm_params = (A, alpha, delta_annual)

og = OG(household_params, firm_params)
