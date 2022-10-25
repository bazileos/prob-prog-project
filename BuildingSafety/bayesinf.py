import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pymc3 as pm
import arviz as az

from scipy import stats
import scipy as sp

from math import sqrt as mathsqrt
from math import log as mathlog

from gym import Env
from gym.spaces import Discrete, Box

import sys

import os
import pickle

import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings("ignore")

from IPython.display import clear_output

from pprint import pprint as pp

from feastruct.pre.material import Steel
from feastruct.pre.section import Section
import feastruct.fea.cases as cases
from feastruct.fea.frame_analysis import FrameAnalysis2D
from feastruct.solvers.naturalfrequency import NaturalFrequency
from feastruct.solvers.linstatic import LinearStatic
from feastruct.solvers.feasolve import SolverSettings

from FORM.ERANataf import ERANataf
from FORM.ERADist import ERADist

class FEModel():
    def __init__(self, L=4, storeys=3, qd=3.6):
        self.L = L
        self.numStoreys = storeys
        self.numCols = 2 * self.numStoreys
        self.numModes = self.numStoreys

        self.qd = qd

        self.crosSecs = {'beam' : {'name' : 'IPE220',
                                   'tf' : 0.0092, 
                                   'bf' : 0.110, 
                                   'hw' : 0.2016, 
                                   'tw' : 0.0059},
                         'col' : {'name' : 'HEA300',
                                  'tf' : 0.014, 
                                  'bf' : 0.300, 
                                  'hw' : 0.262, 
                                  'tw' : 0.0085}}
        # Define the cross-sections
        for cs in ('beam', 'col'):
            self.crosSecs[cs]['A0'] = 2 * self.crosSecs[cs]['tf'] * self.crosSecs[cs]['bf'] + self.crosSecs[cs]['tw'] * self.crosSecs[cs]['hw']
            self.crosSecs[cs]['I0'] = 2 * (self.crosSecs[cs]['bf'] * self.crosSecs[cs]['tf'] ** 3 / 12 + \
                                self.crosSecs[cs]['bf'] * self.crosSecs[cs]['tf'] * (self.crosSecs[cs]['hw'] / 2 + self.crosSecs[cs]['tf'] / 2) ** 2) + \
                                self.crosSecs[cs]['tw'] * self.crosSecs[cs]['hw'] ** 3 / 12
        # Create the node coordinates
        self.nodeCoords = [*[(0., k * self.L) for k in range(self.numStoreys + 1)],
                    *[(self.L, k * self.L) for k in range(self.numStoreys + 1)]]

        self.elemInfo = [{'start' : k, 'end' : k + 1, 'area' : self.crosSecs['col']['A0'], 'ixx' : self.crosSecs['col']['I0']} for k in range(self.numCols + 1) if k != self.numStoreys]

        self.elemInfo.extend([{'start' : k, 'end' : k + (self.numStoreys + 1), 'area' : self.crosSecs['beam']['A0'], 'ixx' : self.crosSecs['beam']['I0']} for k in [c for c in range(1, self.numStoreys+1)]])

        # Copy the element's info in order to change the damages in-place 
        self.defaultElemInfo = self.elemInfo.copy()

        # Store the Boundary Conditions' info
        self.bcInfo = {0 : (0, 1, 5),
                self.numStoreys + 1 : (0, 1, 5)}

        self.analysis = self._create_analysis(default=True)

        # Calculate the equivalent points loads applied to the nodes, to replace the triangular load
        elemDistLoadVals = {k : {"startLoad" : self.qd * self.analysis.elements[k].nodes[0].y / (3 * L),
                            "endLoad" : self.qd * self.analysis.elements[k].nodes[1].y / (3 * L)} for k in range(3)}

        elemPointForces = {k : {'start' : elemDistLoadVals[k]['startLoad'] * self.L / 3 + elemDistLoadVals[k]['endLoad'] * self.L / 6,
                            'end' : elemDistLoadVals[k]['startLoad'] * self.L / 6 + elemDistLoadVals[k]['endLoad'] * self.L / 3} for k in range(3)}

        self.nodeForces = [elemPointForces[0]['start'], *[elemPointForces[k]['end'] + elemPointForces[k+1]['start'] for k in range(2)], elemPointForces[2]['end']]

        # Assign the Boundary Conditions
        self.freedom_case = cases.FreedomCase()

        for nodeId, constDofs in self.bcInfo.items():
            for dofId in constDofs:
                self.freedom_case.add_nodal_support(node=self.analysis.nodes[nodeId], val=0, dof=dofId)

        # Assign the applied loads
        self.load_case = cases.LoadCase()

        for nodeId, loadVal in enumerate(self.nodeForces):
            self.load_case.add_nodal_load(node=self.analysis.nodes[nodeId], val=loadVal * 1e3, dof=0)

        # Set up the analysis case (BCs and Loads)
        self.analysis_case = cases.AnalysisCase(freedom_case=self.freedom_case, load_case=self.load_case)

        # Set up the settings for eigen- and linear static analysis
        self.settings = SolverSettings()
        self.settings.natural_frequency.time_info = False
        self.settings.linear_static.time_info = False
        self.settings.natural_frequency.num_modes = self.numModes

        eigSolver = NaturalFrequency(analysis=self.analysis, analysis_cases=[self.analysis_case], solver_settings=self.settings)
        linStatSolver = LinearStatic(analysis=self.analysis, analysis_cases=[self.analysis_case], solver_settings=self.settings)

        # Run the default analyses
        eigSolver.solve()
        linStatSolver.solve()

        self.undamagedEigmodes = [self.analysis.elements[0].get_frequency_results(analysis_case=self.analysis_case, frequency_mode=k)[0] for k in range(self.numModes)]
        self.undamagedModalDispls = np.array([[self.analysis.nodes[k].get_dofs([True, True, False, False, False, True])[0].get_frequency_mode(self.analysis_case, kk)[1] for k in [1, 2, 3, 5, 6, 7]] for kk in (0, 1, 2)])
        self.undamagedDrift = self.analysis.nodes[self.numStoreys].get_displacements(analysis_case=self.analysis_case)[0]
        
    def _create_analysis(self, mat=Steel(), default=False):
        # create 2d frame analysis object
        analysis = FrameAnalysis2D()

        # create nodes
        for coords in self.nodeCoords:
            analysis.create_node(coords=coords)

        elInfo = self.defaultElemInfo if default else self.elemInfo
        # create elements
        for elem in elInfo:
            analysis.create_element(
                el_type="EB2-2D",
                nodes=(analysis.nodes[elem['start']], analysis.nodes[elem['end']]),
                material=mat,
                section=Section(area=elem['area'], ixx=elem['ixx']))
        
        return analysis
    
    def eigen_analysis(self, damages):
        for k, dam in enumerate(damages):
            self.elemInfo[k]['area'], self.elemInfo[k]['ixx'], _ = self._calc_reduced_props(dam, self.crosSecs['col'])
    
            self.analysis = self._create_analysis()

        solver = NaturalFrequency(analysis=self.analysis, analysis_cases=[self.analysis_case], solver_settings=self.settings)

        solver.solve()

        return np.array([self.analysis.nodes[k].get_dofs([True, True, False, False, False, True])[0].get_frequency_mode(self.analysis_case, 0)[1] for k in [1, 2, 3, 5, 6, 7]])

    def linear_static_analysis(self, damages, node=3, dof=0):
        for k, dam in enumerate(damages):
            self.elemInfo[k]['area'], self.elemInfo[k]['ixx'], _ = self._calc_reduced_props(dam, self.crosSecs['col'])
    
            self.analysis = self._create_analysis()


        solver = LinearStatic(analysis=self.analysis, analysis_cases=[self.analysis_case], solver_settings=self.settings)

        solver.solve()

        # return the displacement along dof, for the input node
        return self.analysis.nodes[node].get_displacements(analysis_case=self.analysis_case)[dof]
    
    # Misc functions
    def _calc_reduced_props(self, ddet, crosSec):
        ddet = np.clip(ddet, a_min=0., a_max=0.999)

        A = 4.0
        B = 2 * (crosSec['tw'] - crosSec['hw'] - 2 * crosSec['tf'] - 2 * crosSec['bf'])
        C = ddet * crosSec['A0']

        c = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

        A_ = 2 * (crosSec['tf'] - 2 * c) * (crosSec['bf'] - 2 * c) + (crosSec['tw'] - 2 * c) * (crosSec['hw'] + 2 * c)

        I_ = (
            2
            * (
                (crosSec['bf'] - 2 * c) * (crosSec['tf'] - 2 * c) ** 3 / 12
                + (crosSec['bf'] - 2 * c) * (crosSec['tf'] - 2 * c) * (crosSec['hw'] / 2 + crosSec['tf'] / 2) ** 2
            )
            + (crosSec['tw'] - 2 * c) * (crosSec['hw'] + 2 * c) ** 3 / 12
        )

        return A_, I_, c

    def _two_point_dist(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1.coords) - np.array(p2.coords)) ** 2))

    def _element_length(self, elem):
        return self._two_point_dist(elem.nodes[0], elem.nodes[1])

    def _element_mass(self, elem):
        return elem.material.rho * elem.section.area * self._element_length(elem)

def FORM_geom(g, distr):
    # initial check if there exists a Nataf object
    if not (isinstance(distr, ERANataf)):
        raise RuntimeError("Incorrect distribution. Please create an ERANataf object!")

    d = len(distr.Marginals)

    # objective function
    dist_fun = lambda u: np.linalg.norm(u)

    # parameters of the minimize function
    u0 = 0.1 * np.ones(d)  # initial search point

    # nonlinear constraint: H(u) <= 0
    H = lambda u: g(distr.U2X(u))
    cons = {"type": "ineq", "fun": lambda u: -H(u)}

    # method for minimization
    alg = "SLSQP"

    # use constraint minimization
    res = sp.optimize.minimize(dist_fun, u0, constraints=cons, method=alg)

    # unpack results
    u_star = res.x
    beta = res.fun
    if np.all(u_star < 0):
        beta *= -1

    # compute design point in orignal space and failure probability
    Pf = sp.stats.norm.cdf(-beta)

    return Pf

class frameEnv(Env):
    def __init__(self, noise, muA, muB, numSteps = 20, u=0.4/0.075**2, 
                 coeffVarA=0.5, coeffVarB=0.2, costReplace=10000, driftThres=0.024):

        self.model = FEModel()

        self.noise = noise
        self.muA = muA
        self.sigmaA = self.muA * coeffVarA
        self.muB = muB
        self.sigmaB = self.muB * coeffVarB
        self.u = u # the scale of the Gamma distributions
        self.R = np.eye(self.model.numCols) # correlation matrix for Nataf transformation

        self.logMuA = mathlog(self.muA**2/mathsqrt(self.muA**2+self.sigmaA**2))
        self.logSigmaA = mathsqrt(mathlog(1+self.sigmaA**2/self.muA**2))
 
        self.numSteps = numSteps
        self.action_space = Discrete(self.model.numCols * 3) # 0: do nothing, 1: repair, 2: replace
        self.observation_space = Box(-np.inf, np.inf, shape=(self.model.numCols * 2,), dtype=np.float32) 
        
        # Costs
        self.costReplace = -1 * costReplace
        self.costRepair = 0.5 * self.costReplace
        self.costFailure = self.model.numCols * self.costReplace

        self.failThres = driftThres

        # Reset the environment
        self.reset()

    def step(self, action):
        
        repairInds = action == 1 # the indices of the components that are being repaired
        replaceInds = action == 2 # the indices of the components that are being replaced

        self.ages[repairInds] = np.clip(self.ages[repairInds] - 2, a_min=0, a_max=None) # subtract 2 age steps for all components being repaired
        self.ages[replaceInds] = 0 # reset the ages of the replaced components
        self.shapes[replaceInds] = 0 # reset the shapes of the Gamma distribution of the damage for the replaced components

        # Bayesian inference
        # Updating the distributions of A, B
        self._bayesian_inference()

        reward = self._calc_pf_FORM() * self.costFailure + repairInds.sum() * self.costRepair + replaceInds.sum() * self.costReplace

        self.ages += 1
        self.decisionStep += 1
        
        done = True if self.decisionStep == self.numSteps else False
        info = {}

        return np.concatenate((self.shapes, self.ages)), reward, done, info, self.trace
    

    def _bayesian_inference(self, draws=4000, tune=2000, targetAccept=0.9):
        """
        Performs the Bayesian Inference, i.e. updates the distributions of parameters A, B, 
        using the damage increments that are obtained using a Gamma process
        """
        with pm.Model() as bayesianInference:

            if self.decisionStep == 0:
                A = pm.Lognormal('A', mu=self.logMuA, sigma=self.logSigmaA)
                B = pm.Normal('B', mu=self.muB, sigma=self.sigmaB)
            else:
                A = self._from_posterior('A', self.trace['A'])
                B = self._from_posterior('B', self.trace['B'])
                
            # Sample a ann b for the components' damage increment distributions
            a = A.random(size=(self.model.numCols))
            b = B.random(size=(self.model.numCols))

            self.shapes += a * (self.ages + 1) ** b - a * self.ages ** b

            # create the continuous RV which is the expected value for the model
            damages = np.random.gamma(shape=self.shapes, scale=1/self.u, size=(5, self.model.numCols))

            modalDispls = np.mean(np.abs(np.array([self.model.eigen_analysis(dams) for dams in damages])), axis=0)

            obsMus = np.abs(self.model.eigen_analysis(np.random.gamma(shape=self.shapes, scale=1/self.u)))

            noises = obsMus * self.noise
            
            eigObs = np.random.normal(loc=obsMus, scale=noises)

            obsModalDispls = pm.Normal('observed', mu=modalDispls.reshape((-1, 1)), sigma=noises.reshape((-1, 1)), observed=eigObs.reshape((-1, 1)))

            self.trace = pm.sample(init='adapt_diag', progressbar=False)

    def _from_posterior(self, param, samples):
        """
        Transforms the posterior distributions to priors for the next iteration inference
        
        Parameters:
        param (str): RV name
        samples (ndarray) : Posterior discrete values (bins)
        
        Returns:
        pymc3.model.TransformedRV : Continuous distribution to be used as prior
  
        """
        smin, smax = np.min(samples), np.max(samples)
        x = np.linspace(smin, smax, 100)
        y = stats.gaussian_kde(samples)(x)

        return pm.distributions.Interpolated(param, x, y)   

    def _calc_pf_MC(self, pfIters=5000): # Damages is an 1D numpy array containing the samples that were generated through NUTS
        damsPf = np.random.gamma(shape=self.shapes, scale=1/self.u, size=(pfIters, self.model.numCols))
        driftsPf = np.array([self.model.linear_static_analysis(ds) for ds in damsPf])

        return (driftsPf > self.failThres).sum() / pfIters
    
    def _calc_pf_FORM(self):

        pi_pdf = [ERADist('gamma', 'PAR', [self.u, sh]) for sh in self.shapes]
        pi_pdf = ERANataf(pi_pdf, self.R)
        g = lambda dams: np.abs(self.failThres - self.model.linear_static_analysis(np.array(dams)))
        pf = FORM_geom(g, pi_pdf)

        return pf

    def reset(self):
        self.decisionStep = 0
        self._new_struct()

        return np.concatenate((self.shapes, self.ages))
    
    def _new_struct(self):
        self.ages = np.zeros((self.model.numCols,))
        self.shapes = np.zeros((self.model.numCols,))

    def render(self):
        pass

    def close(self):
        pass