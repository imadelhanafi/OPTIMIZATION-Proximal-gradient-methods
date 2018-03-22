# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:47:36 2018

@author: Hicham
"""
import pulp
import numpy as np

class Modele(object):
    def __init__(self,param):
        self.param = param
        self.nVariables = self.param.nVariables
        self.nCassures = self.param.nCassures
        self.nIterations = self.param.nIterations
        self.CUT_RHS = self.param.CUT_RHS
        self.CUT_COEFF = self.param.CUT_COEFF
        self.CUT_POINT= self.param.CUT_POINT
        self.xMin = self.param.xMin
        self.xMax = self.param.xMax
        self.name = "CUTTING_PLANE"
        self.x = np.empty((self.nVariables), dtype=object)
        self.alpha = 0 #.setInitialValue(0.)
        self.pb = None
        
    def resolution(self, type_resolution = 'Continuous', **kwargs):
        self.pb = pulp.LpProblem(self.name, pulp.LpMinimize)
        self.def_variables(type_resolution, **kwargs)
        self.def_objective()
        self.def_constraints()
        self.def_constraints_specific(type_resolution)
        self.pb.solve()
        try: 
        #self.pb.writeLP(self.name+".lp")
            assert(pulp.LpStatus[self.pb.status] == "Optimal")
        except:
            print("Status : %s" %(pulp.LpStatus[self.pb.status]))
            
    def def_objective(self):
        #Objective
        #self.pb += self.alpha
        self.pb += self.alpha
        
    def def_variables(self, type_resolution = 'Continuous'):
        #Contraintes par defaut :
        self.alpha = pulp.LpVariable("alpha")
        for i in range(self.nVariables):
            self.x[i] = pulp.LpVariable("sol_coord_%s" %i, self.xMin[i], self.xMax[i])

    def def_constraints(self):
        for ite in range(self.nIterations):
            self.pb += self.alpha >= self.CUT_RHS[ite] + (self.CUT_COEFF[:,ite]*(self.x-self.CUT_POINT[:,ite])).sum(), "cut_ite_%s" %ite

    def def_constraints_specific(self, type_resolution = 'Continuous'):
        return
         
        
    def resultats(self,type_resolution = 'Continuous', **kwargs):
        self.resolution(type_resolution, **kwargs)
        results_ = dict()
        results_['pb'] = self.pb
        results_['obj'] = pulp.value(self.pb.objective)
        results_['x'] = np.array([self.x[i].value() for i in range(self.nVariables)])
        results_['alpha'] = self.alpha.value()
        return results_

'''
if __name__=='__main__':
    import os,sys
    #dirname=os.path.join(os.getcwd(),'maquette_vx')
    dirname=os.path.join(os.getcwd(), '\\Opt non diff\\TP2\\hello')
    sys.path.append(os.path.join(dirname))
    print(os.getcwd())
    
    from param import param
    
    EPS =
    nVariables = 
    nCassures = 
    xMin = 
    xMax = 
    X =
    
    parametres = param(EPS, nVariables, nCassures, xMin, xMax, X)
    m = Model(parametres)
    r = m.resultats()
  
    print(r['obj']) 
'''    
