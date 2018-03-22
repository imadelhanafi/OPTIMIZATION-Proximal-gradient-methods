# -*- coding: utf-8 -*-

import numpy as np
from param import param
from fonctions import X_generator
from modele import Modele
from cutting_plane import cutting_plane
import random 

random.seed(111)


EPS = 1e-6
nVariables = 10
nCassures = 60
xMin = -10*np.arange(nVariables)-10
xMax = +10*np.arange(nVariables)+10
ITERATIONS = []
nIterations = len(ITERATIONS)
CUT_RHS = np.empty(len(ITERATIONS))
CUT_COEFF = np.empty((nVariables, nIterations), dtype = object)
CUT_POINT = np.empty((nVariables, nIterations), dtype = object)

parametres = param(EPS, nVariables, nCassures, nIterations, xMin, xMax, ITERATIONS, CUT_RHS, CUT_COEFF, CUT_POINT)
X = X_generator(parametres)
r = Modele(parametres).resultats()
m = Modele

final_result = cutting_plane(parametres, m, X)

#Print solution

print("x* = %s " %final_result['x'])

#Check : final_result['pb'].status : optimal = 1
        

