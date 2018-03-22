# -*- coding: utf-8 -*-

class param(object):
    def __init__(self, EPS, nVariables, nCassures, nIterations, xMin, xMax, ITERATIONS, CUT_RHS, CUT_COEFF, CUT_POINT):
        self.EPS = EPS
        self.nVariables = nVariables
        self.nCassures = nCassures
        self.xMin = xMin
        self.xMax = xMax
        self.ITERATIONS = ITERATIONS
        self.nIterations = len(self.ITERATIONS)
        self.CUT_RHS = CUT_RHS 
        self.CUT_COEFF = CUT_COEFF 
        self.CUT_POINT = CUT_POINT 