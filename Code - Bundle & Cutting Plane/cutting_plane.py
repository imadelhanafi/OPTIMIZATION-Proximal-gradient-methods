# -*- coding: utf-8 -*-
import numpy as np
#import modele
import fonctions


def cutting_plane(param,modele,X):
    i = 0
    ARRET = 'NON'
    LB = -1e20
    UB = +1e20
    BEST_UB = +1e20
    
    ITERATION = 1
    print("Ite      LB      UB      Best UB")
    computations = fonctions.computations(param,X,param.xMin)
    sub_gradient = computations['sub_gradients']
    nlp_obj = computations['nlp_obj']

    cut_coeff = np.c_[param.CUT_COEFF, sub_gradient]
    cut_point = np.c_[param.CUT_POINT, param.xMin]
    cut_rhs = np.append(param.CUT_RHS, nlp_obj)
    
    param.CUT_COEFF = cut_coeff
    param.CUT_POINT = cut_point
    param.CUT_RHS = cut_rhs

    while ARRET != 'OUI':
        ''' Mise à jour de param avec resolution du probleme '''
        # Mise à jour dans 'param' de ITERATIONS
        param.ITERATIONS.append(ITERATION)
        param.nIterations = ITERATION
        #print('---------------------')
        print('%.10s \t %.10s \t %.10s \t %.10s ' %(param.nIterations, LB, UB,BEST_UB))
        # Resolution du problème de minimisation
        m = modele(param)
        result = m.resultats(param)
        LB = result['alpha']
        x = result['x']
        
        while i < param.nVariables:
            if x[i] == None:
                x = param.xMin
                i = param.nVariables
            i +=1

        computations = fonctions.computations(param,X,x)
        sub_gradient = computations['sub_gradients']
        nlp_obj = computations['nlp_obj']
        
        # Mise à jour dans 'param' de : CUT_COEFF, CUT_POINT, CUT_RHS
        cut_coeff = np.c_[param.CUT_COEFF, sub_gradient]
        cut_point = np.c_[param.CUT_POINT, x]
        cut_rhs = np.append(param.CUT_RHS, nlp_obj)
        
        param.CUT_COEFF = cut_coeff
        param.CUT_POINT = cut_point
        param.CUT_RHS = cut_rhs
        ''' Fin mise à jour de param'''
        
        UB = nlp_obj
        if UB < BEST_UB:
            # Serious Step
            BEST_UB = UB
        if UB-LB <= 1e-3:
            ARRET = 'OUI'
        else:
            ITERATION += 1
    return result
            
    