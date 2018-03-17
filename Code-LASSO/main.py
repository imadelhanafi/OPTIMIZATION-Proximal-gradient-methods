# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from lambda_effect import Lambda_effect, Lambda_effect_accelerated
from lasso import lasso
from lasso_accelerated import lasso_accelerated

'''
INPUTS : 
    - Nom_algo : 'LASSO' ou 'FISTA'
    - Tolerance : erreur = 1e-4
    - Dimensions : (n,p) observations & labels
    - Données : A, b, u_0, v_0
    - Constante de régularisation : Lambda 
    - Intervalle de variation de Lambda : Lambda_min, Lambda_max
RESULTATS :
    {
    'Err_hist' : valeurs de l'erreur
    'Sol_opt' : solution optimale finale
    'nb_iterations' : nombre d'itérations
    }
ETUDE Lambda : variation de Lambda
    - Non_zero : nombre d'elements non nuls dans la solution
    - Solutions : solutions optimales du problème pour chaque lambda
EVOLUTION DES NORMES:
    - Impact de Lambda sur la norme1 et sur la norme2
ANALYSE Convergence :
    - Taux de convergence
'''


#### Inputs : 

# Choix de l'algorithme
np.random.seed(42) 

Nom_algo = 'LASSO'
#Nom_algo = 'FISTA'
print("Resolution avec "+Nom_algo)

# Tolerance

erreur = 0.0001

#Dimensions

n = 20 # Dimension des observations
p = 20 # Dimension des labels

# Matrice A et vecteur b

A = np.random.rand(n,p)
b = np.random.rand(n)

# Point initial
u_0 = np.random.rand(p)
# Pour FISTA:
v_0 = u_0

# constante de régularisation
Lambda = 0.1

# Input pour l'impact de Lambda

Lambda_min = 0
Lambda_max = 20
Nb_Lambda = 10

'''
RESOLUTION DU PROBLEME
'''

if(Nom_algo == 'LASSO'):
    Resultats = lasso(A,b,Lambda, u_0,erreur)
if(Nom_algo == 'FISTA'):    
    Resultats = lasso_accelerated(A,b,Lambda, u_0,v_0,erreur)

#Plot resultats

Erreur_histo = Resultats['Err_hist']
Solution = Resultats['Sol_opt']
Nb_iteration = Resultats['nb_iterations']
print("Solution trouvée :", Solution)
print("Nb_iteration :", Nb_iteration)

plt.figure(figsize=(15,7))
plt.plot(Erreur_histo)
plt.title(" Evolution de l'erreur ")
plt.xlabel('Iterations')
plt.ylabel('Erreur : || U(n) - U(n-1) || / || U(n-1) ||')
plt.grid(True)
plt.savefig("erreur"+Nom_algo+".png")


'''
IMPACT DE LA CONSTANTE LAMBDA
'''


# Calcul de la norme 0 -- Eléments non nuls
if(Nom_algo == 'LASSO'):
    Non_zero,Solutions = Lambda_effect(A,b,u_0,erreur,Lambda_min,Lambda_max,Nb_Lambda)
if(Nom_algo == 'FISTA'):    
    Non_zero,Solutions = Lambda_effect_accelerated(A,b,u_0,u_0,erreur,Lambda_min,Lambda_max,Nb_Lambda)

#Plots

Lambda_i = np.arange(Lambda_min,Lambda_max,(Lambda_max - Lambda_min)/Nb_Lambda)
plt.figure(figsize=(15,7))
plt.plot(Lambda_i,Non_zero)
plt.title(" Evolution de la sparsité de la solution de dimension %d " %(u_0.shape[0]))
plt.xlabel('Lambda')
plt.ylabel("Nombre d'éléments non nuls")
plt.grid(True)
plt.savefig("sparsity"+Nom_algo+".png")


inter = [A.dot(Solutions[i])-b for i in range(Nb_Lambda)]
Norme2  = [np.linalg.norm(inter[i]) for i in range(Nb_Lambda)]

Norme2_carre = 0.5*np.multiply(Norme2,Norme2)
Norme1 = [np.linalg.norm(Solutions[i],1) for i in range(Nb_Lambda)]

plt.figure(figsize=(15,7))
plt.plot(Lambda_i,Norme2_carre,label="Norme2")
plt.plot(Lambda_i,Norme1,label="Norme1")
plt.title(" Evolution des normes L1 et L2 ")
plt.xlabel('Lambda')
plt.ylabel("--")
plt.legend()
plt.grid(True)
plt.savefig("norme"+Nom_algo+".png")


'''
TAUX DE CONVERGENCE
'''

#Impact de la constante Lambda

# LASSO : iter^1  (Thm15)
# FISTA : iter^2 (page 23)
v = np.array(Resultats['Diff_hist'])
iter = np.arange(len(v))
if(Nom_algo == 'LASSO'):
    taux = (iter)*v
if(Nom_algo == 'FISTA'):    
    taux = (iter**2)*v

plt.figure(figsize=(15,7))
plt.plot(taux)
plt.title("Taux de convergence")
plt.xlabel('Itérations')
plt.ylabel("Taux")
plt.grid(True)
plt.savefig("taux"+Nom_algo+".png")


'''
VITESSE DE CONVERGENCE
'''

plt.figure(figsize=(15,7))
v = np.array(Resultats['Diff_hist'])
vitesse = v[1:len(v)]/((v[0:len(v)-1]))
plt.figure(figsize=(15,7))
plt.plot(vitesse)
plt.title("Vitesse de convergence")
plt.xlabel('Itérations')
plt.ylabel("Vitesse")
plt.grid(True)
plt.savefig("speed"+Nom_algo+".png")

# Convergence lineaire - Lasso