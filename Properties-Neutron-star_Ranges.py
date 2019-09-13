import math
import numpy as np
import itertools

# From Tolos FSU2H

# "Constants"
rho_0 = 0.1505 * (197.33**3)       #MeV^3
m_eff = 0.593 * 939                #MeV
m_b = 939                          #MeV
E_binding = -16.28                 #MeV

# Massa in MeV
m_sig = 497.479
m_w = 782.500
m_rho = 763.000

# All possible values of the coupling constants will be added to the lists
g_sigma_N_list = []
g_omega_N_list = []
g_rho_N_list = []

kappa_list = []
lambda_0_list = []
zeta_list = []
Lambda_w_list = []

# The equation from Chen and Piekarewicz 2014
# Where k_f = 257.73, E_f = 613.58, g_w_omega = -629.86
def Nuclear_Properties(rho):
    
    # Coupling Constants combinations
    g_sigN = np.linspace(10.0, 200, 10)
    g_wN = np.linspace(10.0, 200., 10)
    g_rhoN = np.linspace(10., 400., 10)
    
    kappa = np.linspace(0, 10.0, 5)
    lambda_0 = np.linspace(-1, 0., 10)
    zeta = np.linspace(-0.5, 1, 5)
    Lambda_w = np.linspace(0., 0.1, 5)
    
    for combination in itertools.product(g_sigN, g_wN, g_rhoN, kappa, \
        lambda_0, zeta, Lambda_w):
        
        g_s = math.sqrt(combination[0])
        g_w = math.sqrt(combination[1])
        g_r = math.sqrt(combination[2])
        k = combination[3]
        l = combination[4]
        Z = combination[5]
        L_w = combination[6]
        
        # Symmetry Energy E_sym
        k_f = (0.5*3.*rho*math.pi**2)**(1/3.)
        E_f = math.sqrt(k_f**2 + m_eff**2)
        g_w_omega = E_binding - E_f + m_b
        
        J_0 = k_f**2/(6*E_f)
        J_1 = (1/8.)*(rho*g_r**2)/(m_rho**2 + (2*L_w*(g_w_omega**2)*g_r**2))
    
        E_sym = J_0 + J_1
    
        # Slope L
        m_star_sqr_E_f_sqr = m_eff**2/E_f**2   # M*^2/E_f^2
        rho_m_star = 3*rho/m_eff               # 3rho/m_eff
        
        sigma = m_b - m_eff
        m_sig_star_sqr_g_sig_sqr = (m_sig**2/g_s**2) + \
        (k*sigma) + 0.5*l*sigma**2
        
        # Components of the rho_s_prime
        term1 = k_f/E_f
        term2 = E_f**2 + (2*m_eff**2)
        term3 = 3*m_eff**2
        term4 = np.log((k_f + E_f)/m_eff)
        
        rho_s_prime = (1/math.pi**2)*(term1*term2 - term3*term4)
        
        dM_drho = -(m_eff/E_f)*(1./(m_sig_star_sqr_g_sig_sqr+rho_s_prime))  
        
        m_w_star_sqr = m_w**2 + 0.5*Z*(g_w**2)*(g_w_omega**2)
        
        L_0 = J_0*(1 + (m_star_sqr_E_f_sqr*(1. - (rho_m_star*dM_drho))))
        L_1 = 3.*J_1*(1. - 32.*(g_w**2/m_w_star_sqr)*g_w_omega*L_w*J_1)
        
        L = L_0 + L_1
        
        # Compressibility Coefficient K
        M_Ef = m_eff / E_f
        
        dEf_drho = math.pi**2 / (2*k_f*E_f)
        dW0_drho = g_w**2 / m_w_star_sqr
        
        K = 9.*rho*(dEf_drho + dW0_drho + (M_Ef*dM_drho))
        
        # Condition that the combination of constants present the correct 
        # symmetry energy and slope
        if 30 <= E_sym <= 33 and 44 <= L <= 47:
            g_sigma_N_list.append(g_s**2)
            g_omega_N_list.append(g_w**2)
            g_rho_N_list.append(g_r**2)

            kappa_list.append(k)
            lambda_0_list.append(l)
            zeta_list.append(Z)
            Lambda_w_list.append(L_w)
           
    print min(g_sigma_N_list), max(g_sigma_N_list)
    print min(g_omega_N_list), max(g_omega_N_list)
    print min(g_rho_N_list), max(g_rho_N_list)
    print min(kappa_list), max(kappa_list)
    print min(lambda_0_list), min(lambda_0_list)
    print min(zeta_list), max(zeta_list)
    print min(Lambda_w_list), max(Lambda_w_list)
    
    return E_sym, L, K
   
Nuclear_Properties(rho_0)

