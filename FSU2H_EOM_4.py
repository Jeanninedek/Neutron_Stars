import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import optimize
import pandas

# From Tolos et al. 2017 PASA 
# Protons, neutrons, hyperons and leptons

# Read the values from Table 7 from Sharma et al. 2015 (the inner 
# crust used by Tolos)
def Inner_Crust_Sharma():

    f = open('Sharma_inner_crust.txt')
    
    # separate the file entries 
    a = f.read().split('\n')

    P = []
    epsilon = []
    
    for index, i in enumerate(a):
        # Skip the entries that do not have values in them
        if i == '':
            continue
        
        # Separate the entries and add them to their lists
        temp = i.split('&')
        P.append(float(temp[3].strip(' ')))
        epsilon.append(float(temp[2].strip(' ')))
        
    return epsilon, P
    
Epsilon_crust = Inner_Crust_Sharma()[0]
Pressure_crust = Inner_Crust_Sharma()[1]


# This function reads the graph of Pressure vs. Number density from 
# Tolos 2017 paper
def Graph_Tolos():
    data = open('Graph_Tolos.csv')
    
    a = data.read().split('\n')
    #data = pandas.read_csv('Graph_Tolos.csv')
    
    number_density = []
    Pressure = []

    for index, i in enumerate(a):
        # Skip the entries that do not have values in them
        if i == '':
            continue
        
        # Separate the entries and add them to their lists
        temp = i.split(',')
        number_density.append(float(temp[0].strip(' ')))
        Pressure.append(float(temp[1].strip(' ')))
        
    return number_density, Pressure

# Use these values to compare the Pressure graphs    
n = Graph_Tolos()[0]
Pressure = Graph_Tolos()[1]

# This function reads the table in the Tolos paper and returns the 
# Pressure, energy density, and number density
# This is to compare it with what my code gives
def table_Tolos():
    f = open('1708_3.txt')
    
    # separate the file entries 
    a = f.read().split('\n')

    P = []
    epsilon = []
    
    for index, i in enumerate(a):
        # Skip the entries that do not have values in them
        if i == '':
            continue
        
        # Separate the entries and add them to their lists
        temp = i.split('&')
        P.append(float(temp[5].strip(' ')))
        epsilon.append(float(temp[6].strip(' ')))
        
    return epsilon, P
    
Epsilon_Tolos = table_Tolos()[0]
Pressure_Tolos = table_Tolos()[1] 

Eps_T = np.array(Epsilon_Tolos)*1.7827e12
Pres_T = np.array(Pressure_Tolos)*1.6022e33               

# Add all the lists from the various areas together to be able to plot
Total_ED = Epsilon_crust + Eps_T.tolist()
Total_P = Pressure_crust +  Pres_T.tolist()


# Masses of mesons
m_sig = 2.521         #fm^-1
m_w = 3.96544         #fm^-1
m_rho = 3.86662       #fm^-1
m_phi = 5.1662        #fm^-1

#Masses of baryons and leptons in fm^-1
m_e = 2.5896 * 10**-3 
m_mu = 0.53544        
m_n = 4.75853         
m_p = 4.75853        
#m_eff = 2.8188   

# 
J_B = 1/2.
b_B = 1.   

m_l = [m_e, m_mu]
# List of baryon masses (nucleon, Lambda, Sigma, Xi)
m_b = np.array([4.75853, 4.75853, 1115/197.33, 1190/197.33, 1190/197.33,\
1190/197.33, 1315/197.33, 1315/197.33])


# adjust ratio's to fit baryon type 
# Define a matrix with baryons (baryon nr., charge, I_3b, ratio g_sig, 
# ratio g_w, ratio g_rho, ratio g_phi)    
# Proton, neutron, Lambda0, Sigma+, Sigma0, Sigma-, Xi0, Xi-                                                                     
Matrix_b = np.array([
                    [1., 1., 1/2., 1., 1., 1., 0.],\
                    [1., 0., -1/2., 1., 1., 1., 0.],\
                    [1., 0., 0., 0.611, 2/3., 0., -0.471],\
                    [1., 1., 1., 0.467, 2/3., 1., -0.471],\
                    [1., 0., 0., 0.467, 2/3., 1., -0.471],\
                    [1., -1., -1., 0.467, 2/3., 1., -0.471],\
                    [1., 0., 1/2., 0.290, 1/3., 1., -0.943],\
                    [1., -1., -1/2., 0.290, 1/3., 1., -0.943]])

Matrix_l = np.array([[0., -1., 1/2.],[0., -1., 1/2.]])

rho_0 = 0.1505            #fm^-3
dt = 0.1

# FSU2H coupling constants

# Coupling Constants
kappa = 4.0014/197.33    #fm^-1
lambda_0 = -0.013298
zeta = 0.008
Lambda_w = 0.045

# Order nucleon, Lambda, Sigma, Xi
g_phi_list = np.array([0, 0, -6.1379, -6.1379, -6.1379, -6.1379, \
-12.2757, -12.2757])

g_omega_list = np.array([13.0204, 13.0204, 8.6803, 8.6803, 8.6803, \
8.6803, 4.3401, 4.3401])

g_rho_list = np.array([14.0453, 14.0453, 0, 14.0453, 14.0453, 14.0453,\
14.0453, 14.0453])

g_sigma_list = np.array([math.sqrt(102.7200), math.sqrt(102.7200), \
0.611*math.sqrt(102.7200), 0.467*math.sqrt(102.7200),\
0.467*math.sqrt(102.7200), 0.467*math.sqrt(102.7200), \
0.271*math.sqrt(102.7200), 0.271*math.sqrt(102.7200)])

# Initialize lists that want to be plotted
E_list = []
mu_n_list = []
mu_e_list = []
g_w_w_0 = []
g_rho_rho_03 = []
m_star = []
rho_list_plot = []
        
def initial_values(rho):
    
    sigma = g_sigma_list[0]*rho/(m_sig**2)
    rho_03 = - g_rho_list[0]*rho/(2.*(m_rho**2))
    omega = rho*((((m_w**2)/g_omega_list[0]) + \
    (2.*Lambda_w*((g_rho_list[0]*rho_03)**2)*g_omega_list[0]))**(-1.))
    phi = g_phi_list[0]*rho/(m_phi**2)
    m_eff = m_n - (g_sigma_list[0]*sigma) 
    mu_n = m_eff + (g_omega_list[0]*omega) + \
    (g_rho_list[0]*rho_03*Matrix_b[1,2]) + (g_phi_list[0]*phi) 
    mu_e = 0.12*m_e*(rho/rho_0)**(2/3.)

    return sigma, omega, rho_03, phi, mu_n, mu_e
  
def functie(x, rho):
    
    sigma, omega, rho_03, phi, mu_n, mu_e = x

    # Make an array of rho_B and Q_B to be able to 
    #find their sum by adding all the terms in the array
    rho_list = []
    rho_B_list = []
    rho_SB_list = []
    q_list = []
    chem_pot_b = []
    chem_pot_l = []
    
    for i in range(len(Matrix_b)):
        
        m_eff = m_b[i] - (g_sigma_list[i]*sigma) 
    
        # Find the chemical potential for every baryon
        mu_b = Matrix_b[i,0]*mu_n - Matrix_b[i,1]*mu_e              
        chem_pot_b.append(mu_b)
        
        # Use it to calculate the Fermi Energy
        E_fb = mu_b - g_omega_list[i]*omega - \
        g_rho_list[i]*rho_03*Matrix_b[i,2] - phi*g_phi_list[i] 
        
        k_fb_sq = (E_fb**2) - (m_eff**2)
        
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None) 
            E_fb = m_eff 
            
        k_fb = math.sqrt(k_fb_sq)                           
        
        # Denisities 
        rho_B = ((2.*J_B) + 1)*b_B*(k_fb**3) / (6.*math.pi**2)    
        rho_SB = (m_eff/(2.*(math.pi**2)))*(E_fb*k_fb - \
        (m_eff**(2))*np.log((E_fb + k_fb)/m_eff))              

        rho_list.append(rho_B) 
        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)                  
        
        # Calculate the charge and append it to the charge list 
        Q_B = ((2.*J_B) + 1.)*Matrix_b[i,1]*(k_fb**3) / (6.*(math.pi**2))
        q_list.append(Q_B)                        
        
    for j in range(len(Matrix_l)):
        
        # Similarly for the leptons    
        mu_l = Matrix_l[j,0]*mu_n - Matrix_l[j,1]*mu_e              
        E_fl = mu_l
        chem_pot_l.append(mu_l)
        
        k_fl_sq = (E_fl**2) - (m_l[j]**2)
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)                                  
          
        # Calculate the charge and append it to the charge list  
        Q_L = ((2.*J_B) + 1.)*Matrix_l[j,1]*(k_fl**3) / (6.*(math.pi**2))
        q_list.append(Q_L)
        
        # Density
        rho_l = (k_fl**3) / (3.*(math.pi**2))
        rho_list.append(rho_l)  
                                   
    
    f = [((sigma*(m_sig**2))/g_sigma_list[0] + \
        (kappa*(g_sigma_list[0]*sigma)**2)/2. 
        + (lambda_0*(g_sigma_list[0]*sigma)**3)/6. \
        - sum(np.array(rho_SB_list)*Matrix_b[:,3]))**2,
        
        ((omega*m_w**2)/g_omega_list[0] + \
        (zeta*(g_omega_list[0]*omega)**3)/6. \
        + (2.*Lambda_w*g_omega_list[0]*omega*(rho_03*g_rho_list[0])**2)\
        - sum(np.array(rho_B_list)*Matrix_b[:,4]))**2,
        
        ((rho_03*m_rho**2)/g_rho_list[0] \
        + (2.*Lambda_w*g_rho_list[0]*rho_03*(omega*g_omega_list[0])**2) \
        - sum(np.array(rho_B_list)*Matrix_b[:,5]*Matrix_b[:,2]))**2,
        
        ((phi*m_phi**2)/g_omega_list[0] - \
        sum(np.array(rho_B_list)*Matrix_b[:,6]))**2,            
        
        (rho - sum(rho_B_list))**2,
        
        (sum(q_list))**2]
        
    
    return f
    
def Energy_density_Pressure(x, rho):
    sigma, omega, rho_03, phi, mu_n, mu_e = x
    
    # Make an array of rho_B and Q_B to be able to find their sum by 
    # adding all the terms in the array
    rho_list = []
    rho_B_list = []
    rho_SB_list = []
    q_list = []
    energy_b = []
    energy_l = []
    chemical_potential = []
    rho_list_total = []
    
    sigma_term_list = []
    omega_term_list = []
    rho_term_list = []
    phi_term_list = []
    
    for i in range(len(Matrix_b)):
        
        m_eff = m_b[i] - g_sigma_list[i]*sigma
        
        # Find the chemical potential for every baryon
        mu_b = Matrix_b[i,0]*mu_n - Matrix_b[i,1]*mu_e
        chemical_potential.append(mu_b)
        
        # Use it to calculate the Fermi Energy
        E_fb = mu_b - g_omega_list[i]*omega - \
        g_rho_list[i]*rho_03*Matrix_b[i,2]\
        - phi*g_phi_list[i]
        
        k_fb_sq = (E_fb**2) - (m_eff**2)
        
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None) 
            E_fb = m_eff 
            
        k_fb = math.sqrt(k_fb_sq)
        
        # Denisities 
        rho_B = ((2.*J_B) + 1)*b_B*k_fb**3 / (6.*math.pi**2)       
        rho_SB = (m_eff/(2.*(math.pi**2)))*(E_fb*k_fb - \
        (m_eff**(2))*np.log((E_fb + k_fb)/m_eff))

        rho_list.append(rho_B) 
        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)
        rho_list_total.append(rho_B)
        
        # Calculate the charge and append it to the charge list 
        Q_B = ((2.*J_B) + 1.)*Matrix_b[i,1]*(k_fb**3) / (6.*math.pi**2)
        q_list.append(Q_B)
        
        # Compute the energy of baryon and put it in the list
        energy_baryon = (1/(8.*(math.pi**2)))*(k_fb*(E_fb**3) \
        + E_fb*(k_fb**3) - np.log((E_fb + k_fb)/m_eff)*m_eff**4)
        energy_b.append(energy_baryon)
    
        sigma_term_list.append(g_sigma_list[i]*sigma) 
        omega_term_list.append(g_omega_list[i]*omega)
        rho_term_list.append(g_rho_list[i]*rho_03)
        phi_term_list.append(g_phi_list[i]*phi)
        
    for j in range(len(Matrix_l)):
        
        # Similarly for the leptons    
        mu_l = Matrix_l[j,0]*mu_n - Matrix_l[j,1]*mu_e
        E_fl = mu_l
        chemical_potential.append(mu_l)
        
        k_fl_sq = E_fl**2 - m_l[j]**2
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)
          
        # Calculate the charge and append it to the charge list  
        Q_L = ((2.*J_B) + 1.)*Matrix_l[j,1]*k_fl**3 / (6.*math.pi**2)
        q_list.append(Q_L)
        
        # Density
        rho_l = k_fl**3 / (3.*math.pi**2)
        rho_list.append(rho_l)
        rho_list_total.append(rho_l)
   
        # Compute the energy of the lepton and put it in the list
        energy_lepton = (1/(8.*(math.pi**2)))*(k_fl*(E_fl**3) \
        + E_fl*(k_fl**3) - (m_l[j]**4)*np.log((k_fl + E_fl) / m_l[j]))
        energy_l.append(energy_lepton)
    
    sigma_terms = 0.5*((m_sig*sigma)**2) \
    + (kappa*((g_sigma_list[0]*sigma)**3))/6. \
    + (lambda_0*((g_sigma_list[0]*sigma)**4))/24.
    
    omega_terms = 0.5*((m_w*omega)**2) \
    + (zeta*((g_omega_list[0]*omega)**4))/8.
    
    rho_terms = 0.5*((m_rho*rho_03)**2) \
    + (3.*Lambda_w*((g_rho_list[0]*rho_03*g_omega_list[0]*omega)**2))
    
    phi_terms = 0.5*((m_phi*phi)**2)
    
    # Final equation for the energy density
        
    energy_density = sum(energy_b) + sum(energy_l) + sigma_terms \
    + omega_terms + rho_terms + phi_terms
    
    Pressure = sum(np.multiply(np.array(chemical_potential), \
    np.array(rho_list_total))) - energy_density 

    return energy_density, Pressure                 
                                                   
# Define lists to keep track of the initial values 
sig_initial = []
omg_initial = []
rho_03_initial = []
phi_initial = []
mu_n_initial = []
mu_e_initial = []
Energy_list = []
Pressure_list = []
    
sig_initial.append(initial_values(0.1*rho_0)[0])
omg_initial.append(initial_values(0.1*rho_0)[1])
rho_03_initial.append(initial_values(0.1*rho_0)[2])
phi_initial.append(initial_values(0.1*rho_0)[3])
mu_n_initial.append(initial_values(0.1*rho_0)[4])
mu_e_initial.append(initial_values(0.1*rho_0)[5])  
    
    
for i in range(1, 100):
    rho = i*dt*rho_0
    rho_list_plot.append(rho)
        
    # Take the previous values as next initial values
    sol = optimize.root(functie, [(sig_initial[i-1], omg_initial[i-1], \
    rho_03_initial[i-1], phi_initial[i-1], mu_n_initial[i-1], \
    mu_e_initial[i-1])], method='lm', args=(rho))   
        
    Energy_list.append(Energy_density_Pressure(sol.x, rho)[0]*197.33*1.7827*10**12) 
    Pressure_list.append(Energy_density_Pressure(sol.x, rho)[1]*197.33) 
        
    # Put the new solutions in the list so that they can be taken 
    # as initial values for the next step
    sig_initial.append(sol.x[0])
    omg_initial.append(sol.x[1])
    rho_03_initial.append(sol.x[2])
    phi_initial.append(sol.x[3])
    mu_n_initial.append(sol.x[4])
    mu_e_initial.append(sol.x[5])
        
    m_star.append((m_n - g_sigma_list[0]*sol.x[0])*197.33)
    g_w_w_0.append(g_omega_list[0]*sol.x[1]*197.33)
    g_rho_rho_03.append(-g_rho_list[0]*sol.x[2]*197.33)
    mu_n_list.append((sol.x[4] - m_n)*197.33)
    mu_e_list.append(sol.x[5]*197.33)    #Multiply by 197.33 to get MeV
    

# Figure  1
plt.figure()
plt.plot(rho_list_plot, Pressure_list, label='This paper')
plt.plot(n, Pressure, label='Tolos')

plt.yscale('log')
plt.axis([0, 0.9, 0.5, 500])
plt.yticks([1, 10, 100, 500], [1, 10, 100, 500])

plt.xlabel(r'Baryon density (fm$^{-3}$)')
plt.ylabel(r'Pressure (MeV fm$^{-3}$)') 

plt.legend(loc='4')
  
# Figure 2    
plt.figure()
plt.plot(rho_list_plot, m_star, label=r'$m^{*}$')
plt.plot(rho_list_plot, g_w_w_0, label=r'$g_{w}w_{0}$') 
plt.plot(rho_list_plot, g_rho_rho_03, label=r'$-g_{\rho}\rho_{03}$')
plt.plot(rho_list_plot, mu_n_list, '--', label=r'$\mu_{n}-m_{n}$')
plt.plot(rho_list_plot, mu_e_list, label=r'$\mu_{e}$')
plt.axis([0, 1.5, 15, 1000])

plt.yscale('log')

plt.xlabel(r'Baryon Density (fm$^{-3}$)')
plt.ylabel('Energy (MeV)')

plt.legend(loc='4')

#Figure 3
plt.figure()
plt.plot(Energy_list, np.multiply(np.array(Pressure_list),1.6022*10**33), \
label='n+p+leptons+Hyperons') 
plt.plot(Total_ED, Total_P, label='Tolos and Sharma')

plt.yscale('log')
plt.xscale('log')

plt.ylabel(r'P (dyne/cm$^{2}$)')
plt.xlabel(r'$\epsilon$ (g/cm$^{3}$)')
plt.axis([1e13, 1e16, 1e31, 1e36])

plt.legend(loc='4')

# range of Energies that are similar to that of Tolos
Energy_list[5:65], np.array(Epsilon_Tolos)*1.7827*10**12
np.subtract(Pressure_Tolos, Pressure_list[5:65])


# Figure 4
# Plot the change in pressure in the energy range that is similar 
# to that of Tolos
plt.figure()
plt.plot(Energy_list[5:65], \
(1 - np.subtract(Pressure_Tolos, Pressure_list[5:65]))*1.6022*10**33)

plt.ylabel(r'$\Delta$P (dyne/cm$^{2}$)')
plt.xlabel(r'$\epsilon$ (g/cm$^{3}$)')

plt.show()
    