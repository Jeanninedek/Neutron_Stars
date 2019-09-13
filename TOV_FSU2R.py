import numpy
import math
from scipy.interpolate import UnivariateSpline
from scipy.constants import pi
from scipy.integrate import odeint, ode
from matplotlib import pyplot
from scipy import optimize
import matplotlib.pyplot as plt

c = 3e10            # cm/s
G = 6.67428e-8      # dyne cm^2/g^2
Msun = 1.989e33     # g

dyncm2_to_MeVfm3 = 1./(1.6022e33)
gcm3_to_MeVfm3 = 1./(1.7827e12)
oneoverfm_MeV = 197.33


def f0(x):
    return 1./(numpy.exp(x) + 1.)

def SLYfit(rho):
    a = numpy.array([6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, \
    19.105, 0.8938, 6.54, 11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, \
    -1.653, 1.50, 14.67]) ## SLy
    
    part1 = (a[0] + a[1]*rho + a[2]*rho**3.)/(1. + a[3]*rho) \
    * f0(a[4]*(rho-a[5]))
    part2 = (a[6] + a[7]*rho)*f0(a[8]*(a[9]-rho))
    part3 = (a[10] + a[11]*rho)*f0(a[12]*(a[13]-rho))
    part4 = (a[14] + a[15]*rho)*f0(a[16]*(a[17] - rho))
    return part1+part2+part3+part4

def TOV(r, y, inveos):
    pres, m = y

    eps = inveos(pres)
    
    dpdr = -(eps + pres) * (m + 4.*pi*r**3. * pres)
    dpdr = dpdr/(r*(r - 2.*m))
    dmdr = 4.*pi*r**2.0 * eps         
            
    return numpy.array([dpdr, dmdr])

def solveTOV(rhocent, eps, pres):
  
    ## Calculate the interpolated EoS ##
    eos = UnivariateSpline(eps, pres, k=1, s=0)
    inveos = UnivariateSpline(pres, eps, k=1, s=0)

    ## Minimum pressure of the EoS, at this pressure the integration \
    ## should stop 
    Pmin = pres[20]
    
    ## Starting value for r an stepsize in r ##
    r = 4.441e-16
    dr = 10.

    ## Change central density to geometrized units and calculate 
    ## central pressure 
    rhocent =  rhocent * G/c**2. 
    pcent = eos(rhocent)

    ## Calculate initial values ##
    P0 = pcent - (2.*pi/3.)*(pcent + rhocent) *(3.*pcent + rhocent)*r**2.
    m0 = 4./3. *pi *rhocent*r**3.
    stateTOV = numpy.array([P0, m0])


    ## Define the ODE solver ##
    sy = ode(TOV, None).set_integrator('lsoda')
    sy.set_initial_value(stateTOV, r).set_f_params(inveos)

    ## Run ODE solver until the pressure is less than our minimum pressure 
    while sy.successful() and stateTOV[0]>Pmin:
        stateTOV = sy.integrate(sy.t+dr)
        dpdr, dmdr = TOV(sy.t+dr, stateTOV, inveos)
        dr = 0.46 * (1./stateTOV[1] * dmdr - 1./stateTOV[0]*dpdr)**(-1.)

    ## Return the mass and the radius in units of solar mass and 
    ## km respectively 
    return stateTOV[1]*c**2./G/Msun, sy.t/1e5


# From Tolos et al. 2017 PASA 

# Only protons, neutrons and leptons

#term1 = (g_sig/m_sig)**2 
term1 = 16.92                    #fm^2

#term2 = (g_w/m_w)**2 
term2 = 11.60                    #fm^2

#term3 = (g_rho/m_rho)**2 
term3 = 13.81                    #fm^2


# Masses of mesons
m_sig = 2.521       #fm^-1
m_w = 3.96544         #fm^-1
m_rho = 3.86662       #fm^-1

# Coupling Constants
kappa = 3.0911/197.33    #fm^-1
lambda_0 = -0.001680
zeta = 0.024
Lambda_w = 0.045

#Masses of baryons and leptons in fm^-1
m_e = 2.5896 * 10**-3 
m_mu = 0.53544        
m_n = 4.75853         
m_p = 4.75853        
#m_eff = 2.8188   

J_B = 1/2.
b_B = 1     

m_l = [m_e, m_mu]
m_b = [m_p, m_n]

# Define a matrix with baryons (baryon nr., charge, I_3b, ratio g_sig, \
# ratio g_w, ratio g_rho) 
# Only proton and neutron
Matrix_b = numpy.array([[1., 1., 1/2., 1., 1., 1.],[1., 0., -1/2., 1., 1., 1.]])
Matrix_l = numpy.array([[0., -1., 1/2.],[0., -1., 1/2.]])

rho_0 = 0.1505            #fm^-3
dt = 0.1

# Initialize lists that want to be plotted
E_list = []
mu_n_list = []
mu_e_list = []
g_w_w_0 = []
g_rho_rho_03 = []
m_star = []
rho_list_plot = []

#Solve these equations to get initial values
def initial_values(rho):
    sigma = term1*rho
    rho_03 = -term3*rho/2.
    omega = rho*(((1/term2) + 2.*Lambda_w*(rho_03**2))**(-1.))
    m_eff = m_n - sigma
    mu_n = m_eff + omega + rho_03*Matrix_b[1,2]
    mu_e = 0.12*m_e*(rho/rho_0)**(2/3.)

    return sigma, omega, rho_03, mu_n, mu_e

def functie(x, rho):
    sigma, omega, rho_03, mu_n, mu_e = x
    
    # Make an array of rho_B and Q_B to be able to 
    #find their sum by adding all the terms in the array
    rho_list = []
    rho_B_list = []
    rho_SB_list = []
    q_list = []
    chem_pot = []

    for i in range(len(Matrix_b)):
        
        m_eff = m_n - sigma

        # Find the chemical potential for every baryon
        mu_b = Matrix_b[i,0]*mu_n - Matrix_b[i,1]*mu_e
        chem_pot.append(mu_b)
        
        # Use it to calculate the Fermi Energy
        E_fb = mu_b - omega - rho_03*Matrix_b[i,2]

        k_fb_sq = E_fb**2 - m_eff**2
        k_fb_sq = numpy.clip(k_fb_sq, a_min=0.0, a_max=None)
        k_fb = math.sqrt(k_fb_sq)
        
        # Denisities 
        rho_B = ((2*J_B) + 1)*b_B*k_fb**3 / (6.*math.pi**2)       
        rho_SB = (m_eff/(2.*math.pi**2))*(E_fb*k_fb - \
        (m_eff**(2))*numpy.log((E_fb + k_fb)/m_eff))
        
        rho_list.append(rho_B) 
        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)
        
        # Calculate the charge and append it to the charge list 
        Q_B = ((2.*J_B) + 1.)*Matrix_b[i,1]*k_fb**3 / (6.*math.pi**2)
        q_list.append(Q_B)
        
    for j in range(len(Matrix_l)):
        
        # Similarly for the leptons    
        mu_l = Matrix_l[j,0]*mu_n - Matrix_l[j,1]*mu_e
        E_fl = mu_l
        chem_pot.append(mu_l)
        
        k_fl_sq = E_fl**2 - m_l[j]**2
        k_fl_sq = numpy.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)
          
        # Calculate the charge and append it to the charge list  
        Q_L = ((2.*J_B) + 1.)*Matrix_l[j,1]*k_fl**3 / (6.*math.pi**2)
        q_list.append(Q_L)
        
        # Density
        rho_l = k_fl**3 / (3.*math.pi**2)
        rho_list.append(rho_l)
 
    # Sovle the equations of motion       
    f = [((sigma/term1) - sum(numpy.multiply(numpy.array(rho_SB_list), \
        Matrix_b[:,3])) + (kappa*(sigma**2)/2.) + lambda_0*(sigma**3)/6.)**2,
        
        ((omega/term2) - sum(numpy.multiply(numpy.array(rho_B_list), \
        Matrix_b[:,4])) + (zeta*(omega**3))/6.\
        + (2.*Lambda_w*omega*rho_03**2))**2,
        
        (rho_03/term3 - sum(numpy.multiply(numpy.array(rho_B_list),Matrix_b[:,2],\
        Matrix_b[:,5]))+ 2.*Lambda_w*rho_03*(omega**2))**2, 
           
        (rho - sum(rho_B_list))**2,
        
        (sum(numpy.multiply(q_list, rho_list)))**2,
        
        (chem_pot[2] + chem_pot[0] - chem_pot[1])**2]
   
    return f

# Create a function that returns the energy density
# Put as argument x the solutions to functie
def Energy_density_Pressure(x, rho):
    sigma, omega, rho_03, mu_n, mu_e = x
    
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
    
    # BARYONS
    for i in range(len(Matrix_b)):
        
        m_eff = m_n - sigma
        
        # Find the chemical potential for every baryon
        mu_b = Matrix_b[i,0]*mu_n - Matrix_b[i,1]*mu_e
        chemical_potential.append(mu_b)
        
        # Use it to calculate the Fermi Energy
        E_fb = mu_b - omega - rho_03*Matrix_b[i,2]

        k_fb_sq = E_fb**2 - m_eff**2
        k_fb_sq = numpy.clip(k_fb_sq, a_min=0.0, a_max=None)
        k_fb = math.sqrt(k_fb_sq)
        
        # Denisities 
        rho_B = ((2.*J_B) + 1.)*b_B*(k_fb**3) / (6.*math.pi**2)       
        rho_SB = (m_eff/(2.*math.pi**2))*(E_fb*k_fb \
        - (m_eff**(2))*numpy.log((E_fb + k_fb)/m_eff))
        
        rho_list.append(rho_B) 
        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)
        
        rho_list_total.append(rho_B)
        #rho_list_total.append(rho_SB)
        
        # Calculate the charge and append it to the charge list  
        Q_B = (2.*J_B + 1.)*Matrix_b[i,1]*(k_fb**3) / (6.*math.pi**2)
        q_list.append(Q_B)
        
        # Compute the energy of baryon and put it in the list
        energy_baryon = ((2.*J_B + 1.)/(2.*math.pi**2))*((1/4.)\
        *(k_fb*E_fb**3 - 0.5*k_fb*E_fb*m_eff**2 \
        - 0.5*numpy.log((k_fb + E_fb)/m_eff)*m_eff**4))
        
        energy_b.append(energy_baryon)

    # LEPTONS
    for j in range(len(Matrix_l)):

        # Similarly for the leptons    
        mu_l = Matrix_l[j,0]*mu_n - Matrix_l[j,1]*mu_e
        chemical_potential.append(mu_l)
        
        E_fl = mu_l
        
        k_fl_sq = E_fl**2 - m_l[j]**2
        k_fl_sq = numpy.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)
    
    
        # Calculate the charge and append it to the charge list  
        Q_L = (2.*J_B + 1.)*Matrix_l[j,1]*(k_fl**3) / (6.*math.pi**2)
        q_list.append(Q_L)
        
        # Density
        rho_l = k_fl**3 / (3.*math.pi**2)
        rho_list_total.append(rho_l)
        
        # Compute the energy of the lepton and put it in the list
        energy_lepton = (1./(math.pi**2))*((1/4.)\
        * (k_fl*E_fl**3 - 0.5*k_fl*E_fb*m_l[j]**2 \
        - 0.5*numpy.log((k_fl + E_fl)/m_l[j])*m_l[j]**4))
        
        energy_l.append(energy_lepton)
   
    # Split the equation up into multiple equations to make it shorter
    sigma_terms = 0.5*(sigma**2)/term1 + (kappa*(sigma**3))/6. \
    + (lambda_0*(sigma**4))/24.
    
    omega_terms = 0.5*(omega**2)/term2 + (zeta*(omega**4))/8.
    
    rho_terms = 0.5*(rho_03**2)/term3 + 3.*Lambda_w*(omega*rho_03)**2 
       
    # the term with phi is zero as the phi meson does not couple to nucleons    

    # Final equation energy density
    energy_density = sum(energy_b) + sum(energy_l) + sigma_terms \
    + omega_terms + rho_terms 
    
    Pressure = sum(numpy.multiply(numpy.array(chemical_potential), \
    numpy.array(rho_list_total))) - energy_density 

    return energy_density, Pressure

# Define lists to keep track of the initial values 
sig_initial = []
omg_initial = []
rho_03_initial = []
mu_n_initial = []
mu_e_initial = []
Energy_list = []
Pressure_list = []

sig_initial.append(initial_values(0.1*rho_0)[0])
omg_initial.append(initial_values(0.1*rho_0)[1])
rho_03_initial.append(initial_values(0.1*rho_0)[2])
mu_n_initial.append(initial_values(0.1*rho_0)[3])
mu_e_initial.append(initial_values(0.1*rho_0)[4])    

functie(initial_values(rho_0), rho_0)

for i in range(1, 100):
    rho = i*dt*rho_0
    rho_list_plot.append(rho)
    
    # Take the previous values as next initial values
    sol = optimize.root(functie, [(sig_initial[i-1], omg_initial[i-1],\
    rho_03_initial[i-1], mu_n_initial[i-1], mu_e_initial[i-1])], \
    method='lm', args=(rho))

    Energy_list.append(Energy_density_Pressure(sol.x, rho)[0]\
    *197.33*1.7827*10**12) 
    Pressure_list.append(Energy_density_Pressure(sol.x, rho)[1]\
    *197.33*1.6022*10**33)     
    
    # Put the new solutions in the list so that they can be taken as 
    # initial values for the next step
    sig_initial.append(sol.x[0])
    omg_initial.append(sol.x[1])
    rho_03_initial.append(sol.x[2])
    mu_n_initial.append(sol.x[3])
    mu_e_initial.append(sol.x[4])                 


## Calculate mass and radius for a central density of 1e15 g/cm3 ##
eps = numpy.multiply(Energy_list[7::], G/c**2)
pres = abs(numpy.multiply(Pressure_list[7::], G/c**4))

# Defining the EoS, here I use SLy EoS, in geometrized units ##
eps_crust = numpy.logspace(6.5, 14.3, 1000)*G/c**2.
pres_crust = 10**SLYfit(numpy.log10(eps_crust*c**2./G))*G/c**4.

eps_total = numpy.append(eps_crust, eps)
pres_total = numpy.append(pres_crust, pres)

def RM_Tolos():
    f = open('1708.08681')
    
    # separate the file entries 
    a = f.read().split('\n')
    Radius_FSU2R = []
    Radius_FSU2H = []
    Mass_FSU2R = []
    Mass_FSU2H = []
    
    for index, i in enumerate(a):
        # Skip the entries that do not have values in them
        if i == '':
            continue
        
        # Separate the entries and add them to their lists
        temp = i.split('&')
        Radius_FSU2R.append(float(temp[3].strip(' ')))
        Radius_FSU2H.append(float(temp[7].strip(' ')))
        Mass_FSU2R.append(float(temp[4].strip(' ')))
        Mass_FSU2H.append(float(temp[8].strip(' ')))
        
    return Radius_FSU2R, Radius_FSU2H, Mass_FSU2R, Mass_FSU2H
    
Radius_FSU2R = RM_Tolos()[0]
Radius_FSU2H = RM_Tolos()[1]
Mass_FSU2R = RM_Tolos()[2]
Mass_FSU2H = RM_Tolos()[3]


RFSU2R = []
MFSU2R = []

##for density in numpy.arange(1e13, 1e16, 1e13):
for density in numpy.logspace(14.3, 15.6, 30):
    
    RFSU2R.append(solveTOV(density, eps_total, pres_total)[1])
    MFSU2R.append(solveTOV(density, eps_total, pres_total)[0])
    
plt.scatter(RFSU2R, MFSU2R, label='FSU2R')
plt.plot(Radius_FSU2R, Mass_FSU2R, label='FSU2R Tolos')
plt.plot(Radius_FSU2H, Mass_FSU2H, label='FSU2H Tolos')    
plt.axis([9, 20, 0, 2.5])
plt.ylabel(r'M/$M_{Sun}$')
plt.xlabel('R [km]')
plt.legend()
plt.show()