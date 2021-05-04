import numpy as np
from scipy import optimize
from astropy import constants as const
from astropy import units as u
from numba import jit

import importlib
th = importlib.import_module('kilonova_heating_rate.thermalization', __name__)
bt = importlib.import_module('kilonova_heating_rate.bateman', __name__)
root_finder = importlib.import_module('kilonova_heating_rate.root_finder', __name__)

from importlib import resources
with resources.open_text(__package__, 'tables_reshaped.dat') as f:
    fchains_reshaped = np.loadtxt(f)  # per line width of 49 becomes 7 by 7
    fchains = fchains_reshaped.reshape(fchains_reshaped.shape[0],7,7)
    Zchains = fchains[:,:,1].astype(int)
del fchains_reshaped
with resources.open_text(__package__, 'tables_length.dat') as f:
    fchains_length = np.loadtxt(f).astype(int)


day = 86400.
MeV = (1.0e6*u.eV).to(u.g*u.cm**2/u.s**2).value
mu = const.u.cgs.value
c = const.c.cgs.value

@jit(nopython=True) 
def calc_heating_rate(Mej, vmin, vmax, Amin, Amax, ffraction, kappa_effs, n):
    
    
    # Temporary workaround. In order to clean this up also clean up thermalization.py
    vej = vmin
    alpha_max = vmax/vmin
    alpha_min = 1

    t_initial = 0.01*day
    t_final = 3.*day #10.
    delta_t = 0.3#0.3

    Nth = 40#40 for default

    
    Amax_beta = 209
    
    Xtot = 0.0

    fraction = np.zeros(300)
    for i in range(0,len(ffraction)):
        A = ffraction[i,1]
        fraction[int(A)] = float(A)*ffraction[i,2]
    tmpA = 0.0
    for A in range(Amin,Amax+1):
        Xtot+=fraction[A]
        tmpA += float(A)*fraction[A]

    Aave = tmpA/Xtot



###    




    total_heats = []
    total_gammas = []
    total_elects = []
    total_elect_ths = []
    total_gamma_ths = []
    heating_functions = []
    ts = []

    t = t_initial
    while t<t_final:
        total_heats.append(0.)
        total_gammas.append(0.)
        total_elects.append(0.)
        total_elect_ths.append(0.)
        total_gamma_ths.append(0.)
        heating_functions.append(0.)
        ts.append(t)
        t*= 1. + delta_t

    #print 'total time step = ', len(total_heats)
    for A in range(Amin,min(Amax,Amax_beta)+1): 
        each_heats = np.zeros(len(ts))
        each_gammas = np.zeros(len(ts))
        each_gamma_ths = np.zeros(len(ts))
        each_elects = np.zeros(len(ts))
        each_elect_ths = np.zeros(len(ts))
        Xfraction = fraction[A]/Xtot

    
#A, Z, Q[MeV], Egamma[MeV], Eelec[MeV], Eneutrino[MeV], tau[s]
        fchain = fchains[A]

        tmp = []
        N = fchains_length[A]
        for i in range(0,N):
            tmp.append(1.0/fchain[i,6])
        lambdas = np.array(tmp)
####determine the thermalization time in units of day for each element
        tes = np.zeros(N)
        total_numb = np.zeros(N)
        for i in range(0,N):
            Z = fchain[i,1]
            Egamma = fchain[i,3]
            Eele = fchain[i,4]
       
            if(Eele>0.):
                tes[i] = th.calc_thermalization_time(Eele,Mej,vej,Aave,alpha_max,alpha_min,n)


    #ts = []
        tmp_numb = np.zeros(N)


        lambda_sort = np.zeros((N,N))
        coeffs = np.ones(N)
        xs = np.zeros((N,N))

        lambda_sort = np.zeros((N,N))

        for i in range(0,N):
            tmp0 = lambdas[:i+1]
            tmp = np.sort(tmp0)
    
            lambda_sort[i][:i+1] = tmp[::-1]

        for j in range(1,N):
            for i in range(0,j):
                coeffs[j] *= lambda_sort[j-1][i]
   # print j,coeffs[j]

        for j in range(1,N):
            for i in range(0,j):
                xs[j][i] = (lambda_sort[j][j-1-i]-lambda_sort[j][j])
    
    
        for k in range(0,len(ts)):
   # while t<t_final:
            t = ts[k]
        
            for i in range(0,N):
                coeff = coeffs[i]*np.power(t,i)
        
                if(i==6):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_6(xs[6][0]*t,xs[6][1]*t,xs[6][2]*t,xs[6][3]*t,xs[6][4]*t,xs[6][5]*t)

                elif(i==5):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_5(xs[5][0]*t,xs[5][1]*t,xs[5][2]*t,xs[5][3]*t,xs[5][4]*t)
          
                elif(i==4):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_4(xs[4][0]*t,xs[4][1]*t,xs[4][2]*t,xs[4][3]*t)
          #  print i,coeff
                elif(i==3):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_3(xs[3][0]*t,xs[3][1]*t,xs[3][2]*t)
          #  print i,xs[3][0],xs[3][1],xs[3][2]
                elif(i==2):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_2(xs[2][0]*t,xs[2][1]*t)
          #  print i,xs[2][0],xs[2][1]
                elif(i==1):
                    tmp_numb[i] = coeff*np.exp(-t*lambda_sort[i][i])*bt.calc_M0_1(xs[1][0]*t)
          #  print i,xs[1][0]
                elif(i==0):
                    tmp_numb[i] = np.exp(-t*lambda_sort[i][i])
#                else:
                  #  print 'chain is too long'
           # print i,lambda_sort[i][i]
            
        
#        number10s.append(tmp_numb[0])
#        number11s.append(tmp_numb[1])
#        number12s.append(tmp_numb[2])
#        number13s.append(tmp_numb[3])
    
            heat = 0.0
            gam = 0.0
            ele = 0.0
            ele_th = 0.0
            gam_th = 0.0
    
            for i in range(0,N):
                
                Eele = fchain[i,4]
                if(t > 0.003*tes[i]):
                    if(Eele > 0.):
                        tau1 = t/tes[i]
                        if(tau1<2.):
                            tau0 = 0.03*tau1 #0.05*tau1
                            root = root_finder.find_root(th.calc_zero_energy, tau0, args=(tau1,Eele,))
                            tau0 = root
                        else:
                            tau0 = 0.4#0.05*tau1
                #else:
                #    tau0 = 0.01*tau1
                #tau0 = 1.
                
                #print "tau0: ", tau1,tau0
                        delta_t = (tau1-tau0)*tes[i]
            #fth = fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
           # print t,delta_t,delta_t/t,fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
                        t_th = tau0*tes[i]
                        tmp_n_t_th = 1./float(Nth-1)
                        dt_th = np.power(tau1/tau0,tmp_n_t_th)-1.
                        total_numb[i] = 0.0
                        for j in range(0,Nth):
                            coeff = coeffs[i]*np.power(t_th,i)
                            tau = t_th/tes[i]
                    #e_delay= calc_e_tau_tau0(tau,tau1)
                            e_delay = th.epsilon_tau(tau,tau1,Eele)

                            if(i==6):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_6(xs[6][0]*t_th,xs[6][1]*t_th,xs[6][2]*t_th,xs[6][3]*t_th,xs[6][4]*t_th,xs[6][5]*t_th)

                            elif(i==5):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_5(xs[5][0]*t_th,xs[5][1]*t_th,xs[5][2]*t_th,xs[5][3]*t_th,xs[5][4]*t_th)
          
                            elif(i==4):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_4(xs[4][0]*t_th,xs[4][1]*t_th,xs[4][2]*t_th,xs[4][3]*t_th)
                            elif(i==3):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_3(xs[3][0]*t_th,xs[3][1]*t_th,xs[3][2]*t_th)
                            elif(i==2):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_2(xs[2][0]*t_th,xs[2][1]*t_th)
                            elif(i==1):
                                tmp_numb[i] = e_delay*coeff*np.exp(-t_th*lambda_sort[i][i])*bt.calc_M0_1(xs[1][0]*t_th)
                            elif(i==0):
                                tmp_numb[i] = e_delay*np.exp(-t_th*lambda_sort[i][i])
                            

                            total_numb[i] += tmp_numb[i]*dt_th*t_th
                                                                                  
                            t_th = (1.+dt_th)*t_th
                    if(Egamma>0.):
                        Z = Zchains[A,i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[i,2]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[i,3]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[i,3]*lambdas[i]/(mu*float(A))
                    ele += Xfraction*MeV*tmp_numb[i]*fchain[i,4]*lambdas[i]/(mu*float(A))
                    ele_th += np.power(tes[i],2.)*np.power(t,-3.)*Xfraction*MeV*total_numb[i]*fchain[i,4]*lambdas[i]/(mu*float(A))           
                else:
                        

           
                    if(Egamma>0.):
                        Z = Zchains[A,i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[i,2]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[i,3]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[i,3]*lambdas[i]/(mu*float(A))
                    ele += Xfraction*MeV*tmp_numb[i]*fchain[i,4]*lambdas[i]/(mu*float(A))
                    ele_th += Xfraction*MeV*tmp_numb[i]*fchain[i,4]*lambdas[i]/(mu*float(A))



            total_heats[k] += heat
            total_gammas[k] += gam

            total_elects[k] += ele
            total_elect_ths[k] += ele_th
            total_gamma_ths[k] += gam_th
        


            each_heats[k] += heat
            each_gammas[k] += gam
            each_elects[k] += ele
            each_elect_ths[k] += ele_th
            each_gamma_ths[k] += gam_th
#        print A, Xfraction
    #data = {'t': ts,'total':total_heats,'gamma':total_gammas, 'electron':total_elects, 'gamma_th':total_gamma_ths,'electron_th':total_elect_ths}
    #return data
    return ts, total_gamma_ths, total_elect_ths      
       # t *= 1.0 + delta_t
#    print 'end'

