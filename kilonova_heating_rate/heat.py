import numpy as np
import pandas as pd
from scipy import optimize
from astropy import constants as const
from astropy import units as u

import importlib
th = importlib.import_module('kilonova_heating_rate.thermalization', __name__)
bt = importlib.import_module('kilonova_heating_rate.bateman', __name__)



day = 86400.
MeV = (1.0e6*u.eV).to(u.g*u.cm**2/u.s**2).value
mu = const.u.cgs.value
c = const.c.cgs.value


def calc_heating_rate(Mej, vmin, vmax, Amin, Amax, ffraction, kappa_effs, n):
    
    # Temporary workaround. In order to clean this up also clean up thermalization.py
    vej = vmin
    alpha_max = vmax/vmin
    alpha_min = 1

    t_initial = 0.01*day
    t_final = 10.*day
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
    
        filename = 'input_files/table/'+str(A)+'.txt'
    
#A, Z, Q[MeV], Egamma[MeV], Eelec[MeV], Eneutrino[MeV], tau[s]
        fchain = pd.read_csv(filename,delim_whitespace=True,header=None)
#    print 'length of the chain',A,len(fchain)

        tmp = []
        N = len(fchain)
        for i in range(0,N):
            tmp.append(1.0/fchain[6][i])
        lambdas = np.array(tmp)
####determine the thermalization time in units of day for each element
        tes = np.zeros(N)
        total_numb = np.zeros(N)
        for i in range(0,N):
            Z = fchain[1][i]
            Egamma = fchain[3][i]
            Eele = fchain[4][i]
       
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
                
                Eele = fchain[4][i]
                if(t > 0.003*tes[i]):
                    if(Eele > 0.):
                        tau1 = t/tes[i]
                        if(tau1<2.):
                            tau0 = 0.03*tau1#0.05*tau1
                            root = optimize.newton(th.calc_zero_energy, tau0, args=(tau1,Eele,))
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
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[2][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    ele_th += np.power(tes[i],2.)*np.power(t,-3.)*Xfraction*MeV*total_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))           
                else:
                        

           
                    if(Egamma>0.):
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[2][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    ele_th += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))



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
    data = {'t':np.multiply(ts,1./day),'total':total_heats,'gamma':total_gammas, 'electron':total_elects, 'gamma_th':total_gamma_ths,'electron_th':total_elect_ths}
    return data        
       # t *= 1.0 + delta_t
#    print 'end'


def calc_heating_rate_sf(Mej,vej, Amin,Amax,ffraction,ffission_A,ffission_X,kappa_effs,alpha_max,alpha_min,n):

    t_initial = 0.01*day
    t_final = 3000.*day
    delta_t = 0.3#0.3

    Nth = 300#40 for default

    
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

   # print 'total time step = ', len(total_heats)
    for ii in range(0,len(ffission_A)):
        A = ffission_A[ii]
#        each_heats = np.zeros(len(ts))
#        each_gammas = np.zeros(len(ts))
#        each_gamma_ths = np.zeros(len(ts))
#        each_elects = np.zeros(len(ts))
#        each_elect_ths = np.zeros(len(ts))
        Xfraction = float(A)*ffission_X[ii]
    
        filename = '../table_fission/'+str(A)+'.txt'
    
#A, Z, Q[MeV], Egamma[MeV], Eelec[MeV], Eneutrino[MeV], tau[s]
        fchain = pd.read_csv(filename,delim_whitespace=True,header=None)
#    print 'length of the chain',A,len(fchain)

        tmp = []
        N = len(fchain)
        for i in range(0,N):
            tmp.append(1.0/fchain[10][i])
        lambdas = np.array(tmp)
####determine the thermalization time in units of day for each element
        te1s = np.zeros(N)
        te2s = np.zeros(N)
        total_numb = np.zeros(N)
        Egamma = 0.
        for i in range(0,N):
            Z = fchain[1][i]
            Eele1 = fchain[4][i]
            Eele2 = fchain[5][i]
            
            if(Eele1>0.):
                te1s[i] = th.calc_thermalization_time_sf(Eele1,Mej,vej,Aave,alpha_max,alpha_min,n)
            if(Eele2>0.):
                te2s[i] = th.calc_thermalization_time_sf(Eele2,Mej,vej,Aave,alpha_max,alpha_min,n)


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
                #else:
                  #  print 'chain is too long'
           # print i,lambda_sort[i][i]
            
        
#        number10s.append(tmp_numb[0])
#        number11s.append(tmp_numb[1])
#        number12s.append(tmp_numb[2])
#        number13s.append(tmp_numb[3])
    
            heat = 0.0
            gam = 0.0
            ele1 = 0.0
            ele_th1 = 0.0
            ele2 = 0.0
            ele_th2 = 0.0
            gam_th = 0.0
    
            for i in range(0,N):
                
                Eele1 = fchain[4][i]
                if(t > 0.03*te1s[i]):
                    if(Eele1 > 0.):
                        tau1 = t/te1s[i]
                        if(tau1<2.):
                            tau0 = 0.03*tau1#0.05*tau1
                            root = optimize.newton(th.calc_zero_energy_sf, tau0, args=(tau1,Eele1,))
                            tau0 = root
                        else:
                            tau0 = 0.4#0.05*tau1
                #else:
                #    tau0 = 0.01*tau1
                #tau0 = 1.
                
                #print "tau0: ", tau1,tau0
                        delta_t = (tau1-tau0)*te1s[i]
            #fth = fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
           # print t,delta_t,delta_t/t,fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
                        t_th = tau0*te1s[i]
                        tmp_n_t_th = 1./float(Nth-1)
                        dt_th = np.power(tau1/tau0,tmp_n_t_th)-1.
                        total_numb[i] = 0.0
                        for j in range(0,Nth):
                            coeff = coeffs[i]*np.power(t_th,i)
                            tau = t_th/te1s[i]
                    #e_delay= calc_e_tau_tau0(tau,tau1)
                            e_delay = th.epsilon_tau_sf(tau,tau1,Eele1)

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
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele1 += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    ele_th1 += np.power(te1s[i],2.)*np.power(t,-3.)*Xfraction*MeV*total_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))           
                else:
                        

           
                    if(Egamma>0.):
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele1 += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))
                    ele_th1 += Xfraction*MeV*tmp_numb[i]*fchain[4][i]*lambdas[i]/(mu*float(A))

##
                Eele2 = fchain[5][i]
                if(t > 0.03*te2s[i]):
                    if(Eele2 > 0.):
                        tau1 = t/te2s[i]
                        if(tau1<2.):
                            tau0 = 0.03*tau1#0.05*tau1
                            root = optimize.newton(th.calc_zero_energy_sf, tau0, args=(tau1,Eele2,))
                            tau0 = root
                        else:
                            tau0 = 0.4#0.05*tau1
                #else:
                #    tau0 = 0.01*tau1
                #tau0 = 1.
                
                #print "tau0: ", tau1,tau0
                        delta_t = (tau1-tau0)*te2s[i]
            #fth = fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
           # print t,delta_t,delta_t/t,fchain[6][i]*tes[i]*tes[i]*(np.exp(delta_t/fchain[6][i])-1.0)*np.power(t,-3.)
                        t_th = tau0*te2s[i]
                        tmp_n_t_th = 1./float(Nth-1)
                        dt_th = np.power(tau1/tau0,tmp_n_t_th)-1.
                        total_numb[i] = 0.0
                        for j in range(0,Nth):
                            coeff = coeffs[i]*np.power(t_th,i)
                            tau = t_th/te2s[i]
                    #e_delay= calc_e_tau_tau0(tau,tau1)
                            e_delay = th.epsilon_tau_sf(tau,tau1,Eele2)

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
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele2 += Xfraction*MeV*tmp_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))
                    ele_th2 += np.power(te2s[i],2.)*np.power(t,-3.)*Xfraction*MeV*total_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))           
                else:
                        

           
                    if(Egamma>0.):
                        Z = fchain[1][i]
                        kappa_eff = kappa_effs[A][Z]
                        fth_gamma = th.calc_gamma_deposition(kappa_eff,t,Mej,vej,alpha_min,alpha_max,n)
                    else:
                        fth_gamma = 0.
                
                    heat += Xfraction*MeV*tmp_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))
                    gam += Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    gam_th += fth_gamma*Xfraction*MeV*tmp_numb[i]*fchain[3][i]*lambdas[i]/(mu*float(A))
                    ele2 += Xfraction*MeV*tmp_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))
                    ele_th2 += Xfraction*MeV*tmp_numb[i]*fchain[5][i]*lambdas[i]/(mu*float(A))


##
            total_heats[k] += heat
            total_gammas[k] += gam

            total_elects[k] += ele1 + ele2
            total_elect_ths[k] += ele_th1 + ele_th2
            total_gamma_ths[k] += gam_th
        


#            each_heats[k] += heat
#            each_gammas[k] += gam
#            each_elects[k] += ele
#            each_elect_ths[k] += ele_th
#            each_gamma_ths[k] += gam_th
#        print A, Xfraction
    data = {'t':np.multiply(ts,1./day),'total':total_heats,'gamma':total_gammas, 'electron':total_elects, 'gamma_th':total_gamma_ths,'electron_th':total_elect_ths}
    return data        
       # t *= 1.0 + delta_t
    #print 'end'



def hokan(x,x1,x2,y1,y2):
    return (x2-x)*y1/(x2-x1) + (x-x1)*y2/(x2-x1)
#Katz integral for a given final time
def calc_Katz_integral(Mej,data,t_kf):
    ts = data['t']*day
    integs = []
    integ = ts[0]*ts[0]*(data['gamma_th'][0]+data['electron_th'][0])/(day*day)
    integs.append(integ)
    for i in range(0,len(ts)-1):
        dt = ts[i+1] - ts[i]
        integ += 0.5*(ts[i+1]*data['gamma_th'][i+1]+ts[i+1]*data['electron_th'][i+1] + ts[i]*data['gamma_th'][i]+ts[i]*data['electron_th'][i])*dt/(day*day)
        integs.append(integ)
    k = 0
    while(data['t'][k]<t_kf):
        k +=1
    katz = hokan(t_kf,data['t'][k-1],data['t'][k],integs[k-1],integs[k])
    return katz*Mej
#Katz integral time series
def calc_Katz_integral_timeseries(Mej,data):
    ts = data['t']*day
    integs = []
    integ = ts[0]*ts[0]*(data['gamma_th'][0]+data['electron_th'][0])/(day*day)

    integs.append(integ)

    for i in range(0,len(ts)-1):
        dt = ts[i+1] - ts[i]
        integ += 0.5*(ts[i+1]*data['gamma_th'][i+1]+ts[i+1]*data['electron_th'][i+1] + ts[i]*data['gamma_th'][i]+ts[i]*data['electron_th'][i])*dt/(day*day)
        integs.append(integ*Mej)
    output = {'t':data['t'],'katz':integs}


    return output
