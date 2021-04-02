import math
import numpy as np
from numba import jit

heatingrate = np.loadtxt("input_files/Heating_Korobkin2012.dat")
heat_time = 10**heatingrate[:,0]
heat_rate = 10**heatingrate[:,1]

day = 86400.
c = 2.99792458e10
sigma_SB = 5.670373e-5

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def calc_lightcurve(Mej,vej,alpha_max,alpha_min,n,kappa_low,kappa_high,be_kappa):    

    Nbeta = 100
    rho0 = Mej*(n-3.)/(4.*np.pi*np.power(vej,3.))/(1.-np.power(alpha_max/alpha_min,-n+3.))
    be_min = vej*alpha_min/c
    be_max = vej*alpha_max/c

    be = be_min
    dbe = be_min/float(Nbeta)

    be_tmps =[]
    tau_tmps = []
    dM_tmps = []
    td_tmps = []
    M = 0.
    while(be <= be_max):
        if(be > be_kappa):
            tau = kappa_low*be_min*c*rho0*(np.power(be/be_min,-n+1.)-np.power(be_max/be_min,-n+1.))/(n-1.)
        else:
            tau = kappa_low*be_min*c*rho0*(np.power(be_kappa/be_min,-n+1.)-np.power(be_max/be_min,-n+1.))/(n-1.)+kappa_high*be_min*c*rho0*(np.power(be/be_min,-n+1.)-np.power(be_kappa/be_min,-n+1.))/(n-1.)
        dM = 4.*np.pi*np.power(vej,3.)*rho0*np.power(be/be_min,-n+2.)*dbe/be_min
        td2 = tau*be
        tau_tmps.append(tau)
        td_tmps.append(td2)
        be_tmps.append(be)
        dM_tmps.append(dM)
    
        be += dbe

    bes = np.array(be_tmps)
    dMs = np.array(dM_tmps)
    tds = np.array(td_tmps)
    taus = np.array(tau_tmps)
    Eins = np.zeros((len(bes)))



    dt = 0.005*day
    t = 0.01*day
    ts = []
    Ls = []
    temps = []
    j = 0
    k=0
    while(t<30.*day):
        

    
        while(t>heat_time[k]*day):
            k += 1
        heat_th0 = interp(t/day,heat_time[k-1],heat_time[k],heat_rate[k-1],heat_rate[k]) 

    
        while(t+0.5*dt>heat_time[k]*day):
            k += 1
        heat_th1 = interp((t+0.5*dt)/day,heat_time[k-1],heat_time[k],heat_rate[k-1],heat_rate[k]) 
    
        while(t+dt>heat_time[k]*day):
            k += 1
        heat_th2 = interp((t+dt)/day,heat_time[k-1],heat_time[k],heat_rate[k-1],heat_rate[k]) 
 
    
        Ltot = 0.
        for i in range(0,len(bes)):
            vel =bes[i]*c
        #RK step 1
            E_RK1 = Eins[i]
            t_RK1 = t
            t_dif = tds[i]/t_RK1
            heat = dMs[i]*(heat_th0)
        
            if(t_RK1>t_dif):
                tesc = t_dif + bes[i]*t_RK1
            else:
                tesc = t_RK1 + bes[i]*t_RK1
        
            ymax = np.sqrt(0.5*t_dif/t_RK1)
            erfc = math.erfc(ymax)
       
            L_RK1 = erfc*E_RK1/tesc
            dE_RK1 = (-E_RK1/t_RK1 - L_RK1 + heat)*dt
       
        
        #RK step 2
            E_RK2 = Eins[i] + 0.5*dE_RK1    
            t_RK2 = t+0.5*dt       
            t_dif = tds[i]/t_RK2 
            heat = dMs[i]*(heat_th1)
        
            if(t_RK2>t_dif):
                tesc = t_dif + bes[i]*t_RK2
            else:
                tesc = t_RK2 + bes[i]*t_RK2
            
            ymax = np.sqrt(0.5*t_dif/t_RK2)
            erfc = math.erfc(ymax)

            L_RK2 = erfc*E_RK2/tesc
            dE_RK2 = (-E_RK2/t_RK2 - L_RK2 + heat)*dt
        #print '2',L_RK2, dE_RK2,erfc,heat,tesc
        
        #RK step 3
            E_RK3 = Eins[i] + 0.5*dE_RK2
            t_RK3 = t+0.5*dt       
            t_dif = tds[i]/t_RK3
            heat = dMs[i]*(heat_th1)
        
            if(t_RK3>t_dif):
                tesc = t_dif + bes[i]*t_RK3
            else:
                tesc = t_RK3 + bes[i]*t_RK3

            ymax = np.sqrt(0.5*t_dif/t_RK3)
            erfc = math.erfc(ymax)
 
            L_RK3 = erfc*E_RK3/tesc
            dE_RK3 = (-E_RK3/t_RK3 - L_RK3 + heat)*dt
        #print '3',L_RK3, dE_RK3,erfc,heat
        
        #RK step 4
            E_RK4 = Eins[i] + dE_RK3
            t_RK4 = t +dt        
            t_dif = tds[i]/t_RK4
            heat = dMs[i]*(heat_th2)
        
            if(t_RK4>t_dif):
                tesc = t_dif + bes[i]*t_RK4
            else:
                tesc = t_RK4 + bes[i]*t_RK4
     
            ymax = np.sqrt(0.5*t_dif/t_RK4)
            erfc = math.erfc(ymax)

            L_RK4 = erfc*E_RK4/tesc
            dE_RK4 = (-E_RK4/t_RK4 - L_RK4 + heat)*dt
        #print '4',L_RK4, dE_RK4,erfc,heat
            Eins[i] += (dE_RK1+2.*dE_RK2+2.*dE_RK3+dE_RK4)/6.
            Ltot += (L_RK1 + 2.*L_RK2 + 2.*L_RK3+L_RK4)/6.
        t += dt
    #search for the shell of tau = 1
    
        if(taus[0]/(t*t)>1. and taus[len(bes)-1]/(t*t)<1.):
            l=0
            while(taus[l]/(t*t)>1.):
                l+=1
            be = interp(t*t,taus[l-1],taus[l],bes[l-1],bes[l])
            r = be*c*t
            Eint = interp(t*t,taus[l-1],taus[l],Eins[l-1],Eins[l])
        elif(taus[len(bes)-1]/(t*t)>1.):
            l = len(bes)-1
            be = bes[l]
            r = be*c*t
            Eint = Eins[l]
        else:
            l = 0
            be = bes[0]
            r = be*c*t
            Eint = Eins[0]
        tmp = Ltot/(4.*np.pi*sigma_SB*r*r);
        temp = np.power(tmp,0.25)
        if(j<10):
            Ls.append(Ltot)
            ts.append(t)
            temps.append(temp)
        elif(j<100):
            if(j%3==0):
                Ls.append(Ltot)
                ts.append(t)
                temps.append(temp)
        elif(j<1000):
            if(j%30==0):
                Ls.append(Ltot)
                ts.append(t)
                temps.append(temp)
        elif(j<10000):
            if(j%100==0):
                Ls.append(Ltot)
                ts.append(t)
                temps.append(temp)



        j += 1

    for i in range(len(ts)):
        ts[i]=ts[i]/day


    data = {'t':np.array(ts),'LC':np.array(Ls),'T':np.array(temps)}
    return data        
       # t *= 1.0 + delta_t
    #print 'end'

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def interp(x, x1, x2, y1, y2):
    n_p = np.log(y2/y1)/np.log(x2/x1)
    f0 = y1*np.power(x1,-n_p)
    return f0*np.power(x,n_p)
    
