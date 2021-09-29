# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:20:34 2020

@author: S.D. Nelemans
"""

import os
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
import multiprocessing as mp
import datetime
import copy
from estimagic.optimization import pounders

os.chdir('D:/Users/Documents/Research/Heterogeneous Habits/Constantinides Ghosh')
# os.chdir('/home/p283303/')

#############################################
#SECTION 1: Hardcoded numerical basis
#############################################

glquadtable = np.array([[0.07053988969198875,	0.16874680185111386],
                        [0.37212681800161144,	0.29125436200606828],
                        [0.91658210248327356,	0.26668610286700129],
                        [1.70730653102834388,	0.16600245326950684],
                        [2.74919925530943213,	0.074826064668792371],
                        [4.0489253138508869,	0.0249644173092832211],
                        [5.6151749708616165,	0.006202550844572237],
                        [7.4590174536710633,	0.0011449623864769082],
                        [9.5943928695810968,	1.55741773027811975e-4],
                        [12.0388025469643163,	1.54014408652249157e-5],
                        [14.81429344263074,	1.08648636651798235e-6],
                        [17.948895520519376,	5.3301209095567148e-8],
                        [21.478788240285011,	1.757981179050582e-9],
                        [25.4517027931869055,	3.7255024025123209e-11],
                        [29.932554631700612,	4.7675292515781905e-13],
                        [35.013434240479,	3.372844243362438e-15],
                        [40.8330570567285711,	1.15501433950039883e-17],
                        [47.6199940473465021,	1.53952214058234355e-20],
                        [55.8107957500638989,	5.286442725569158e-24],
                        [66.5244165256157538,	1.6564566124990233e-28]]) #https://keisan.casio.com/exec/system/1281279441#

ghquadtable = np.asarray(pd.read_csv("ghquadtable.csv"))

primes = np.ravel(np.asarray(pd.read_csv("primes.txt",delim_whitespace=True,skiprows=[0,1,2,1004],header=None))) #https://primes.utm.edu/lists/small/10000.txt


#############################################
#SECTION 2: quasiMC and quadrature functions
#############################################0
def qmctorus(lnth,prim):
    primmat = np.transpose(np.tile(prim,(lnth,1)))
    kmat = np.tile(range(1,lnth+1),(len(prim),1))
    qmcmat, _ = np.modf(kmat*np.sqrt(primmat))
    return qmcmat     

def cdfstatevar(x,ptarget,ppois,nu,zmax): #used to generate quantiles from state variable distribution
    return np.average(stats.gamma.cdf(np.repeat(x,zmax+1),np.repeat(nu,zmax+1)+np.array(range(zmax+1))),weights=stats.poisson.pmf(range(zmax+1),ppois)) - ptarget 
    
def qmcstategrid(pars,hyperpars): #returns a state variable grid
    nu = pars[8]
    xi = pars[9]
    rho_x = pars[10]
    Tt = hyperpars["Tt"]
    K = hyperpars["qmcXsize"]
  
    qmcmat = qmctorus(K,primes[(5*Tt):(6*Tt)])
    ygrid = np.zeros((Tt,K))
  
    for i in range(K):
        ygrid[0,i] = optimize.brentq(cdfstatevar,0,1000,args=(qmcmat[0,i],rho_x*nu/(1-rho_x),nu,hyperpars["zmax"]))
  
    for t in range(1,Tt):
        for i in range(K):
            ygrid[t,i] = optimize.brentq(cdfstatevar,0,1000,args=(qmcmat[t,i],rho_x*ygrid[t-1,i],nu,hyperpars["zmax"]))
  
    xgrid = xi*ygrid
    return(xgrid)

def qmcaggcgrid(pars,hyperpars): #returns an aggregate consumption grid
    mu = pars[11]
    sigma_a = pars[12]
    Tt = hyperpars["Tt"]
    
    qmcmat = qmctorus(hyperpars["qmcXsize"],primes[(4*Tt):(5*Tt)])
    aggcgrid = mu + stats.norm.ppf(qmcmat)*sigma_a #TODO: trim support normal
    return(aggcgrid)

def qmcdeltagrid(pars,hyperpars,statevec): #returns a grid of values for delta_{i,t}
    sigma = pars[13]
    omega_hat = pars[14]
    sigma_hat = pars[15]
    Tt = hyperpars["Tt"]

    qmcmat = qmctorus(hyperpars["qmcDsize"],primes[0:(4*Tt)])
    etagrid = stats.norm.ppf(qmcmat[0:(2*Tt),:]) #TODO: trim support normal
    jgrid1 = stats.poisson.ppf(qmcmat[(2*Tt):(3*Tt),:],np.tile(statevec,(hyperpars["qmcDsize"],1)).transpose())
    jgrid2 = stats.poisson.ppf(qmcmat[(3*Tt):(4*Tt),:],omega_hat)
    deltagrid = np.sqrt(jgrid1)*sigma*etagrid[0:Tt,:] - jgrid1*sigma*sigma/2 + np.sqrt(jgrid2)*sigma_hat*etagrid[Tt:(2*Tt),] - jgrid2*sigma_hat*sigma_hat/2
    for t in range(1,Tt):
        deltagrid[t,] = deltagrid[t-1,] + deltagrid[t,]
    
    return(np.exp(deltagrid))

#############################################
#SECTION 3: h, MRS and Euler
#############################################
def hgridfun(pars,aggcons,deltamat): #generates a grid of h for a grid of delta
    a = pars[0]
    phi = pars[2]
  
    Tt = len(aggcons)
    K = np.size(deltamat,1)
    h = np.zeros((Tt,K))
    h[0,:] = np.repeat(np.exp(-aggcons[0]),K)/deltamat[0,:]
    h[1,:] = (np.ones(K)+phi*(deltamat[0,:]-np.ones(K)))/deltamat[1,:]*np.repeat(np.exp(aggcons[1]),K) + np.repeat((1-a),K)/deltamat[0,:]/np.repeat(np.exp(aggcons[0])*np.exp(aggcons[1]),K)
    for t in range(2,Tt):
        h[t,:] = (np.ones(K)+phi*(deltamat[t-1,:]-np.ones(K)))/deltamat[t,:]/np.repeat(np.exp(aggcons[t]),K)
        for j in range(1,t):
            h[t,:] = h[t,:] + ((1-a)**j)*(np.ones(K)+phi*(deltamat[t-j-1,:]-np.ones(K)))/deltamat[t,:]/np.repeat(np.exp(np.sum(aggcons[(t-j):(t+1)])),K)
    return(h)

def mrsxmrs(row,pars,deltavec,hvec,statevar): #returns the future MRS, the lhs of the Euler eq. and the proportion of households that dropped out 
    a = pars[0]
    b = pars[1]
    phi = pars[2]
    gamma = pars[3]
    rho = pars[4]
    alpha_d = pars[5]
    beta_d = pars[6]
    sigma_d = pars[7]
    nu = pars[8]
    xi = pars[9]
    rho_x = pars[10]
    B0 = pars[16]
    B1 = pars[17]
    
    xnext = row[0]
    ddecnext = row[1]
    weight = row[2]
    K = len(deltavec)
  
    Ezm = B0 + B1*nu*xi/(1-rho_x)
    k0 = np.log(np.exp(Ezm)+1) - Ezm*np.exp(Ezm)/(np.exp(Ezm)+1)
    k1 = np.exp(Ezm)/(np.exp(Ezm)+1)
    
    glogmrs = ((np.repeat(ddecnext,K)-(np.ones(K)+phi*(deltavec-np.ones(K)))*b/deltavec-(1-a)*b*hvec)/(1-b*hvec))
    nacount = np.sum(glogmrs<0)*weight
    glogmrs[glogmrs<0] = np.nan
    if(~np.isnan(glogmrs).all()):
        mrs = rho*np.nanmean(glogmrs**(-gamma))*weight
    else:
        mrs = np.nan
    return np.array([mrs,np.exp(k1*B1*xnext-(-k0+B0-k1*B0-alpha_d-sigma_d*sigma_d/2+(B1-beta_d)*statevar))*mrs,nacount])

def EmrsEmrsx(pars,hyperpars,deltavec,hvec,statevar): #returns 3 columns with future x, future deltadeltaec, weights
    # gamma = pars[3]
    nu = pars[8]
    xi = pars[9]
    rho_x = pars[10]
    mu = pars[11]
    sigma_a = pars[12]
    sigma = pars[13]
    omega_hat = pars[14]
    sigma_hat = pars[15]
  
    zpoismax = int(max(stats.poisson.ppf(hyperpars["statePmax"],rho_x*statevar/xi),5))
    GLweights = np.average(np.power(np.transpose(np.tile(glquadtable[:,0],(zpoismax+1,1))),(np.tile(np.repeat(nu-1,zpoismax+1)+range(zpoismax+1),(np.size(glquadtable,0),1))))/special.gamma(np.tile(np.repeat(nu,zpoismax+1)+range(zpoismax+1),(np.size(glquadtable,0),1))),
                              axis=1,weights=stats.poisson.pmf(range(zpoismax+1),rho_x*statevar/xi))*glquadtable[:,1]

    [Emrs,Emrsx,nacount] = np.array([0,0,0])
    
    for i in range(np.size(glquadtable,0)):
        GLw = glquadtable[i,0]
        # GLw = glquadtable[i,0]/(exp(gamma*(gamma-1)*sigma*sigma/2)-1)
        Jmax = int(stats.poisson.ppf(hyperpars["jPmax"],GLw))
        Jhatmax = int(stats.poisson.ppf(hyperpars["jPmax"],omega_hat))
        jetaaggcnext = np.array([(j,jh,ghquadtable[ghind,0],ghquadtable[ghind,1],ghquadtable[ghind,2]) for j in range(Jmax+1) for jh in range(Jhatmax+1) for ghind in range(np.size(ghquadtable,0))])
        jprobsvec = stats.poisson.pmf(range(Jmax+1),GLw)
        jhatprobsvec = stats.poisson.pmf(range(Jhatmax+1),omega_hat)
        weightcompon = np.array([(jprobsvec[j]*jhatprobsvec[jh]*ghquadtable[ghind,3]) for j in range(Jmax+1) for jh in range(Jhatmax+1) for ghind in range(np.size(ghquadtable,0))])

        xddecnext = list(np.transpose(np.vstack([np.repeat(glquadtable[i,0],np.size(jetaaggcnext,0)),
                                                     np.exp(mu + sigma_a*jetaaggcnext[:,4] + np.sqrt(jetaaggcnext[:,0])*sigma*jetaaggcnext[:,2] - jetaaggcnext[:,0]*sigma*sigma/2 + np.sqrt(jetaaggcnext[:,1])*sigma_hat*jetaaggcnext[:,3] - jetaaggcnext[:,1]*sigma_hat*sigma_hat/2),
                                                     weightcompon/stats.poisson.cdf(Jmax,GLw)/stats.poisson.cdf(Jhatmax,omega_hat)*GLweights[i]])))    
        EmrsExmrsgrid = np.vstack([mrsxmrs(row,pars,deltavec,hvec,statevar) for row in xddecnext])

        reweight = np.sum(np.array(xddecnext)[~np.isnan(EmrsExmrsgrid[:,0]),2])
        Emrs,Emrsx,nacount = [Emrs,Emrsx,nacount] + np.nansum(EmrsExmrsgrid,0)/reweight
    
    return Emrs,Emrsx,nacount



#############################################
#SECTION 4: Moment and penalty calculation
#############################################
def moment_penalty(pars,hyperpars,empiricalmoments):
    print(pars)
    # a = pars[0]
    # b = pars[1]
    # phi = pars[2]
    # gamma = pars[3]
    # rho = pars[4]
    alpha_d = pars[5]
    beta_d = pars[6]
    sigma_d = pars[7]
    nu = pars[8]
    xi = pars[9]
    rho_x = pars[10]
    mu = pars[11]
    sigma_a = pars[12]
    sigma = pars[13]
    omega_hat = pars[14]
    sigma_hat = pars[15]
    B0 = pars[16]
    B1 = pars[17]
    Tt = hyperpars["Tt"] # Number of periods
    B = hyperpars["B"] # Burn-in phase
    K = hyperpars["qmcXsize"] # Number of simulations
  
    xgrid = qmcstategrid(pars,hyperpars)
    aggcgrid = qmcaggcgrid(pars,hyperpars)
    Emrs = np.zeros((Tt,K))
    Exmrs = np.zeros((Tt,K))
    nacount = np.zeros((Tt,K))
    for i in range(K):
        deltagrid = qmcdeltagrid(pars,hyperpars,xgrid[:,i])
        hgrid = hgridfun(pars,aggcgrid[:,i],deltagrid)
        # if __name__ == '__main__':
        #     pool = mp.Pool(int(os.environ['SLURM_JOB_CPUS_PER_NODE']))
        #     EmrsExmrsgrid = pool.starmap(EmrsEmrsx,[(pars,hyperpars,deltagrid[t,:],hgrid[t,:],xgrid[t,i]) for t in range(Tt)])
        #     pool.close()
        #     pool.join()
        EmrsExmrsgrid = [EmrsEmrsx(pars,hyperpars,deltagrid[t,:],hgrid[t,:],xgrid[t,i]) for t in range(Tt)]
        Emrs[:,i] = np.array(EmrsExmrsgrid)[:,0]
        Exmrs[:,i] = np.array(EmrsExmrsgrid)[:,1]
        nacount[:,i] = np.array(EmrsExmrsgrid)[:,2]
    
    Emrs_trim = np.delete(Emrs,range(B),0)
    Exmrs_trim = np.delete(Exmrs,range(B),0)
    nacount_trim = np.delete(nacount,range(B),0)
    
    Ex = nu*xi/(1-rho_x)
    Vx = nu*xi*xi/(1-rho_x)/(1-rho_x)
    MUx = (2*nu*xi*xi/(1-rho_x*rho_x*rho_x))*(1+(3*rho_x/(1-rho_x)*(1+(rho_x/(1-rho_x)))))
    Ew = Ex
    Vw = Vx
    MUw = MUx
    # repar = 1/(exp(gamma*(gamma-1)*sigma*sigma/2)-1)
    # Ew = repar*Ex
    # Vw = repar*repar*Vx
    # MUw = (repar**3)*MUx
    Ezm = B0 + B1*nu*xi/(1-rho_x)
    k0 = np.log(np.exp(Ezm)+1) - Ezm*np.exp(Ezm)/(np.exp(Ezm)+1)
    k1 = np.exp(Ezm)/(np.exp(Ezm)+1)
    rfmat = np.ones((Tt-B,K))/Emrs_trim
    rfautocors = np.zeros(K)
    for i in range(K):
        [rfautocors[i],_] = stats.pearsonr(rfmat[range(1,Tt-B),i],rfmat[range(Tt-B-1),i])
    
    moments_pen = np.zeros(18)
    moments_pen[0] = mu
    moments_pen[1] = sigma_a
    moments_pen[2] = alpha_d + beta_d*Ex
    moments_pen[3] = np.sqrt(beta_d*beta_d*Vx + sigma_d*sigma_d)
    moments_pen[4] = beta_d*beta_d*rho_x*Vx/(beta_d*beta_d*Vx+sigma_d*sigma_d)
    moments_pen[5] = -sigma*sigma/2*Ew - sigma_hat*sigma_hat/2*omega_hat
    moments_pen[6] = (sigma*sigma + (sigma**4)/4)*Ew + (sigma**4)/4*Vw + (sigma_hat*sigma_hat + (sigma_hat**4)/4)*omega_hat
    moments_pen[7] = (-3*(sigma**4)/2 - (sigma**6)/8)*Ew - (3*(sigma**4)/2 + 3*(sigma**6)/8)*Vw - (sigma**6)/8*MUw - (3*(sigma_hat**4)/2 + (sigma_hat**6)/8)*omega_hat
    moments_pen[8] = np.average(rfmat)
    moments_pen[9] = np.average(np.std(rfmat,axis=0))
    moments_pen[10] = np.average(rfautocors)
    moments_pen[11] = B0 + B1*Ex
    moments_pen[12] = B1*np.sqrt(Vx)
    moments_pen[13] = rho_x
    moments_pen[14] = k0 + (k1-1)*B0 + alpha_d + (beta_d + (k1-1)*B1)*Ex
    moments_pen[15] = np.sqrt((k1*k1*B1*B1 + (beta_d-B1)*(beta_d-B1))*Vx + 2*k1*B1*(beta_d-B1)*rho_x + sigma_d*sigma_d)
    moments_pen[16] = (k1*B1*(beta_d-B1)*(1+rho_x*rho_x) + (k1*k1*B1*B1 + (beta_d-B1)*(beta_d-B1))*rho_x)*Vx/((k1*k1*B1*B1+(beta_d-B1)*(beta_d-B1))*Vx+2*k1*B1*(beta_d-B1)*rho_x+sigma_d*sigma_d)
    
    moments_pen[17] = hyperpars["Bpenalty"]*np.sum(np.absolute(Exmrs_trim-np.ones((Tt-B,K)))) + hyperpars["napenalty"]*np.sum(nacount_trim)
    
    momentweights = np.ones(18)
    momentweights[[8,14]] = [10,10]
    
    print(datetime.datetime.now())
    print(sum(np.array((moments_pen-np.append(empiricalmoments,0))*momentweights)**2))
    print(np.sum(Exmrs_trim-np.ones((Tt-B,K))))
    print(np.sum(nacount_trim))
    return np.array(moments_pen-np.append(empiricalmoments,0))*momentweights


#############################################
#SECTION 5: Reparametrisation
#############################################
def depar_fun(repars):
    pars = copy.deepcopy(repars)
    pars[3] = np.exp(pars[3]/(1-pars[3]))
    pars[[5,7,8,9,10,11,12,13,14,15]] = pars[[5,7,8,9,10,11,12,13,14,15]]/(1-pars[[5,7,8,9,10,11,12,13,14,15]])
    pars[[6,16,17]] = np.log(pars[[6,16,17]]/(1-pars[[6,16,17]]))
    return pars

def repar_fun(pars):
    repars = copy.deepcopy(pars)
    repars[3] = np.log(repars[3])/(1+np.log(repars[3]))
    repars[[5,7,8,9,10,11,12,13,14,15]] = repars[[5,7,8,9,10,11,12,13,14,15]]/(1+repars[[5,7,8,9,10,11,12,13,14,15]])
    repars[[6,16,17]] = np.exp(repars[[6,16,17]])/(1+np.exp(repars[[6,16,17]]))
    return repars

def mom_habit(repars):
    pars = depar_fun(repars)
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_nohabit(repars):
    pars = depar_fun(np.concatenate((np.array([.99,0,.5]),repars)))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_nohabit_fix(repars):
    pars = depar_fun(np.concatenate((np.array([.99,0,.5]),repars[0:8],myempiricalmoments[[0,1]],repars[8:13])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_inthabit_fix(repars):
    pars = depar_fun(np.concatenate((np.array([.99,0.05,1]),repars[0:8],myempiricalmoments[[0,1]],repars[8:13])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_exthabit_fix(repars):
    pars = depar_fun(np.concatenate((np.array([.99,0.05,0]),repars[0:8],myempiricalmoments[[0,1]],repars[8:13])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_inthabit_emp(repars):
    pars = depar_fun(np.concatenate((repars[0:2],[1],repars[2:10],myempiricalmoments[[0,1]],repars[10:15])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_exthabit_emp(repars):
    pars = depar_fun(np.concatenate((repars[0:2],[0],repars[2:10],myempiricalmoments[[0,1]],repars[10:15])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)

def mom_mixhabit_emp(repars):
    pars = depar_fun(np.concatenate((repars[0:11],myempiricalmoments[[0,1]],repars[11:16])))
    return moment_penalty(pars,myhyperpars,myempiricalmoments)


#############################################
#SECTION 6: Initialisation
#############################################
mypars = np.array([
  0.8,  # a 0
  0.02,  # b 1
  0.2,  # phi 2
  6.343,  # gamma 3
  0.93,  # rho 4
  0.001,  # alpha_d 5
  -40.1,  # beta_d 6
  0.02,  # sigma_d 7
  1.4,  # nu 8
  0.08,  # xi 9
  0.876,  # rho_x 10
  0.008,  # mu 11
  0.004,  # sigma_a 12
  0.0718,  # sigma 13
  0.110,  # omega_hat 14
  0.0623,  # sigma_hat 15
  2, # B0 16
  5  # B1 17
])

myrepars = repar_fun(mypars)
myrepars_nohabit_fix = myrepars[[3,4,5,6,7,8,9,10,13,14,15,16,17]]

mynewpars = np.array([ 9.90000000e-01,  0.00000000e+00,  5.00000000e-01,  5.24619336e+00,
  9.48564796e-01,  2.57790053e-02, -7.13138621e-01,  3.13737509e-02,
  1.49573296e+00,  1.98734470e-03,  9.23005315e-01,  3.97393964e-03,
  8.33150804e-03,  2.48308494e-01,  8.54166196e-02,  5.37916070e-02,
  3.81372274e+00, -7.05747995e-01])
mynewrepars = repar_fun(mynewpars)

mynewrepars_nohabit = mynewrepars[3:18]
mynewrepars_nohabit_fix = mynewrepars[[3,4,5,6,7,8,9,10,13,14,15,16,17]]
mynewrepars_inthabit_emp = mynewrepars[[0,1,3,4,5,6,7,8,9,10,13,14,15,16,17]]
mynewrepars_exthabit_emp = mynewrepars[[0,1,3,4,5,6,7,8,9,10,13,14,15,16,17]]
mynewrepars_mixhabit_emp = mynewrepars[[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17]]

myhyperpars = {
  "Tt":20, #number of periods
  "B":10, #burn-in period
  "qmcXsize":10, #number of simulated aggregate trajectories
  "qmcDsize":10, #number of simulated individual trajectories
  "zmax":50, #terms in Poisson weighted sum of gamma distributions for statevar cdf
  "jPmax":1-1e-5, #cut-off probability for possible next-period outcomes of Poisson variables in delta 
  "statePmax":1-1e-5, #cut-off probability for possible next-period outcomes of Poisson variables in state variable generation
  "Bpenalty":1, #pentalty weight parameter for deviations from the Euler equation
  "napenalty":1 #pentalty weight parameter for households dropping out of the model
}

myempiricalmoments = np.loadtxt("empiricalmoments.txt")


#############################################
#SECTION 7: And now, we fly
#############################################


print(datetime.datetime.now().time())
# print(pounders.minimize_pounders_np(mom_nohabit_fix, myrepars_nohabit_fix, (np.repeat(0,13),np.repeat(1,13)),init_tr=.001))
print(optimize.least_squares(mom_inthabit_emp, mynewrepars_inthabit_emp, bounds=(0,1), xtol=None))
print(datetime.datetime.now().time())

