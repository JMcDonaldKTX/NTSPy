####################################################################
# Author: Jason McDonald
# 2/22/2023
# Attempting to replicate the R Package NTS function mTAR.sim
# which can generate bivariate time series data with provided
# information.
# Source: https://github.com/cran/NTS
# R Package Authors: Xialu Liu, Rong Chen, Ruey Tsay
####################################################################


import numpy as np
from numpy import linalg as LA


class mTAR():
    def __init__(self, n_obs, thr, phi1, phi2, sigma1, sigma2=None, c1=None, c2=None, delay=[1,1], ini=500):
        self.n_obs = n_obs
        self.thr = thr
        #phi1, phi2, sigma1, and sigma2 must be arrays
        #if sigma2 is None, set it equal to sigma1's array
        self.phi1 = phi1
        self.phi2 = phi2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        if not isinstance(phi1, np.ndarray):
            if isinstance(phi1, (list, tuple)):
                self.phi1 = np.array(phi1)
            else: self.phi1 = np.array([phi1])
        if not isinstance(phi2, np.ndarray):
            if isinstance(phi2, (list, tuple)):
                self.phi2 = np.array(phi2)
            else: self.phi2 = np.array([phi2])
        if not isinstance(sigma1, np.ndarray):
            if isinstance(sigma1, (list, tuple)):
                self.sigma1 = np.array(sigma1)
            else: self.sigma1 = np.array([sigma1])
        if (sigma2).all() == None: self.sigma2 = self.sigma1
        else:
            if not isinstance(sigma2, np.ndarray):
                if isinstance(sigma2, (list, tuple)):
                    self.sigma2 = np.array(sigma2)
                else: self.sigma2 = np.array([sigma2])
        self.k1 = self.phi1.shape[0]
        self.k2 = self.phi2.shape[0]
        self.k3 = self.sigma1.shape[0]
        self.k4 = self.sigma2.shape[0]
        self.k = min(self.k1, self.k2, self.k3, self.k4)
        if c1 is None: self.c1 = np.zeros(self.k)
        else: self.c1 = c1[:self.k]
        if c2 is None: self.c2 = np.zeros(self.k)
        else: self.c2 = c2[:self.k]
        #self.c1 = c1[:self.k]
        #self.c2 = c2[:self.k]
        self.p1 = self.phi1.shape[1]
        self.p2 = self.phi2.shape[1]
        self.delay = delay
        self.ini = ini
        self.s1 = self.mtxroot(sigma1[:self.k,:self.k])
        self.s2 = self.mtxroot(sigma2[:self.k,:self.k])
        self.nT = self.n_obs + self.ini
        self.et = np.random.normal(size=(self.nT, self.k))
        #print(self.s1)
        #print(self.et)
        self.a1 = np.dot(self.et, self.s1)
        #print(self.a1)
        self.a2 = np.dot(self.et, self.s2)
        self.d = self.delay[1]
        if self.d <= 0:
            self.d = 1
        self.p = max(self.p1, self.p2, self.d)
        #print(self.a1)
        #self.zt = np.dot(np.ones((self.p, 1)), np.array([self.c1]).T) + self.a1[:self.p, :]
        self.zt = np.ones((1, self.p)) @ (np.ones((1, self.k))*self.c1).T + self.a1[:self.p]
        self.resi = np.zeros((self.nT, self.k))
        self.ist = self.p
        for i in range(self.ist, self.ini):
            self.wk = np.zeros(self.k)
            #print(self.zt)
            #print(i)
            if self.zt[(i-self.d), self.delay[0]-1] <= self.thr:
                self.resi[i,:] = self.a1[i,:]
                self.wk = self.wk + self.a1[i,:] + self.c1
                if self.p1 > 0:
                    for j in range(self.p1):
                        idx = (j) * self.k
                        phi = self.phi1[:, idx-1:(idx+self.k)]
                        #print(phi)
                        #print(self.wk)
                        #print(self.zt[i-j-1 ].T)
                        self.wk = self.wk + np.dot(phi, self.zt[i-j-1 ])
            else:
                self.resi[i,:] = self.a2[i,:]
                self.wk = self.wk + self.a2[i,:] + self.c2
                if self.p2 > 0:
                    for j in range(self.p2):
                        idx = (j) * self.k
                        phi = self.phi2[:, idx-1:(idx+self.k)]
                        #print(self.zt)
                        #print(self.wk)
                        #print(self.zt[i-j-1 ].T)
                        self.wk = self.wk + np.dot(phi, self.zt[i-j-1 ])
            self.zt = np.vstack((self.zt, self.wk))

        
    def mtxroot(self,sigma):
        if not isinstance(sigma, np.ndarray):
            sigma = np.array(sigma)
        sigma = (sigma.T + sigma)/2
        mw, mv = LA.eig(sigma)
        P = mv
        L = np.diag(np.sqrt(mw))
        sigmah = np.dot(P, np.dot(L, P.T))
        return sigmah
    
    def sim(self):
        n1 = 0
        for i in range(self.ini, self.nT):
            self.wk = np.zeros(self.k)
            if self.zt[(i-self.d)-1, self.delay[0]-1] <= self.thr:
                n1+=1
                self.resi[i,:] = self.a1[i,:]
                self.wk = self.wk + self.a1[i,:] + self.c1
                if self.p1 > 0:
                    for j in range(self.p1):
                        idx = (j) * self.k
                        phi = self.phi1[:, idx-1:(idx+self.k)]
                        self.wk = self.wk + np.dot(phi, self.zt[i-j-1 ])
            else:
                self.resi[i,:] = self.a2[i,:]
                self.wk = self.wk + self.a2[i,:] + self.c2
                if self.p2 > 0:
                    for j in range(self.p2):
                        idx = (j) * self.k
                        phi = self.phi2[:, idx-1:(idx+self.k)]
                        #print(phi)
                        #print(self.wk)
                        #print(self.zt[i-j-1 ])
                        self.wk = self.wk + np.dot(phi, self.zt[i-j-1 ])
            self.zt = np.vstack((self.zt, self.wk))
        mTAR_sim = {'series': self.zt[self.ini+1:self.nT,:], 'at': self.resi[self.ini+1:self.nT,:], 'threshold':self.thr,
                    'delay': self.delay, 'n1':n1, 'n2':(self.n_obs - n1)}
        return mTAR_sim
                        

