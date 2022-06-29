"""
Class that contains the instrumental response model for single ion species based on the CTOF in-flight calibration.
The CTOF in-flight calibration is given in CTOF_cal.py
Author: Nils Janitzek (2021)
"""

#pylab imports
import pylab

#numpy imports
from numpy import *

#scipy imports
from scipy import optimize, stats
from scipy.special import gamma,gammainc, erf, binom
from scipy.optimize import leastsq
from scipy.optimize import fmin_bfgs as minimizer 

#matplotlib imports
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm, cbook
import matplotlib.colors as colors
from matplotlib.colors import Normalize, LogNorm
from matplotlib.path import Path
 
#import selfmade python modules
#from pylib import dbData
from CTOF_cal import *
#from libsoho.libctof import getionvel
from Libsoho import ctoflv1

#import time modules
from time import clock 
import datetime as dt

#import peakshape functions
from peakshape_functions import *


#outdated imports
#import fileinput
#import re
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.offsetbox import AnchoredText
#from collections import OrderedDict

iondict={"He+":[4,1,2],"He2+":[4,2,2],"C+":[12,1,6],"C2+":[12,2,6],"C3+":[12,3,6],"C4+":[12,4,6],"C5+":[12,5,6],"C6+":[12,6,6],"N+":[14,1,7], "N2+":[14,2,7],"N3+":[14,3,7],"N4+":[14,4,7],"N5+":[14,5,7],"N6+":[14,6,7],"N7+":[14,7,7],"O+":[16,1,8],"O2+":[16,2,8],"O3+":[16,3,8],"O4+":[16,4,8],"O5+":[16,5,8],"O6+":[16,6,8],"O7+":[16,7,8],"O8+":[16,8,8],"Ne+":[20,1,10],"Ne2+":[20,2,10],"Ne3+":[20,3,10],"Ne4+":[20,4,10],"Ne5+":[20,5,10],"Ne6+":[20,6,10],"Ne7+":[20,7,10],"Ne8+":[20,8,10],"Ne9+":[20,9,10],"Ne10+":[20,10,10],"Na+":[23,1,11],"Na2+":[23,2,11],"Na3+":[23,3,11],"Na4+":[23,4,11],"Na5+":[23,5,11],"Na6+":[23,6,11],"Na7+":[23,7,11],"Na8+":[23,8,11],"Na9+":[23,9,11],"Na10+":[23,10,11],"Na11+":[23,11,11],"Mg+":[24,1,12],"Mg2+":[24,2,12],"Mg3+":[24,3,12],"Mg4+":[24,4,12],"Mg5+":[24,5,12],"Mg6+":[24,6,12],"Mg7+":[24,7,12],"Mg8+":[24,8,12],"Mg9+":[24,9,12],"Mg10+":[24,10,12],"Mg11+":[24,11,12],"Mg12+":[24,12,12],"Al+":[27,1,13],"Al2+":[27,2,13],"Al3+":[27,3,13],"Al4+":[27,4,13],"Al5+":[27,5,13],"Al6+":[27,6,13],"Al7+":[27,7,13],"Al8+":[27,8,13],"Al9+":[27,9,13],"Al10+":[27,10,13],"Al11+":[27,11,13],"Al12+":[27,12,13],"Al13+":[27,13,13],"Si+":[28,1,14],"Si2+":[28,2,14],"Si3+":[28,3,14],"Si4+":[28,4,14],"Si5+":[28,5,14],"Si6+":[28,6,14],"Si7+":[28,7,14],"Si8+":[28,8,14],"Si9+":[28,9,14],"Si10+":[28,10,14],"Si11+":[28,11,14],"Si12+":[28,12,14],"Si13+":[28,13,14],"Si14+":[28,14,14],"S+":[32,1,16],"S2+":[32,2,16],"S3+":[32,3,16],"S4+":[32,4,16],"S5+":[32,5,16],"S6+":[32,6,16],"S7+":[32,7,16],"S8+":[32,8,16],"S9+":[32,9,16],"S10+":[32,10,16],"S11+":[32,11,16],"S12+":[32,12,16],"S13+":[32,13,16],"S14+":[32,14,16],"S15+":[
32,15,16],"S16+":[32,16,16],"Ar+":[40,1,18],"Ar2+":[40,2,18],"Ar3+":[40,3,18],"Ar4+":[40,4,18],"Ar5+":[40,5,18],"Ar6+":[40,6,18],"Ar7+":[40,7,18],"Ar8+":[40,8,18],"Ar9+":[40,9,18],"Ar10+":[40,10,18],"Ar11+":[40,11,18],"Ar12+":[40,12,18],"Ar13+":[40,13,18],"Ar14+":[40,14,18],"Ar15+":[40,15,18],"Ar16+":[40,16,18],"Ar17+":[40,17,18],"Ar18+":[40,18,18],"Ca+":[40,1,20],"Ca2+":[40,2,20],"Ca3+":[40,3,20],"Ca4+":[40,4,20],"Ca5+":[40,5,20],"Ca6+":[40,6,20],"Ca7+":[40,7,20],"Ca8+":[40,8,20],"Ca9+":[40,9,20],"Ca10+":[40,10,20],"Ca11+":[40,11,20],"Ca12+":[40,12,20],"Ca13+":[40,13,20],"Ca14+":[40,14,20],"Ca15+":[40,15,20],"Ca16+":[40,16,20],"Ca17+":[40,17,20],"Ca18+":[40,18,20],"Ca19+":[40,19,20],"Ca20+":[40,20,20],"Fe+":[56,1,26],"Fe2+":[56,2,26],"Fe3+":[56,3,26],"Fe4+":[56,4,26],"Fe5+":[56,5,26],"Fe6+":[56,6,26],"Fe7+":[56,7,26],"Fe8+":[56,8,26],"Fe9+":[56,9,26],"Fe10+":[56,10,26],"Fe11+":[56,11,26],"Fe12+":[56,12,26],"Fe13+":[56,13,26],"Fe14+":[56,14,26],"Fe15+":[56,15,26],"Fe16+":[56,16,26],"Fe17+":[56,17,26],"Fe18+":[56,18,26],"Fe19+":[56,19,26],"Fe20+":[56,20,26],"Fe21+":[56,21,26],"Fe22+":[56,22,26],"Fe23+":[56,23,26],"Fe24+":[56,24,26],"Fe25+":[56,25,26],"Fe26+":[56,26,26],"Ni+":[59,1,28],"Ni2+":[59,2,28],"Ni3+":[59,3,28],"Ni4+":[59,4,28],"Ni5+":[59,5,28],"Ni6+":[59,6,28],"Ni7+":[59,7,28],"Ni8+":[59,8,28],"Ni9+":[59,9,28],"Ni10+":[59,10,28],"Ni11+":[59,11,28],"Ni12+":[59,12,28],"Ni13+":[59,13,28],"Ni14+":[59,14,28],"Ni15+":[59,15,28],"Ni16+":[59,16,28],"Ni17+":[59,17,28],"Ni18+":[59,18,28],"Ni19+":[59,19,28],"Ni20+":[59,20,28],"Ni21+":[59,21,28],"Ni22+":[59,22,28],"Ni23+":[59,23,28],"Ni24+":[59,24,28],"Ni25+":[59,25,28],"Ni26+":[59,26,28],"Ni27+":[59,27,28],"Ni28+":[59,28,28]}

iondict_minimium={"He+":[4,1],"He2+":[4,2],"C4+":[12,4],"C5+":[12,5],"C6+":[12,6],"N5+":[14,5],"O5+"
:[16,5],"O6+":[16,6],"O7+":[16,7],"Ne8+":[20,8],"Mg10+":[24,10],"Si7+":[28,7],"Si8+":
[28,8],"Si9+":[28,9],"Si10+":[28,10],"Fe7+":[56,7],"Fe8+":[56,8],"Fe9+":[56,9],"Fe10+":[56,10],"Fe11+":[56,11],"Fe12+":[56,12],"Fe13+":[56,13],"Fe14+":[56,14]}
#dictionary with ion mass and charge for all ions visible by eye




class Ion(object):
    """
    class to handle the calibration parameters of different ions
    """

    def __init__(self,name,mass,charge,atomnumber,intensity=0,peakshape="gauss2d",print_ions=True):#peakshape=peakfunc name
        if name in iondict.keys():
            if intensity==None:
                if print_ions==True:
                    print("Initializing Ion %s: m=%i, q=%i, atomic number=%i,peakshape=%s, intensity=None"%(name,mass,charge,atomnumber,peakshape))				
            else :
                if print_ions==True:
                    print("Initializing Ion %s: m=%i, q=%i, atomic number=%i, peakshape=%s,intensity=%.2f"%(name,mass,charge,atomnumber,peakshape,intensity))				

        a0=2.00723e-1#time conversion in ns/ch 
        b0=-1.46909#time offset in ns		
        A0=0.5098
        ESSDnuc_0=A0*(0.5*1.66*10**(-27)*(70.5*10**(-3))**2/(1.6*10**(-16)))/((10**-9)**2)#constant in Energy ch/s

        self.name = name
        self.mass = mass
        self.charge = charge
        self.atomnumber = atomnumber
        self.steps = arange(0,116,1)*1.
        
        self.EAcc = step_to_E_acc(self.steps,self.charge)
        self.posTOF = tof(self.steps, self.mass, self.charge)
        self.sigTOF = tofsig(self.steps, self.mass, self.charge)
        self.sigTOF_left = tofsig_left(self.steps, self.mass, self.charge)
        self.kappaTOF_left=array([10.0]*len(self.steps))
        self.tailscale_TOF=tailscale_linear(tofch=self.posTOF,tail_grad=0.0035,tail_offset=0)#calibrated with iron and helium
        
        
        self.alpha_PHD=interpol_phd(self.atomnumber,self.posTOF)#PHD is not velocity-dependent for all ions, but helium
        self.ESSD_0=self.alpha_PHD*self.mass*ESSDnuc_0
        #self.posE = ESSD(self.posTOF,self.mass,self.atomnumber)
        self.posE = self.ESSD_0*(a0*self.posTOF+b0)**-2
        
        self.sigE = Esig(self.posE)
        self.sigE_kappa_moyalpar=Esig_kappa_moyalpar(self.posE)
        
        
        if self.mass==4:
            self.kappaTOF_left=array([1.5]*len(self.steps))
        if self.mass==12:
            self.kappaTOF_left=array([1.8]*len(self.steps))
        if self.mass==56:
            self.sigE_low=Esig_iron_low(self.posE)-1.#check!
            self.sigE_up=Esig_iron_up(self.posE)-2.
            self.Ekappa_low=zeros((len(self.steps)))+5.
            self.Ekappa_up=zeros((len(self.steps)))+2.5
        
        
        #from 26/02/2019 on,preliminary
        c=1/100.
        self.sigTOF_right=self.tofsig_rel(c,self.posTOF)*self.sigTOF_left
        self.sigTOF_left_He2=tofsig_left_He(self.posTOF)
        
        
        self.kappaTOF=kappa_linear(self.posTOF,tofrange=[200,600],kapparange=[3,0.1])
        
        self.kappaTOF_right=ones((len(self.steps)))
        self.counts=ones((len(self.steps)))
        self.intensity=intensity
        self.hyppars_1=ones((len(self.steps)))
        self.hyppars_2=zeros((len(self.steps)))
        self.tofch_min=zeros((len(self.steps)))
        
        self.pars=None
        self.peakfuncs=None
        self.ellmasks=None
        self.ints=None
        self.peakfuncs_countnormed=None

        #load empirical intensities from long-time fit. Caution: These intensities depend sensitively on the ions included in the long-time fits and their assumed peak shapes!
        #currently the following ions are included in the long-time fits (with gauss2d peakshape only!): All ions within "iondict_minimium".
        if intensity==None:			
            #print self.name,iondict_emp.keys()
            path="/home/asterix/janitzek/ctof/CTOF_response_simulation/"
            infile="ion_intensities_DOY150-220.dat"
            empion_file=open(path+infile,"r")
            empion_keys=empion_file.readlines()[1].split()[1:]
            ion_filekey="intensities_"+self.name
            if ion_filekey in empion_keys:
                ion_column=empion_keys.index(ion_filekey)				
                if print_ions==True:
                    print("Ion exists among the empirically found CTOF ions, therefore it is initialized with its step-dependent empirical intensity when no intensity is given.")
                    print("Ion file coulumn:", ion_column)
                    print("\n")
                empion_steps,empion_intensities=loadtxt(path+infile,skiprows=2,usecols=(0,ion_column),unpack=True)
                self.intensity=empion_intensities
            else:
                self.intensity=zeros((len(self.steps)))
                if print_ions==True:
                    print("WARNING: Ion %s exists within the CTOF calibration but is not among the empirically found CTOF ions, therefore it is initialized with zero intensity if no intensity is given."%(name))
        else:
                self.intensity=array([intensity]*len(self.steps))#intensity is a float number (only) in this case, and assumed to be constant for all steps!
        
        
        if peakshape=="gauss2d":
            self.peakfunc=gauss2d
            self.peakpars_names=array(["counts","posTOF","posE","sigTOF","sigE"])				
            self.peakpars_values=array([self.intensity,self.posTOF,self.posE,self.sigTOF,self.sigE]).T
            self.peakpars_varies=array([[True,False,False,False,False]]*len(self.steps))
            self.peakpars_bounds=array([[[0,None],[None,None],[None,None],[None,None],[None,None]]]*len(self.steps))
            
        if peakshape=="kappa_moyalpar_Easym": 
                                    
            if self.mass==56:
                self.peakfunc=kappa_moyalpar_hyp_2d_Ekappa_asym
                self.peakpars_names=array(["counts","posTOF","sigTOF","kappaTOF_left","tailscale_TOF","ESSD_0", "sigE_low","sigE_up", "Ekappa_low","Ekappa_up","hyppars_0","hyppars_1"])
                self.peakpars_values=array([self.intensity,self.posTOF,self.sigTOF_left_He2,self.kappaTOF_left,self.tailscale_TOF,self.tofch_min,self.ESSD_0,self.sigE_low,self.sigE_up, self.Ekappa_low,self.Ekappa_up,self.hyppars_1,self.hyppars_2]).T
                
                
            else:
                self.peakfunc=kappa_moyalpar_hyp_2d
                self.peakpars_names=array(["counts","posTOF","sigTOF","kappaTOF_left","tailscale_TOF","ESSD_0","sigE","hyppars_0","hyppars_1"])
                #self.peakpars_values=array([self.intensity,self.posTOF,self.sigTOF_left_He2,self.kappaTOF_left,self.tailscale_TOF,self.tofch_min, self.ESSD_0,self.sigE,self.hyppars_1,self.hyppars_2]).T
                self.peakpars_values=array([self.intensity,self.posTOF,self.sigTOF_left_He2,self.kappaTOF_left,self.tailscale_TOF,self.tofch_min, self.ESSD_0,self.sigE_kappa_moyalpar,self.hyppars_1,self.hyppars_2]).T
                
                self.peakpars_varies=array([[True,False,False,False,False,False,False,False,False,False]]*len(self.steps))
                self.peakpars_bounds=array([[[0,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]]*len(self.steps))
                



    def show_intensities(self):
        plt.figure()
        plt.plot(self.steps,self.intensity,marker="o",label="%s"%(self.name))
        plt.xlabel("CTOF E/q step")
        plt.ylabel("absolute intensity")
        plt.title("empirical ion peak intensity,\nobtained from 2D Gaussian long-time data fits (DOY 150-220 1996)")
        plt.legend(loc="upper right")
        plt.show()

    def create_peakfunc(self,step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99):
        
        t0=clock()
        f=self.peakfunc
        t1=clock()
        p=self.peakpars_values[step]
        t2=clock()		
        #print "p, tails:", p
        #print self.peakpars_values,step
        hg=f(p,xg,yg)
        t3=clock()
        
        if normalize_integral==True:
            norm=1./sum(hg)
            self.intensity[step]=self.intensity[step]*norm#think, whether this is what one wants!      
            hg=hg/float(sum(hg))
            if max(isnan(hg))==True:
                hg=zeros((len(hg)))
        
        if cutoff==True:
            mask_ion=self.create_peakint_mask(step,xg,yg,coverage_rel=coverage_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=False)
            hg=hg*mask_ion
        
        t4=clock()
        #print "intra_peakfunc run times:", t1-t0, t2-t1, t3-t2, t4-t3
        
        return hg

    def create_peakproj_scalable(self,step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99,height=None):
        hg=self.create_peakfunc(step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99)
        if height!=None:
            hg=float(height)/max(hg)*hg
        x,y=unique(xg),unique(yg)
        H=hg.reshape(len(x),len(y))
        tofproj_scaled=sum(H,axis=1)
        return tofproj_scaled


    def tofkappa_scalable(self,step,x,c,height,kappatof=100):
        
        #p=[height,self.posTOF[step],self.sigTOF_left[step],self.sigTOF_right[step],self.kappaTOF[step]]
        p=[height,self.posTOF[step],self.sigTOF_left[step],self.tofsig_rel(c,self.posTOF[step])*self.sigTOF_right[step],kappatof]
        h=abs(gausskappa(p,x))
        return h

    def tofsig_rel(self,c,tofch):
        tofrel=abs(c)*tofch-1.
        return tofrel

    def kappa(self,tofch):
        """
        is not allowed to become negative before tofch=550 (for step 116), or at least tofch 533 (for step90)
        """
        #tof=self.posTOF[step]
        kappa=-1./200*tofch+3.5#model_1
        #kappa=-1./150*tofch+3.5#model_2
        return kappa

    def gauss1d_scalable(self,step,x,height):
        p=[height,self.posTOF[step],self.sigTOF[step]]
        h=abs(gauss1d(p,x))
        return h
                    
    def create_peakmincounts_mask(self,step,xg,yg,countlevel_rel=0.1,Plot=False,CBlog=False,vmin=1e-10):
        hg=self.create_peakfunc(step,xg,yg,normalize_integral=False)
        min_abs=countlevel_rel*max(hg)
        mask_bool=(hg>=min_abs)
        mask=mask_bool.astype(float)
        hgmask=hg*mask
        
        if Plot==True:
                x,y=unique(xg),unique(yg)
                hmask=hgmask.reshape(len(x),len(y))
                fig, ax = plt.subplots(1,1)
                my_cmap = cm.get_cmap("Spectral_r",1024*16)
                my_cmap.set_under('w')
                if CBlog==True:
                        Cont1=ax.pcolor(x,y,hmask.T,cmap=my_cmap,norm=colors.LogNorm(vmin=vmin, vmax=max(ravel(hmask))))
                else:
                    Cont1=ax.pcolor(x,y,hmask.T,cmap=my_cmap,vmin=vmin,vmax=max(ravel(hmask)))
                ax.set_xlabel("tof [ch]")		
                ax.set_ylabel("energy [ch]")
                ax.set_title("modeled peakfunc relevant surface,\nion: %s, step: %i, min countlevel=%.1f"%(self.name,step,countlevel_rel))
                cb1 = fig.colorbar(Cont1)
                cb1.set_label("counts per bin")				
                plt.show()		
        return mask	    


    def create_peakint_mask(self,step,xg,yg,coverage_rel=0.99,N_sigE=3,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=False):
        hg=self.create_peakfunc(step,xg,yg,normalize_integral)
        x,y=unique(xg),unique(yg)
        h=hg.reshape(len(x),len(y))
        inds_sort=argsort(hg)[::-1]
        hg_sort=hg[inds_sort]
        
        mask_sort=zeros((len(xg))).astype("bool")
        S=0.
        i=0
        j=0
        while i<len(xg):
            S=S+hg_sort[i]
            if S/sum(hg)<coverage_rel:
                mask_sort[i]=True
            else:
                if j==0:
                    mask_sort[i]==True#fill current column to be over desired the coverage 
                    j+=1
            #print i,j,S/sum(hg),xg[i]
            i+=1
        mask=mask_sort[argsort(inds_sort)]
        
        hg_mask=hg*mask
        hmask=hg_mask.reshape(len(x),len(y))
        
        if Plot==True:
            #x,y=unique(xg),unique(yg)
            #print shape(hg_surf),shape(x),shape(y)
            #h_surf=hg_surf.reshape(len(x),len(y))
            fig, ax = plt.subplots(1,1)
            my_cmap = cm.get_cmap("Spectral_r",1024*16)
            my_cmap.set_under('w')
            if CBlog==True:
                    Cont1=ax.pcolor(x,y,hmask.T,cmap=my_cmap,norm=colors.LogNorm(vmin=vmin, vmax=max(ravel(hmask))))
            else:
                Cont1=ax.pcolor(x,y,hmask.T,cmap=my_cmap,vmin=vmin,vmax=max(ravel(hmask)))
            ax.set_xlabel("tof [ch]")		
            ax.set_ylabel("energy [ch]")
            ax.set_title("modeled peakfunc relevant surface,\nion: %s, step: %i, integral coverage=%.1f"%(self.name,step,coverage_rel))
            cb1 = fig.colorbar(Cont1)
            cb1.set_label("counts per bin")				
            plt.show()
        
        return mask	    


    def create_probfunc(self,peakshape,step,intensity=None):

        if intensity==None:
                intensity=self.get_intensity_emp(peakshape,step,norm=True)#to be done!
        P=concatenate([array([intensity]),self.peakpars_values[step][1:]])
        #return P

        if peakshape=="gauss2d":
            f=gauss2d
        elif peakshape=="gausskappa_hyp_2d_lm":
            f=gausskappa_hyp_2d_lm

        self.probfunc=cancel_funcpar(fitfunc=f,params=P)

        return self.probfunc

            
    def ion_pars(self):	
            self.pars=array([None]*len(self.steps))
            k=0
            while k<len(self.steps):
                steppars = Parameters_mod()
                i=0
                #return ion.peakpars_names, ion.peakpars_values,ion.peakpars_varies,ion.peakpars_bounds

                while i<len(self.peakpars_names):
                    steppars.add("%s_%s"%(self.peakpars_names[i],self.name[:-1]),
    value=self.peakpars_values[k][i], vary=self.peakpars_varies[k][i],
    min=self.peakpars_bounds[k][i][0],max=self.peakpars_bounds[k][i][-1])
                    i=i+1
                self.pars[k]=steppars
                k=k+1	
            return self.pars	

    def ion_sigmaenvs(self,xg,yg,N_sigmaTOF,N_sigmaE,Nb=20,ang=0):
            #self.xgs_valid=list()
            #self.ygs_valid=list()
            self.ellmasks=list()
            i=0		
            while i<len(self.steps):
                #create_sigma ellipses
                ellx,elly=ellipse(x0=self.posTOF[i],
    y0=self.posE[i], rx=N_sigmaTOF*self.sigTOF[i],ry=N_sigmaE*self.sigE[i],Nb=Nb,ang= ang)
                #get sigma ellipse mask in tof and energy
                verts=zip(ellx,elly)
                points=zip(xg,yg)
                ellmask=points_inside_poly(points,verts)
                #xg_valid,yg_valid=xg[ellmask],yg[ellmask]
                #self.xgs_valid.append(xg_valid)	
                #self.ygs_valid.append(yg_valid)
                self.ellmasks.append(ellmask)	
                i=i+1
            #self.xgs_valid,self.ygs_valid=array(self.xgs_valid),array(self.ygs_valid)	
            self.ellmasks=array(self.ellmasks)
            #return self.xgs_valid,self.ygs_valid,self.ellmasks
            return self.ellmasks

    def ion_peakfuncs(self,peakfunc,xg,yg):	
            self.peakfuncs=zeros((len(self.steps),len(xg)))
            ion_pars=self.ion_pars()
            k=0
            while k<len(self.steps):	
                param=ion_pars[k].get_values()
                self.peakfuncs[k]=peakfunc(p=param,x=xg,y=yg)
                k=k+1
            return self.peakfuncs
            
            
            
    def ion_peakintegrals_norm(self,peakfunc,xg,yg):	
            self.ints=zeros((len(self.steps),len(xg)))
            k=0
            while k<len(self.steps):
                xg_valid=xg[self.ellmasks[k]]
                yg_valid=yg[self.ellmasks[k]]
                param=self.pars[k].get_values()
                I=sum(peakfunc(p=param,x=xg_valid,y=yg_valid))
                self.ints[k]=zeros((len(xg)))+I
                k=k+1
            self.ints.astype(float)
            return self.ints

    def ion_peakfuncs_countnorm(self,peakfunc,xg,yg):#check whether fitfunc should end after N sigmas!
            ion_peakfuncs=self.ion_peakfuncs(peakfunc,xg,yg)
            ion_peakintegrals_norm=self.ion_peakintegrals_norm(peakfunc,xg,yg)
            self.peakfuncs_countnormed=ion_peakfuncs/ion_peakintegrals_norm
            return self.peakfuncs_countnormed


    def get_sigma_ellenvs(self, mask,xg,yg,hdata,step,N_sigmaTOF,N_sigmaE,
    Plot=False,linewidth=2.0,markerstyle="o",color="m"):
            """
            returns sigma ellipse for each ion and the corresponding 2d-bins of the ET-matrix (where the 
            lower left corner of the bin is within the ellipse contour). The included bins are called "sigma environment".
            It might be practical to substitute this method completely by its parts and mask as argument might be dropped.
            "calc_sigma_ellipses","plot_sigma_ellipses","get_sigmaenvs","plot_sigmaenvs" at some point.
            If Plot==True, both the ion ellipses and the environments are plotted.
            """
            ellipses_X,ellipses_Y=self.calc_sigma_ellipses(mask,xg,yg,hdata,step,N_sigmaTOF,N_sigmaE)
            xgs_valid,ygs_valid=self.get_sigmaenvs(mask,xg,yg,hdata,ellipses_X,ellipses_Y)
            if Plot==True:
                self.plot_sigma_ellipses(ellipses_X,ellipses_Y,linewidth=linewidth,color=color)
                #self.plot_sigmaenvs(xgs_valid,ygs_valid,markerstyle=markerstyle,color=color)
            return ellipses_X,ellipses_Y,xgs_valid,ygs_valid	
