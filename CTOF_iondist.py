"""
Class that contains the instrumental response model for a set of ion species. The response model is calculated as a superposition of the single ion species responses that are defined in CTOF_ion.py
Author: Nils Janitzek (2021)
"""

from CTOF_ion import Ion

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
from CTOF_ion import Ion, iondict, iondict_minimium
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




"""
3) IonDist data class 
"""				
class IonDist(object):
        """
        class to handle the fit of multiple ions
        """
        def __init__(self):
                self.Ions = []
                self.Ion_names=[]
                self.Ion_masses=[]
                self.Ion_charges=[]
                self.Ion_atomnumbers=[]
                self.pars=None
                self.peakfuncs=None
                self.ellmasks=None
                self.xgs_valid=None
                self.ygs_valid=None
                #self.ionlist=[]
                #self.ionlists_longtime_fit=[set(["O6+"]),set(["C4+","O6+"])]#test
                self.ionlists_longtime_fit=iondict_minimium.keys()


        def add_ion(self,name,peakshape="gauss2d",intensity=None,print_ions=True):

            """
            Adds an ion to the ion distribution. Only the ion name, intensity and peakshape have to be selected. Ion charge and mass are assigned automatically from the 				iondict dictionary.
            """
            #print name
            if name in iondict.keys():
                ion = Ion(name=name, mass=iondict[name][0], charge=iondict[name][1], atomnumber=iondict[name][2], intensity=intensity, peakshape=peakshape, print_ions=print_ions)
                if ion.name in self.Ion_names:
                    print("Ion already exists in iondist, therefore it cannot be added.")
                    print("\n")
                else:
                    self.Ions.append(ion)
                    self.Ion_names.append(ion.name)
                    self.Ion_masses.append(ion.mass)
                    self.Ion_charges.append(ion.charge)
                    self.Ion_atomnumbers.append(ion.atomnumber)
            else:
                print("WARNING: Ion %s does not exist within the CTOF calibration, please only select ions that are listed in iondict.keys()"%(name))
                print("\n")

				
        def add_ionlist(self,names,peakshapes,intensities=None,print_ions=False):		
            """
            Adds a list of ions to the ion distribution. For each ion name, intensity and peakshape have to be selected. Ion charge and mass are assigned automatically from the iondict dictionary.
            Example: IonDist.add_ionlist(names=["C5+","O6+"],intensities=[50,100],peakshapes=["gauss2d","gausskappa_hyp_2d_lm"])
            """
            for name in names:
                i=names.index(name)
                if peakshapes=="gauss2d" or peakshapes=="gausskappa_hyp_2d_lm" or peakshapes=="kappa_moyalpar" or peakshapes=="kappa_moyalpar_Easym":
                    if intensities[i]==None:				
                        self.add_ion(name,peakshapes,print_ions=print_ions)
                    else:
                        #print "intensities test:",intensities, i
                        self.add_ion(name,peakshapes,intensities[i],print_ions=print_ions)						
                    #if name not in self.ionlist:
                    #	self.ionlist.append(name)
                else:	
                    print("Error, no valid peakshape selected!")



        def add_ionlist_stepdep(self,step,names,peakshapes,intensities=None,stepcovranges=None,print_ions=False):
            if stepcovranges!=None:
                i=0
                for name in names:
                    ion_stepcovs=arange(stepcovranges[i][0],stepcovranges[i][-1],1)
                    if step in ion_stepcovs:
                        if intensities==None:				
                            self.add_ion(name,peakshapes,print_ions=print_ions)
                        else:
                            print("intensities test:",intensities, i)
                            self.add_ion(name,peakshapes,intensities[i],print_ions=print_ions)
                    else:
                        pass		
                    i=i+1								
            else:
                self.add_ionlist(names,peakshapes,intensities=None,print_ions=False)
            self.check_ions() 
            return True		


        def check_ions(self):
            for Ion in self.Ions:
                print(Ion.name)


        def remove_ion(self,name):
            for Ion in self.Ions:
                if Ion.name==name:
                    self.Ions.remove(Ion)
                    self.Ion_names.remove(Ion.name)
                    self.Ion_masses.remove(Ion.mass)
                    self.Ion_charges.remove(Ion.charge)
                    self.Ion_atomnumbers.remove(Ion.atomnumber)

        def remove_ionlist(self,names):
            for name in names:
                self.remove_ion(name)

					
        def return_ionnames(self):
            l=[]
            for Ion in self.Ions:
                l.append(Ion.name)
            return l				




        def create_xygrid(self,Xrange=[100,600],Yrange=[1,120],Xbin=2,Ybin=2):		
                """
                Creates a flattened grid (xg,yg) from a given xrange,yrange and the respective binsizes xbin,ybin. The lowest meaningful binsize for CTOF is binx=biny=2 due to binning errors on smaller scale within the instrument.	
                """
                x=arange(Xrange[0],Xrange[1],Xbin)		
                y=arange(Yrange[0],Yrange[1],Ybin)
                ygrid,xgrid=meshgrid(y,x)	
                xg,yg=ravel(xgrid),ravel(ygrid)
                return xg,yg

        def create_params(self,step):
                """
                Creates a Parameters object (from "lmfit" library) containing all peak shape parameters of all ions included in the current ion distribution for the selected energy-per charge step. This object will be used to create the model distribution and random distribution. It can be also used for fitting random distributions. (For more information see "lmfit" library).	
                """		
                params=Parameters()
                for ion in self.Ions:
                    j=0
                    while j<len(ion.peakpars_names):
                        params.add("%s_%s"%(ion.peakpars_names[j],ion.name[:-1]),value=ion.peakpars_values[step][j], vary=ion.peakpars_varies[step][j],min=ion.peakpars_bounds[step][j][0],max=ion.peakpars_bounds[step][j][-1])
                        j=j+1
                return params

        def create_modeldist(self,params,xg,yg):
                """
                Creates a model peak distribution H(TOF,ESSD) for a selected set of parameters, stored in the Parameter object params. The grid on which H is calculated is given by xg,yg.  
                """
                Hg=zeros((len(xg)))
                for ion in self.Ions:
                    i=self.Ions.index(ion)
                    P=zeros((len(ion.peakpars_names)))
                    j=0
                    while j<len(ion.peakpars_names):	
                        P[j]=params["%s_%s"%(ion.peakpars_names[j],ion.name[:-1])].value
                        j=j+1
                    hg=ion.peakfunc(P,xg,yg)
                    Hg=Hg+hg
                return Hg

        def plot_modeldist(self,step,xg,yg,Hg,CBlog=False):
                """
                Plots the model peak distribution H as 2-dimensional color plot on the grid xg,yg. 
                """
                x,y=unique(xg),unique(yg)
                z=Hg.reshape(len(x),len(y))
                fig = pylab.figure()
                ax = fig.gca()
                if CBlog==True:
                    Cont=ax.pcolor(x,y,z.T,norm=colors.LogNorm(vmin=1, vmax=max(ravel(z))))
                else:	
                    Cont=ax.pcolor(x,y,z.T,vmin=1, vmax=max(ravel(z)))
                Cont.cmap.set_under('w')
                cb = fig.colorbar(Cont)
                cb.set_label("Counts per bin")
                ax.set_title("model distribution, step: %i"%(step)) 
                ax.set_xlabel("TOF [ch]")
                ax.set_ylabel("Energy [ch]")
                plt.show()	


        def get_modeldist(self,step,ionlist,peakshapes="gauss2d",intensities=None,Xrange=[100,600],Yrange=[1,120],Xbin=2,Ybin=2,Plot=True,CBlog=False,remove_ions=True,print_ions=True):
                """
                Fuction:
                Model peak distribution H(TOF,ESSD) for the selected ESA step and ions.		

                Input:
                step: selected ESA step as integer. 
                ionlist: selected set of ions as list of strings.
                peakshapes: can be selected as string "gauss2d" or "gausskappa_hyp_2d_lm". For each ion an individual peakshape can be selected if the input is an array of strings with len(ionlist).
                intensities: float or integer array of len(ionlist), if None is selected, the ions are initialized with their empiric intensities taken from CTOF longtime data fits if they are in iondict_emp. If they are in iondict but not in iondict_emp they are initilized with zero intensity if no intensity is given. Ions that are not even in iondict, are not accepted.  
                Xrange,Yrange: determines the CTOF TOF and Energy channels as list of lowest channel and highest channel [ch_min,ch_max], respectively.
                Xbin,Ybin: TOF and Energy channel binnings given as integers.
                Plot: if True function is plotted.
                remove_ions: if True ionlist is emptied after each method call.
                print_ions: selected ions are printed on screen with their initialized parameters.
                CBlog: if True the colorbar is scaled logarithmically, this setting should be the standard setting for the case that the ionlist contains ion species with big differences in abundance among each other at a given step.

                Output: 
                arrays xg,yg,Hg: xy-grid on which the model is defined, model intensity function

                Examples: self.get_modeldist(step=50,ionlist=["He2+","O6+","Fe9+"],peakshapes=["gauss2d","gauss2d","gausskappa_hyp_2d_lm"],intensities=[None,1000,None],Plot=True)
                self.get_modeldist(step=70,ionlist=sort(iondict_emp.keys()),peakshapes=["gausskappa_hyp_2d_lm"]*len(iondict.keys()),intensities=[None]*len(iondict.keys()),Plot=True,CBlog=True)
                """

                if type(ionlist)==ndarray:
                    ionlist=list(ionlist)
                if type(peakshapes)==ndarray:
                    peakshapes=list(peakshapes)
                if type(intensities)==ndarray:
                    intensities=list(intensities)

                if remove_ions==True:
                    self.remove_ion("all")
                self.add_ionlist(names=ionlist,peakshapes=peakshapes,intensities=intensities,print_ions=print_ions)
                P=self.create_params(step=step)
                xg,yg=self.create_xygrid(Xrange=Xrange,Yrange=Yrange,Xbin=Xbin,Ybin=Ybin)
                Hg=self.create_modeldist(params=P,xg=xg,yg=yg)
                if Plot==True:
                    self.plot_modeldist(step=step,xg=xg,yg=yg,Hg=Hg,CBlog=CBlog)
                if remove_ions==True:
                    self.remove_ion("all")
                return xg,yg,Hg	









        def posclean_ionlist(self,step,tofrange,Erange): 
            print(" len(self.Ions)", len(self.Ions))
            ionlist_removed=[]
            Ionlist_removed=[]
            i=0
            for ion in self.Ions:
                print(ion.name, ion.posTOF[step], tofrange[0],tofrange[-1],i)
                if (ion.posTOF[step]<tofrange[0]) or (ion.posTOF[step]>=tofrange[-1]-2) or (ion.posE[step]<Erange[0]) or (ion.posE[step]>=Erange[-1]-2):#-2ch is taken to cut out ions on the edge (avoiding rounding effects)!
                    ionlist_removed.append(ion.name)
                    Ionlist_removed.append(ion)
                    print("removed ion:",ion.name,ion.posTOF[step],ion.posE[step])
                    #self.remove_ion(ion.name)
                i=i+1	
            self.remove_ionlist(ionlist_removed)	
            if len(ionlist_removed)>0:
                A=[]
                for ion in Ionlist_removed:
                    a=[step, ion.name, ion.posTOF[step], tofrange[0],tofrange[-1]-2, ion.posE[step], Erange[0],Erange[-1]-2]  
                    A.append(a)
                return A

        def iondist_pars(self):	
                self.pars=array([None]*len(self.Ions[0].steps))
                k=0
                while k<len(self.Ions[0].steps):
                        steppars = Parameters_mod()
                        for ion in self.Ions:
                            print("ion name:",ion.name)
                            i=0
                            #return ion.peakpars_names, ion.peakpars_values,ion.peakpars_varies,ion.peakpars_bounds
                            while i<len(ion.peakpars_names):
                                steppars.add("%s_%s"%(ion.peakpars_names[i],ion.name[:-1]),
        value=ion.peakpars_values[k][i], vary=ion.peakpars_varies[k][i],min=ion.peakpars_bounds[k][i][0],max=ion.peakpars_bounds[k][i][-1] )

                                i=i+1
                        print("ion parameters", steppars)
                        self.pars[k]=steppars
                        k=k+1	
                return self.pars	


        def sortout_ionpos(self,tofpos_distmin=5,Epos_distmin=5,steps_checkrange=[0,115]): 
            ionlist_tooclose=[]
            tofpos=zeros((len(self.Ions),116))
            Epos=zeros((len(self.Ions),116))
            for i,ion in enumerate(self.Ions):
                tofpos[i]=ion.posTOF
                Epos[i]=ion.posE
            for i,ion in enumerate(self.Ions):
                tofpos_diff=abs(tofpos-ion.posTOF)
                tofpos_diff[i]=tofpos_diff[i]+1e4
                tofpos_diffmin=amin(tofpos_diff,axis=1)
                #return ion.posTOF,tofpos,tofpos_diffmin
                Epos_diff=abs(Epos-ion.posE)
                #Epos_diff[i]=Epos_diff[i]+1e4
                Epos_diffmin=amin(Epos_diff,axis=1)
                
                dm=(tofpos_diffmin<tofpos_distmin)*(Epos_diffmin<Epos_distmin)
                ions_tooclose=array(self.return_ionnames())[dm]
                if ions_tooclose.shape[0]>0:
                    ionlist_tooclose.append((ion.name,ions_tooclose))
            
            return ionlist_tooclose
            

        def iondist_sigmaenvs(self,xg,yg,N_sigmaTOF,N_sigmaE,Nb=20,ang=0):
                #self.xgs_valid=list()
                #self.ygs_valid=list()
                self.ellmasks=list()
                for ion in self.Ions:		
                    #xgs_valid,ygs_valid,ellmasks=ion.ion_sigmaenvs(xg=xg,yg=yg,N_sigmaTOF=N_sigmaTOF,
                    #N_sigmaE=N_sigmaE,Nb=Nb,ang=ang)
                    ellmasks=ion.ion_sigmaenvs(xg=xg,yg=yg,N_sigmaTOF=N_sigmaTOF,N_sigmaE=N_sigmaE,Nb=Nb,ang=ang)
                    #return self.xgs_valid
                    #self.xgs_valid.append(xgs_valid)
                    #self.ygs_valid.append(ygs_valid)
                    self.ellmasks.append(ellmasks)
                #self.xgs_valid=array(self.xgs_valid)
                #self.ygs_valid=array(self.xgs_valid)
                self.ellmasks=array(self.ellmasks)
                #self.xgs_valid=self.xgs_valid.transpose(1,0)
                #self.ygs_valid=self.ygs_valid.transpose(1,0)
                self.ellmasks=self.ellmasks.transpose(1,0,2)
                #return self.xgs_valid,self.ygs_valid,self.ellmasks	
                return self.ellmasks
                

        def iondist_peakfuncs(self,peakfunc,xg,yg):				
                self.peakfuncs=zeros((len(self.Ions),len(self.Ions[0].steps),len(xg)))
                for ion in self.Ions:
                    i=self.Ions.index(ion)
                    self.peakfuncs[i]=ion.ion_peakfuncs(peakfunc,xg,yg)
                    
                self.peakfuncs=transpose(self.peakfuncs,(1,0,2))
                return self.peakfuncs	


                
        def iondist_peakintegrals_norm(self,peakfunc,xg,yg):	
                self.ints=zeros((len(self.Ions),len(self.Ions[0].steps),len(xg)))
                for ion in self.Ions:
                    i=self.Ions.index(ion)
                    ion.ion_pars()
                    ints=ion.ion_peakintegrals_norm(peakfunc,xg,yg)
                    self.ints[i]=ints
                self.ints=self.ints.transpose(1,0,2)
                return self.ints
                

        def iondist_peakfuncs_countnorm(self,peakfunc,xg,yg):
                #self.peakfuncs_countnorm=zeros((len(self.Ions),len(self.Ions[0].steps),len(xg)))
                iondist_peakfuncs=self.iondist_peakfuncs(peakfunc,xg,yg)
                iondist_peakintegrals_norm=self.iondist_peakintegrals_norm(peakfunc,xg,yg)
                self.peakfuncs_countnorm=iondist_peakfuncs/iondist_peakintegrals_norm
                return self.peakfuncs_countnorm
                    

        def create_peakfunc_allions(self,step,xg,yg,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.01,normalize_integrals=False):
        
            hg_allions=zeros((len(xg)))
            for ion in self.Ions:
                hg_ion=ion.create_peakfunc(step,xg,yg,normalize_integral=normalize_integrals)
                if ion_peakbounds=="coverage":
                    mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=coverage_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                elif ion_peakbounds=="countlevel": 
                    mask_ion=ion.create_peakmincounts_mask(step,xg,yg,countlevel_rel=countlevel_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                    #return mincounts_mask_ion
                hg_ion=hg_ion*mask_ion
                hg_allions=hg_allions+hg_ion
                #print ion.name, max(hg_allions)
                
            return hg_allions
        
        def create_peakfunc_scalable_allions(self,step,heights,xg,yg,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.01,normalize_integrals=False):
            
            t0=clock()
            
            hg_allions=zeros((len(xg)))
            i=0
            for ion in self.Ions:
                hg_ion=heights[i]*ion.create_peakfunc(step,xg,yg,normalize_integral=normalize_integrals)/max(ion.create_peakfunc(step,xg,yg,normalize_integral=normalize_integrals))
                print(ion.name, max(hg_ion))
                if ion_peakbounds=="coverage":
                    mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=coverage_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                elif ion_peakbounds=="countlevel": 
                    mask_ion=ion.create_peakmincounts_mask(step,xg,yg,countlevel_rel=countlevel_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                    #return mincounts_mask_ion
                hg_ion=hg_ion*mask_ion
                hg_allions=hg_allions+hg_ion
                i=i+1
            
            t1=clock()
            
            print("fitfunc runtime:", t1-t0)
            
            return hg_allions
        
		
		
		
        def create_peakfuncs_norm(self,step,xg,yg,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.001):
        
            t0=clock()
            hg_allions_norm=zeros((len(self.Ions),len(xg)))
            i=0
            for ion in self.Ions:
                #print "len(ionlist)",len(self.Ions)
                #print "ion name test",ion.name, ion.posTOF[step],ion.posE[step]
                t00=clock()
                hg_ion=ion.create_peakfunc(step,xg,yg,normalize_integral=False)
                
                if ion_peakbounds=="coverage":
                    mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=coverage_rel,Plot=False,CBlog=False,vmin=1e-10)
                elif ion_peakbounds=="countlevel": 
                    mask_ion=ion.create_peakmincounts_mask(step,xg,yg,countlevel_rel=countlevel_rel,Plot=False,CBlog=False,vmin=1e-10)#much faster
                t02=clock()
                
                hg_allions_norm[i]=hg_ion*mask_ion/float(sum(hg_ion*mask_ion))
                i=i+1
            t1=clock()
            return hg_allions_norm
        
        
        
        
        def create_peakproj_scalable_allions(self,step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99,heights=None):
            x=unique(xg)
            tofproj_allions=zeros((len(x)))
            i=0
            for ion in self.Ions:
                if heights!=None:
                    #print "yes"
                    #print heights,shape(heights),i,self.Ions[i].name,len(self.Ions)
                    tofproj_ion=ion.create_peakproj_scalable(step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99,height=heights[i])		
                else:
                    #print "no"
                    tofproj_ion=ion.create_peakproj_scalable(step,xg,yg,normalize_integral=False,cutoff=False,coverage_rel=0.99,height=None)			
                tofproj_allions=tofproj_allions+tofproj_ion
                i=i+1
            return tofproj_allions
        
        #def tofkappa_allions_scalable(self,step,x,heights,kappatof=100):
        def tofkappa_allions_scalable(self,step,x,c,heights,kappatof=100):
            H=zeros((len(x)))
            i=0
            for ion in self.Ions:
                #h=ion.tofkappa_scalable(step,x,c=heights[-1],height=heights[i],kappatof=100)
                h=ion.tofkappa_scalable(step,x,c,height=heights[i],kappatof=ion.kappa(step))
                H=H+h
                i=i+1
            return H	
                
        
        def gauss1d_allions_scalable(self,step,x,heights):
            H=zeros((len(x)))
            i=0
            for ion in self.Ions:
                h=ion.gauss1d_scalable(step,x,height=heights[i])
                H=H+h
                i=i+1
            return H	
                
            
        
        def create_mincounts_mask(self,step,xg,yg,mincounts_rel=0.1,Plot=False,CBlog=False,vmin=1e-10):
        
            mincounts_mask_allions=zeros((len(xg)),dtype=bool)
            for ion in self.Ions:
                mincounts_mask_ion=ion.create_peakmincounts_mask(step,xg,yg,mincounts_rel=mincounts_rel,Plot=Plot,CBlog=CBlog,vmin=vmin)
                mincounts_mask_allions=mincounts_mask_allions+mincounts_mask_ion
            mincounts_mask_allions=mincounts_mask_allions.astype(float)	
            return mincounts_mask_allions
        
        
        def show_intensities(self):
            plt.figure()
            for ion in self.Ions:
                plt.plot(ion.steps,ion.intensity,marker="o",label="%s"%(ion.name))
            plt.xlabel("CTOF E/q step")
            plt.ylabel("absolute intensity")
            plt.title("empirical ion peak intensities,\nobtained from 2D Gaussian long-time data fits (DOY 150-220 1996)")
            plt.legend(loc="upper right")
            plt.show()
                            
        def scale_intensities(self,scaling):
            i=0
            for ion in self.Ions: 
                ion.intensity=scaling[i]*ion.intensity
                ion.peakpars_values.T[0]=ion.intensity
                #return ion.peakpars_values
                print("max. scaled ion intensities:", ion.name, max(ion.intensity))
                i+=1
                
				
        def create_relionprobs(self,steps,xg,yg,Plot=False,ionlist_plot=[],plot_maxrelcont=False,CBlog=False,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.01,normalize_integrals=False):         

            t0=clock()

            #return set(self.Ion_names), self.ionlists_longtime_fit
            #if set(self.Ion_names) in self.ionlists_longtime_fit:#include peakshape as well, this is the proper way to treat all fitted ions as ensemble, beacause stricly speaking, no ion intensity can be treated seperately! 
            if min(in1d(array(self.Ion_names),array(self.ionlists_longtime_fit)))==True:#check whether all selected ions were fitted
            
                x,y=unique(xg),unique(yg)
                hg_rel=zeros((len(steps),len(self.Ion_names),len(xg)))
                k=0
                for step in steps:
                    
                    t01=clock()
                    
                    #print "step:",step
                    hg_allions=self.create_peakfunc_allions(step,xg,yg,ion_peakbounds=ion_peakbounds,coverage_rel=coverage_rel,countlevel_rel=countlevel_rel,normalize_integrals=normalize_integrals)
                    #return hg_allions
                    self.ionprobs_step_all=hg_allions#test
                    H_all=hg_allions.reshape(len(x),len(y))
                    t02=clock()
                    
                    if Plot==True:
                        fig, ax = plt.subplots(1,1)
                        my_cmap = cm.get_cmap("Spectral_r",1024*16)
                        my_cmap.set_under('w')
                        if CBlog==True:
                                Cont1=ax.pcolor(x,y,H_all.T,cmap=my_cmap,norm=colors.LogNorm(vmin=1e-5, vmax=max(ravel(H_all.T))))
                        else:
                            Cont1=ax.pcolor(x,y,H_all.T,cmap=my_cmap,vmin=1e-5)
                        ax.set_xlabel("tof [ch]")		
                        ax.set_ylabel("energy [ch]")
                        ax.set_title("fitted long-time counts, all ions, step: %i"%(step))
                        cb1 = fig.colorbar(Cont1)
                        cb1.set_label("counts per bin")				
                    
                    hg_allions[hg_allions==0]=-1#treating zeros in denominator
                    
                    hg_rel_step=zeros((len(self.Ion_names),len(xg)))
                    
                    i=0
                    for ion in self.Ions:
                        hg_ion=ion.create_peakfunc(step,xg,yg,normalize_integrals)
                        if ion_peakbounds=="coverage": 
                            mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=coverage_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                        elif ion_peakbounds=="countlevel":
                            mask_ion=ion.create_peakmincounts_mask(step,xg,yg,countlevel_rel=countlevel_rel,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=normalize_integrals)
                        hg_ion=hg_ion*mask_ion
                            
                        hg_ionrel_step=hg_ion/hg_allions
                        hg_ionrel_step[hg_allions<0]=0#treating zeros in denominator
                        
                                                
                        if Plot==True:
                            if ion.name in ionlist_plot: 
                                H=hg_ionrel_step.reshape(len(x),len(y))
                                H_ion=hg_ion.reshape(len(x),len(y))
                            
                                """
                                fig, ax = plt.subplots(1,1)
                                my_cmap = cm.get_cmap("Spectral_r",1024*16)
                                my_cmap.set_under('w')
                                if CBlog==True:
                                    Cont1=ax.pcolor(x,y,H_ion.T,cmap=my_cmap,norm=colors.LogNorm(vmin=1e-3, vmax=max(ravel(H_ion.T))))
                                else:
                                    Cont1=ax.pcolor(x,y,H_ion.T,cmap=my_cmap,vmin=1e-3)
                                ax.set_xlabel("tof [ch]")		
                                ax.set_ylabel("energy [ch]")
                                ax.set_title("fitted long-time counts, ion: %s step: %i"%(ion.name,step))
                                cb1 = fig.colorbar(Cont1)
                                cb1.set_label("counts per bin")				
                                """

                                fig, ax = plt.subplots(1,1)
                                my_cmap = cm.get_cmap("Spectral_r",1024*16)
                                my_cmap.set_under('w')
                                if CBlog==True:
                                    Cont1=ax.pcolor(x,y,H.T,cmap=my_cmap,norm=colors.LogNorm(vmin=1e-3, vmax=max(ravel(H.T))))
                                else:
                                    Cont1=ax.pcolor(x,y,H.T,cmap=my_cmap,vmin=1e-3)
                                ax.set_xlabel("tof [ch]")		
                                ax.set_ylabel("energy [ch]")
                                ax.set_title("relative count contribution, ion: %s step: %i"%(ion.name,step))
                                cb1 = fig.colorbar(Cont1)
                                cb1.set_label("rel. probability per bin")				

                        hg_rel_step[i]=hg_ionrel_step
                        i+=1
                    self.ionprobs_step=hg_rel_step
                    
                    hg_rel[k]=hg_rel_step  
                    k+=1  
                    
                    t03=clock()
                    #print "runtime (for create_peakfunc_allion) per step:", t02-t01
                    #print "runtime (for create_peakfunc) per step:", t03-t02
                    
                    #print "total runtime per step:", t03-t01
                
                if plot_maxrelcont==True:
                    
                    fig, ax = plt.subplots(1,1)
                    meanionconts=zeros((len(ionlist_plot),len(steps)))
                    i=0
                    for ion in self.Ions:
                        
                        if ion.name in ionlist_plot:
                            #print ion.name
                            
                            k=0
                            h_rel=zeros((len(steps)))
                            while k<len(steps):
                                #return (where(xg==tofmeans_grid[k])[0]),(where(yg==Emeans_grid[k])[0]) 
                                tofmean=tof(steps[k],ion.mass,ion.charge)
                                Emean=ESSD(tofmean,ion.mass,ion.atomnumber)
                                tofmean_grid=find_nearest(xg,tofmean)
                                Emean_grid=find_nearest(yg,Emean)
                                h_ind=where((xg==tofmean_grid)*(yg==Emean_grid))[0]
                                if len(h_ind)>0:
                                    h_rel[k]=hg_rel[k][i][h_ind]
                                else:
                                    h_rel[k]=-1#if calculated mean position is not within the grid
                                k=k+1 
                        
                            meanionconts=h_rel
                            #return h_rel
                            ax.plot(steps,meanionconts,marker="o",label="%s"%(ion.name))
                            ax.legend()
                            ax.set_xlabel("E/q step")
                            ax.set_ylabel("relat. contribution in the ion position")
                        i+=1
                        
                t1=clock()
                
                
                    
                #print("total method runtime:", t1-t0)
                
                self.ionprobs_step=hg_rel[0]
                    
                return hg_rel
        
            
        #def iterate_ioncount_probs(self,termcond=0.01,Plot=True):
        def iterate_ioncount_probs(self,step,xg,yg,hdata_timestamp,termcond=0.01, Plot=True,CBlog=False,N_iter=100, Plot_ioncount_evolution=True,random_data=True,gen_data=True,random_counts=None):
            
            
            """
            -get ion-count contribution	
            -calculate new current probabilities for the different ions out of that, use only matrix operations for this step to keep runtime short
            -iterate this step until convergence
            -save final ion-count contribution (and previous steps to see convergence history)
            -think in which class this method should be integrated best
            
            -comment objects:"""

            N = random_counts
            self.P_ions=zeros((len(self.Ions),len(xg)))
            PC_ions=zeros((len(self.Ions),len(xg)))
            self.W_ions=zeros((len(self.Ions),len(xg)))
            if gen_data:
                self.r_data=zeros(len(xg))
            for i,ion in enumerate(self.Ions):
                mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=0.99,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=True)
                print(mask_ion)
                self.P_ions[i]=ion.create_peakfunc(step,xg,yg,True)*mask_ion
                self.P_ions[i]/=sum(self.P_ions[i])
                print(i,sum(self.P_ions[i]))
                        
            Wts=1+random.random(len(self.Ions))*99
            Psum=zeros(self.P_ions[0].shape)
            for i in range(len(self.Ions)):
                Psum+=self.P_ions[i]*Wts[i]
            #Psum=sum(self.P_ions,axis=0)
            Psum[Psum==0]=1.
            for i in range(self.P_ions.shape[0]):
                #self.W_ions[i]=self.P_ions[i]/Psum#sum(self.P_ions,axis=0)#, start c_i= 1 for 0th step,  TODO: only calculate the sum once
                self.W_ions[i]=Wts[i]*self.P_ions[i]/Psum

            if gen_data:
                for i in range(self.P_ions.shape[0]):
                    self.r_data+=random_choice(N[i],self.P_ions[i])
            if not(random_data):
                self.M_t=hdata_timestamp
            else:
                self.M_t=self.r_data
                print("sum(self.M_t):",sum(self.M_t))
            mask=sum(self.W_ions,axis=0)>0
            print(sum(self.M_t[mask]))

            Counts=[]
            for n in range(N_iter):#TODO find meaningful number of iterations or termination condition
                counts=sum(self.W_ions*self.M_t.astype(float),axis=1)
                Counts.append(counts)
                for i in range(self.P_ions.shape[0]):
                    PC_ions[i]=self.P_ions[i]*counts[i]
                PCsum=sum(PC_ions,axis=0)
                PCsum[PCsum==0]=1.
                for i in range(self.P_ions.shape[0]):	
                    self.W_ions[i]=PC_ions[i]/PCsum#sum(PC_ions,axis=0)
            
            Counts=array(Counts)
                
            if Plot_ioncount_evolution==True:		
                plt.figure()
                i=0
                while i<len(self.Ions):
                    print(len(Counts.T[i]))
                    plt.plot(Counts.T[i],label=self.Ion_names[i])	
                    plt.xlabel("N_iter")
                    plt.ylabel("calc. total counts per ion")
                    plt.legend()
                    i+=1
            
            return Counts
        
        
        #def iterate_ioncount_probs(self,termcond=0.01,Plot=True):
        def iterate_ioncount_probs_vectorized(self,steps,xg,yg,hdata_timestamp,peak_coverage_rel=0.99,W_start=None,termcond=0.01,Plot=True,CBlog=False,N_iter=100,Plot_ioncount_evolution=True,random_data=True,gen_data=True,random_counts=None):
            
            """
            -get ion-count contribution	
            -calculate new current probabilities for the different ions out of that, use only matrix operations for this step to keep runtime short
            -iterate this step until convergence
            -save final ion-count contribution (and previous steps to see convergence history)
            -think in which class this method should be integrated best
            
            -comment objects:"""

            t0=clock()
            #Pi=I.iondist_peakfuncs(peakfunc=gauss2d,xg=xg,yg=yg)[0]#could be used later for initializing peak shapes

            #inititalizing the peak shapes, has to be done only once for analyzing the whole measurement period DOY150-220 1996
            P=zeros((len(steps),len(self.Ions),len(xg)))
            for k,step in enumerate(steps):
                for i,ion in enumerate(self.Ions):
                    P[k][i]=ion.create_peakfunc(step=step,xg=xg,yg=yg,normalize_integral=True,cutoff=True,coverage_rel=peak_coverage_rel)			
            
            t1=clock()
            
            #inititalizing weights, has to be done only once for analyzing the whole measurement period DOY150-220 1996
            if W_start==None:
                renormalize_intensities=True#work around, generalize later
            else:	
                renormalize_intensities=False		
            W=self.create_relionprobs(steps=steps,xg=xg,yg=yg,Plot=False,ionlist_plot=[],plot_maxrelcont=False,CBlog=False,ion_peakbounds="coverage",coverage_rel=peak_coverage_rel,countlevel_rel=0.01,normalize_integrals=renormalize_intensities)
            
            t2=clock()

            #start loop over all steps
            Counts=zeros((len(steps),N_iter,len(self.Ions)))
            for k,step in enumerate(steps):
                
                Counts_step=zeros((N_iter,len(self.Ions)))
                
                #get step-specific normalized peak shapes
                P_step_unnorm=P[k]
                P_step_normfactor=sum(P_step_unnorm,axis=1).astype("float")
                P_step=P_step_unnorm/P_step_normfactor[:,newaxis]
                return P_step
                
                #get start weigths (can be step-specific, but should not matter) 
                W_step_j=W[k]	
                
                hdata_step=hdata_timestamp[0]#generalize later!	
                
                #start counts iteration	for all ions simultaneously 
                for j in range(N_iter):
                    
                    #calculate counts
                    Counts_step_j=sum(W_step_j*hdata_step,axis=1)
                    
                    #calculate weights
                    P_rescaled=(P_step.T*Counts_step_j).T
                    P_rescaled_normfactor=sum(P_rescaled,axis=0).astype("float")
                    P_rescaled_normfactor[P_rescaled_normfactor==0]=1.#avoiding nans outside the generalized sigma environments
                    W_step_j=P_rescaled/P_rescaled_normfactor
                    
                    Counts_step[j]=Counts_step_j
                    
                    t3=clock()
                
                print(t1-t0)
                print(t2-t1)
                print(t3-t2)

                return Counts_step
                
            
                

        def iterate_ioncount_probs_vectorized_intime(self,steps,xg,yg,hdata,peak_coverage_rel=0.99,W_start=None,termcond=0.01,Plot=True,CBlog=False,N_iter=100,Plot_ioncount_evolution=True,random_data=True,gen_data=True,random_counts=None):
            
            """
            -comment objects:
            """

            t0=clock()
            #Pi=I.iondist_peakfuncs(peakfunc=gauss2d,xg=xg,yg=yg)[0]#could be used later for initializing peak shapes

            #inititalizing the peak shapes, has to be done only once for analyzing the whole measurement period DOY150-220 1996
            P=zeros((len(steps),len(self.Ions),len(xg)))
            for k,step in enumerate(steps):
                for i,ion in enumerate(self.Ions):
                    P[k][i]=ion.create_peakfunc(step=step,xg=xg,yg=yg,normalize_integral=True,cutoff=True,coverage_rel=peak_coverage_rel)			
            
            t1=clock()
            
            #inititalizing weights, has to be done only once for analyzing the whole measurement period DOY150-220 1996
            if W_start==None:
                renormalize_intensities=True#work around, generalize later
            else:	
                renormalize_intensities=False		
            W=self.create_relionprobs(steps=steps,xg=xg,yg=yg,Plot=False,ionlist_plot=[],plot_maxrelcont=False,CBlog=False,ion_peakbounds="coverage",coverage_rel=peak_coverage_rel,countlevel_rel=0.01,normalize_integrals=renormalize_intensities)
            #return W
            
            
            t2=clock()

            #start loop over all steps
            Counts=zeros((len(steps),N_iter,len(self.Ions)))
            for k,step in enumerate(steps):
                
                Counts_step=zeros((N_iter,len(hdata),len(self.Ions)))
                
                #get step-specific normalized peak shapes
                P_step_unnorm=array([P[k]]*len(hdata))
                #return P_step_unnorm
                P_step_normfactor=array([sum(P_step_unnorm,axis=2).astype("float")]*len(xg)).transpose(1,2,0)
                P_step_normfactor[P_step_normfactor==0]=1.#avoids nans and should not introduce errors, since the normfactor can only be zero if all counts are zero.
                #return P_step_unnorm, P_step_normfactor
                P_step=P_step_unnorm/P_step_normfactor
                #return P_step
                
                #get start weigths (can be chosen step-specific, but should not matter) 
                W_step_j=array([W[k]]*len(hdata))	
                
                hdata_step=hdata#generalize later!	
                
                #return P_step, W_step_j, hdata 
                
                #start counts iteration	for all ions simultaneously 
                for j in range(N_iter):
                    
                    #calculate counts
                    #Counts_step_j=sum(W_step_j*hdata_step,axis=1)
                    C=product([W_step_j,hdata],axis=0)
                    C1=vstack(C).reshape(len(hdata),len(self.Ions),len(xg))
                    #return C1
                    Counts_step_j=sum(C1,axis=2)	
                    #return P_step,Counts_step_j
                    
                    #calculate weights
                    C=product([P_step,Counts_step_j],axis=0)
                    D=ravel(C)
                    E=vstack(D).reshape(len(D),len(xg))
                    F=E.reshape(len(hdata),len(self.Ions),len(xg))
                    P_rescaled=F
                    P_rescaled_normfactor=sum(P_rescaled,axis=1).astype("float")
                    P_rescaled_normfactor[P_rescaled_normfactor==0]=1.
                    W_step_j=P_rescaled/P_rescaled_normfactor[:,newaxis]
                    #return P_step,P_rescaled,P_rescaled_normfactor,W_step_j
                    
                    #old version
                    #P_rescaled=(P_step.T*Counts_step_j).T
                    #P_rescaled_normfactor=sum(P_rescaled,axis=0).astype("float")
                    #P_rescaled_normfactor[P_rescaled_normfactor==0]=1.#avoiding nans outside the generalized sigma environments
                    #W_step_j=P_rescaled/P_rescaled_normfactor
                    
                    Counts_step[j]=Counts_step_j
                    
                t3=clock()
                
                #print t1-t0
                #print t2-t1
                #print t3-t2

                Counts_step=Counts_step.transpose(1,0,2)
                
                
                return Counts_step



            
        def create_random_dist_iter(self,step,xg,yg,N_c,N_iter):
            """
            N_c=arraylike, Number of counts per ion
            """
            self.P_ions=zeros((len(self.Ions),len(xg)))				
            for i,ion in enumerate(self.Ions):
                mask_ion=ion.create_peakint_mask(step,xg,yg,coverage_rel=0.99,Plot=False,CBlog=False,vmin=1e-10,normalize_integral=True)
                #print mask_ion
                self.P_ions[i]=ion.create_peakfunc(step,xg,yg,True)*mask_ion
                self.P_ions[i]/=sum(self.P_ions[i])
            
            t0=clock()
            R_data=zeros((N_iter,len(xg)))
            j=0
            while j<N_iter:
            
                print(j)
                i=0
                while i<len(self.Ions):
                    r_data=random_choice(N_c[i],self.P_ions[i])
                    R_data[j]=R_data[j]+r_data
                    i+=1
                j+=1
                t1=clock()
            print("accumulated random sample creation time: %i seconds for %i distributions"%(t1-t0,N_iter)) 
            return R_data,N_c


        
        
            
            
        def test_ioncount_iteration(self,ionlist,true_counts,peakshapes,step,intensities=None,tofrange=[150,600],Erange=[1,100],tofbin=2,Ebin=2,Plot_Iondist=True,N_iter=100,N_test=1):	
            """
            TODO: Substitute random_counts (equal scalar quantity for all ions) by vector counts_true.
            """
            t0=clock()
            
            xg,yg=self.create_xygrid(Xrange=tofrange,Yrange=Erange,Xbin=tofbin,Ybin=Ebin)		
            self.add_ionlist(ionlist,peakshapes=peakshapes,intensities=intensities)
            self.create_relionprobs(steps=[step],xg=xg,yg=yg,Plot=Plot_Iondist,ionlist_plot=[],plot_maxrelcont=False,CBlog=False,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.01,normalize_integrals=True)
            
            results=zeros((N_test,len(ionlist)))
            j=0
            while j<N_test:
                res=self.iterate_ioncount_probs(step=step,xg=xg,yg=yg,hdata_timestamp=None,termcond=None,Plot=False,CBlog=False,N_iter=N_iter,random_data=True,random_counts=true_counts,Plot_ioncount_evolution=False,gen_data=True)[-1]
                #return results,res
                results[j]=res
                j+=1
            
            i=0
            while i<len(ionlist):
                results_ion=results.T[i]
                mean=average(results_ion)
                plt.figure()
                plt.hist(results_ion,arange(0,1.1*max(results_ion)),label="estimated mean counts=%.2f, true counts=%.2f"%(mean,true_counts[i]))
                plt.title("ion: %s, %i"%(self.Ion_names[i],step))
                plt.xlabel("estimeated counts")
                plt.ylabel("occurences")
                plt.legend()
                plt.show()
                i+=1
            
            t1=clock()	
            print("run time:", t1-t0)
            
                
        
        def iterate_ioncounts_test(self,ionlist,peakshapes,step,xg,yg,hdata_timestamp,intensities=None,normalize_integrals=False,termcond=0.01,Plot=True,CBlog=False,N_iter=10,random_data=False,randomcounts=1e5,Plot_ioncount_evolution=True,gen_data=True):	
            
            self.add_ionlist(ionlist,peakshapes=peakshapes,intensities=intensities)			 
            self.create_relionprobs(steps=[step],xg=xg,yg=yg,Plot=False,ionlist_plot=[],plot_maxrelcont=False,CBlog=False,ion_peakbounds="coverage",coverage_rel=0.99,countlevel_rel=0.01,normalize_integrals=normalize_integrals)
            res=self.iterate_ioncount_probs(step=step,xg=xg,yg=yg,hdata_timestamp=hdata_timestamp,termcond=termcond,Plot=Plot,CBlog=CBlog,N_iter=N_iter,random_data=random_data,randomcounts=randomcounts,Plot_ioncount_evolution=Plot_ioncount_evolution,gen_data=gen_data)
            return res