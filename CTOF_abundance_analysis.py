from numpy import *
import math
import scipy
import matplotlib.pyplot as plt
from CTOF_datafit import ctof_paramfit, eff_unc
import pickle

import pandas as pd

#import selfmade python modules
from pylib.dbData._dbData import dbData
from CTOF_cal import *
from peakshape_functions import *
from CTOF_ion import Ion, iondict, iondict_minimium
from CTOF_iondist import IonDist

#from libsoho.libctof import getionvel
from Libsoho._ctoflv1 import ctoflv1

#import time modules
from time import perf_counter as clock
import datetime as dt

#import peakshape functions
from peakshape_functions import *

def normalfunc(x, mu, sigma):
    return 1/(sigma*sqrt(2*pi))*exp(-(1/2)*((x-mu)/sigma)**2)
def polyfunc(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d
def ratiofunc(x, a, b):
    return a*x+b

ionlist_analyze=["C4+","C5+","C6+","N5+","N6+","N7+","O5+","O6+","O7+","O8+","Ne8+","Mg6+","Mg7+","Mg8+","Mg9+","Mg10+","Si7+","Si8+","Si9+","Si10+","Si11+","Si12+","S7+","S8+","S9+","Ca10+","Ca11+","Fe7+","Fe8+","Fe9+","Fe10+","Fe11+","Fe12+","Fe13+"]
#ionlist_analyze=["C4+","C5+","C6+","O5+","O6+","O7+","O8+","Si5+","Si6+","Si7+","Si8+","Si9+","Si10+","Si11+","Fe6+","Fe7+","Fe8+","Fe9+","Fe10+","Fe11+","Fe12+","Fe13+"]
class ctof_abundances(ctof_paramfit):
    """
    Docstring!
    """
    def analyze_slowwind_350400(self,ions_plot=ionlist_analyze,velmin_select=None,velmax_select=None): 
        s=open("fitres_slow350400","rb")
        cs=pickle.load(s)
        steps_350400=arange(30,97,1)

        vs_350400=self.analyze_veldist(ionlist=cs[-1],Chi=cs,modelnumber=0,steps=steps_350400,ions_plot=ionlist_analyze,cfracs=[0.61,0.32,0.14],velref=375.,runavg=5,MAX_velmin=250, MAX_velmax=500,stopstep=80, cmult=True,plot_evalrange=True, Xrange=[240,510],Yrange=None,Xrange_log=[110,750],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=18,labelsize=20,ticklabelsize=16,vproton=375,figtitle="",savefigure=False,figpath="",figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False,Nboot=1000,plot_steps=False, Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=30,save_meanvels=False,filepath="", filename="",scale_ions=False)
        self.apply_efficiency_correction(velmin_select,velmax_select)

    def analyze_fastwind_480570(self,ions_plot=ionlist_analyze,velmin_select=None,velmax_select=None): 
        f=open("fitres_fast480570","rb")
        cf=pickle.load(f)
        steps_480570=arange(15,75,1)

        vs_480570=self.analyze_veldist(ionlist=cf[-1],Chi=cf,modelnumber=0,steps=steps_480570,ions_plot=ionlist_analyze,cfracs=[0.61,0.32,0.14],velref=525.,runavg=5,MAX_velmin=400, MAX_velmax=650,stopstep=63, cmult=True,plot_evalrange=True, Xrange=[350,720],Yrange=None,Xrange_log=[210,1150], Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=-0.038,lgy=1.9,legsize=18,labelsize=20,ticklabelsize=16,vproton=525,figtitle="",savefigure=False,figpath="",figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False,Nboot=1000,plot_steps=False, Plot=True,figformat_autom=True, fitgauss=True,vth_fitguess=60,save_meanvels=False,filepath="", filename="",scale_ions=False)
        self.apply_efficiency_correction(velmin_select,velmax_select)

    
    
    def analyze_slowwind(self,ions_plot=ionlist_analyze,velmin_select=None,velmax_select=None): 
        s=open("fitdata_slowwind","rb")
        cs=pickle.load(s)
        Steps_slow=arange(35,95,1)

        vs=self.analyze_veldist(ionlist=cs[-1],Chi=cs,modelnumber=0,steps=Steps_slow, ions_plot=ionlist_analyze,cfracs=[0.61,0.32,0.14], velref=335.,runavg=3,MAX_velmin=200, MAX_velmax=470, stopstep=90, cmult=True,plot_evalrange=True, Xrange=[270,400],Yrange=None, Xrange_log=[240,560],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=0.1,lgy=1.9, legsize=18,labelsize=20,ticklabelsize=16,vproton=None,figtitle="",savefigure=False,figpath="",
        figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False, Nboot=1000, plot_steps=False,Plot=False,figformat_autom=True, fitgauss=True, vth_fitguess=15, save_meanvels=False, filepath="/Test/",filename="Test", scale_ions=False,save_totalcounts=False, counts_filename="Test_counts")
        self.apply_efficiency_correction(velmin_select,velmax_select)


    def analyze_fastwind(self,ions_plot=ionlist_analyze,velmin_select=None,velmax_select=None): 
        f=open("fitdata_fastwind","rb")
        cf=pickle.load(f)
        Steps_fast=arange(18,75,1)
        vf=self.analyze_veldist(ionlist=cf[-1],Chi=cf,modelnumber=0,steps=Steps_fast,ions_plot=ionlist_analyze,cfracs=[0.61,0.32,0.14],velref=505.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=68, cmult=True,plot_evalrange=True,Xrange=[350,720],Yrange=None,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=0.1,lgy=1.9,legsize=18,labelsize=20,ticklabelsize=16,vproton=None,figtitle="",savefigure=False,figpath="",figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False, Nboot=1000,plot_steps=False,Plot=False,figformat_autom=True, fitgauss=True, vth_fitguess=45, save_meanvels=False,filepath="/Test/",filename="Test",scale_ions=False,save_totalcounts=False, counts_filename="Test_counts")
        self.apply_efficiency_correction(velmin_select,velmax_select)

    def plot_relative_chargestate_abundances(self,ions_plot=["O6+","O7+","O8+"], yscale_log=True,labelsize=16,norm="max",figx=15, figy=7,save_figure=False,figtitle="test_chargestates"):
            Ioncounts_effcor=[]
            Ioncounts_effcor_unc=[]
            for i,ionname in enumerate(ions_plot):
                j=where(ionname==self.Ions_effcor)[0]
                ioncounts_effcor=self.Counts_effcor[j]
                ioncounts_effcor_unc=self.Counts_effcor_unc[j]
                Ioncounts_effcor.append(ioncounts_effcor)
                Ioncounts_effcor_unc.append(ioncounts_effcor_unc)
            Ioncounts_effcor=array(Ioncounts_effcor)
            Ioncounts_effcor_unc=array(Ioncounts_effcor_unc)
            if norm=="sum":
                Sumcounts=float(sum(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Sumcounts for i in arange(len(ions_plot))])
                abundances_unc = concatenate([Ioncounts_effcor_unc[i]/Sumcounts for i in arange(len(ions_plot))])
            elif norm=="max":
                Maxcounts=float(max(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Maxcounts for i in arange(len(ions_plot))])
                abundances_unc = concatenate([Ioncounts_effcor_unc[i]/Maxcounts for i in arange(len(ions_plot))])
            fig, ax = plt.subplots(figsize=(figx, figy))
            ax.bar(arange(len(ions_plot)), abundances ,edgecolor = 'black', label='SOHO')
            ax.errorbar(arange(len(ions_plot)), abundances, abundances_unc,linestyle="None",color="k")
            ax.set_xticks(arange(len(ions_plot)))
            labels=(elementn for elementn in ions_plot)
            ax.set_xticklabels(labels, fontsize=13, rotation=90)
            ax.legend(loc="upper left")
            ax.set_ylim(0,1.5)
            if yscale_log==True:
                ax.set_ylim(min(abundances)/2.,1.5)
                ax.set_yscale('log')
            ax.grid(which='both', axis='y')
            #ax.set_xlabel(r"$ \rm{ion \ species}$",fontsize=labelsize+5)
            ax.set_ylabel(r"$ \rm{relative \ abundance \ to \ O^{6+}}$", fontsize=labelsize+5)
            ax.set_title(r"$ \rm{rel.\ charge \ state \ abundances \ at \ v_p = (%i \pm 5) \ km/s}$"%(self.Proton_speed_eval),fontsize=23)
            fig.show()
            self.abundances=abundances
            self.abundances_unc=abundances_unc
            
            if save_figure==True:
                plt.savefig("%s"%figtitle)
                    
    def plot_chargestate_abundances_chianticomparison(self,elem='O',ions_plot=["O6+","O7+","O8+"], yscale_log=True,labelsize=16,norm="max",figx=15, figy=7,save_figure=False,figtitle="test_chargestates"):
            Ioncounts_effcor=[]
            Ioncounts_effcor_unc=[]
            for i,ionname in enumerate(ions_plot):
                j=where(ionname==self.Ions_effcor)[0]
                ioncounts_effcor=self.Counts_effcor[j]
                ioncounts_effcor_unc=self.Counts_effcor_unc[j]
                Ioncounts_effcor.append(ioncounts_effcor)
                Ioncounts_effcor_unc.append(ioncounts_effcor_unc)
            Ioncounts_effcor=array(Ioncounts_effcor)
            Ioncounts_effcor_unc=array(Ioncounts_effcor_unc)
            if norm=="sum":
                Sumcounts=float(sum(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Sumcounts for i in arange(len(ions_plot))])
                abundances_unc = concatenate([Ioncounts_effcor_unc[i]/Sumcounts for i in arange(len(ions_plot))])
            elif norm=="max":
                Maxcounts=float(max(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Maxcounts for i in arange(len(ions_plot))])
                abundances_unc = concatenate([Ioncounts_effcor_unc[i]/Maxcounts for i in arange(len(ions_plot))])
            else:
                abundances = concatenate([Ioncounts_effcor[i] for i in arange(len(ions_plot))])
                abundances_unc = concatenate([Ioncounts_effcor_unc[i] for i in arange(len(ions_plot))])
            ChiantiData = pd.read_csv("./Data/ChiantiResults/"+elem+"Long.csv")[ions_plot]
            width = 0.25
            t = 1.e+4*arange(150)+5.e+5
            chis=zeros_like(t)
            mus=zeros_like(t)
            mus_err=zeros_like(t)
            sigmas=zeros_like(t)
            ratios=zeros_like(t)         
            for temperature in range(len(t)):
                if elem == 'O':
                    ratios[temperature]=ChiantiData.iloc[temperature][0]/ChiantiData.iloc[temperature][1]
                else:   
                    popt, pcov = scipy.optimize.curve_fit(normalfunc, arange(len(ions_plot)), ChiantiData.iloc[temperature])
                    mus[temperature]=popt.tolist()[0]
                    sigmas[temperature]=popt.tolist()[1]
                    mus_err[temperature]=sqrt(pcov[0][0])
                    '''if temperature==70 and elem == 'Fe':
                        fig, ax = plt.subplots()
                        ax.scatter(arange(len(ions_plot)), ChiantiData.iloc[temperature])
                        ax.plot(arange(0,len(ions_plot)-1,0.1),normalfunc(arange(0,len(ions_plot)-1,0.1), *popt))
                        fig.show()'''
                chidummy = 0
                for chargestate in range(len(abundances)):
                    chidummy += (((abundances/sum(abundances))[chargestate]-(ChiantiData.iloc[temperature]/sum(ChiantiData.iloc[temperature]))[chargestate])/(abundances_unc/sum(abundances))[chargestate])**2
                chis[temperature] = chidummy
            if elem == 'O':
                closest_temp=where(ratios==min(ratios, key=lambda x:abs(x-abundances[0]/abundances[1])))[0][0]
                temp_error=t[closest_temp]/3*(abundances_unc[0]/abundances[0])#closest_temp*(d/a+e/b+f/c)#closest_temp*(((abundances_unc[0]/abundances[0]+abundances_unc[1]/abundances[1])*abundances[0]/abundances[1] + o) / (abundances[0]/abundances[1]-m) + n/l)
            else:
                if elem == 'C':
                    templimit = 80
                else:
                    templimit = 110
                popt, pcov = scipy.optimize.curve_fit(ratiofunc,t[45:templimit],mus[45:templimit],sigma=mus_err[45:templimit])
                fig, ax = plt.subplots()
                ax.errorbar(t[45:templimit],mus[45:templimit],yerr=mus_err[45:templimit])
                ax.plot(t[45:templimit],ratiofunc(t[45:templimit], *popt))
                fig.show()
                a,b=popt.tolist()[0],popt.tolist()[1]
                def normalfunc_t(x,t,s):
                    return 1/((s)*sqrt(2*pi))*exp(-(1/2)*((x-(a*t+b))/(s))**2)
                popt,pcov=scipy.optimize.curve_fit(normalfunc_t,arange(len(ions_plot)),abundances/sum(abundances),sigma=abundances_unc/sum(abundances),absolute_sigma=True)
                closest_temp=where(t==min(t, key=lambda x:abs(x-popt.tolist()[0])))[0][0]
                print('temp='+str(popt.tolist()[0])+'  Closest in list= '+str(closest_temp))
                temp_error=sqrt(pcov[0][0])
            fig, ax = plt.subplots(figsize=(figx, figy))
            ax.bar(arange(len(ions_plot)), abundances/sum(abundances), width=width,edgecolor = 'black', label='SOHO')
            #ax.bar(arange(len(ions_plot))+width, ChiantiData.iloc[minpos]/sum(ChiantiData.iloc[minpos]), width=width,edgecolor = 'black', label='Chianti')
            ax.bar(arange(len(ions_plot))+width, ChiantiData.iloc[closest_temp]/sum(ChiantiData.iloc[closest_temp]), width=width,edgecolor = 'black', label='Chianti')
            ax.errorbar(arange(len(ions_plot)), abundances/sum(abundances), abundances_unc/sum(abundances),linestyle="None",color="k")
            ax.set_xticks(arange(len(ions_plot)))
            labels=(elementn for elementn in ions_plot)
            ax.set_xticklabels(labels, fontsize=20)
            ax.legend(loc="upper left",prop={'size': 24})
            #ax.set_ylim(0,0.5)
            if yscale_log==True:
                ax.set_ylim(min(abundances)/2.,1.5)
                ax.set_yscale('log')
            ax.grid(which='both', axis='y')
            ax.set_xlabel(r"$ \rm{ion \ species}$",fontsize=labelsize+5)
            ax.set_ylabel(r"$ \rm{rel.abundance \ at \ v_p = (%i \pm 5) \ km/s}$"%(self.Proton_speed_eval), fontsize=labelsize+5)
            ax.set_title('rel. '+elem+' charge state abundances at {:.2f}'.format(t[closest_temp]/1e6)+r'$\pm$'+str(math.ceil(3*temp_error/1e4)/1e2)+r' MK ($\chi^2_r$ = {:.2f})'.format(chis[closest_temp]/len(ions_plot)), fontsize=26)
            if elem == 'O': 
                ax.set_title('rel. '+elem+' charge state abundances at {:.2f}'.format(t[closest_temp]/1e6)+r'$\pm$'+str(math.ceil(3*temp_error/1e4)/1e2)+' MK', fontsize=26)
                ax.legend(loc="upper left",prop={'size': 24})
            if elem == 'Si':
                ax.set_title('rel. '+elem+' charge state abundances at 1.25'+r'$\pm$'+str(math.ceil(3*temp_error/1e4)/1e2)+' MK ($\chi^2_r$ = {:.2f})'.format(chis[closest_temp]/len(ions_plot)), fontsize=26)
            fig.show()
            self.abundances=abundances
            self.abundances_unc=abundances_unc
            if save_figure==True:
                plt.savefig("%s"%figtitle)


    def plot_chianticomparison_of_ratios(self,elem='O',ions_plot=["O6+","O7+","O8+"], yscale_log=True,labelsize=16,norm="max",figx=15, figy=7,save_figure=False,figtitle="test_chargestates"):
            Ioncounts_effcor=[]
            for i,ionname in enumerate(ions_plot):
                j=where(ionname==self.Ions_effcor)[0]
                ioncounts_effcor=self.Counts_effcor[j]
                Ioncounts_effcor.append(ioncounts_effcor)
            Ioncounts_effcor=array(Ioncounts_effcor)
            if norm=="sum":
                Sumcounts=float(sum(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Sumcounts for i in arange(len(ions_plot))])
            elif norm=="max":
                Maxcounts=float(max(Ioncounts_effcor))
                abundances = concatenate([Ioncounts_effcor[i]/Maxcounts for i in arange(len(ions_plot))])
            abundances=abundances[::-1]
            ChiantiData = pd.read_csv("./Data/ChiantiResults/"+elem+"Long.csv")[ions_plot]
            ions_plot.reverse()
            width = 0.25
            t = 1.e+4*arange(150)+5.e+5
            TempResults,abundanceRatio = zeros(len(ions_plot)-1),zeros(len(ions_plot)-1)
            for i in range(len(ions_plot)-1):
                abundanceRatio[i] = abundances[i]/abundances[i+1]
                chis=zeros_like(t)
                for temperature in range(len(t)):
                    chis[temperature]=abs(abundanceRatio[i]-(ChiantiData.iloc[temperature][(-1-i)]/ChiantiData.iloc[temperature][-2-i]))
                minpos = where(chis==min(chis))[0][0]
                TempResults[i] = t[minpos]
            fig, ax = plt.subplots(figsize=(figx, figy))
            ax.bar(arange(len(ions_plot)-1), TempResults, width=width,edgecolor = 'black', label='SOHO')
            ax.set_xticks(arange(len(ions_plot)-1))
            labels=(ions_plot[i]+'/'+ions_plot[i+1] for i in range(len(ions_plot)-1))
            ax.set_xticklabels(labels)
            ax.legend(loc="upper left")
            ax.set_ylim(t[0],t[-1])
            if yscale_log==True:
                ax.set_ylim(t[0],t[-1])
                ax.set_yscale('log')
            ax.grid(which='both', axis='y')
            rects = ax.patches
            for rect, label in zip(rects, [str(temp/1e6)+' MK' for temp in TempResults]):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom")
            ax.set_xlabel(r"$ \rm{ion \ ratios}$",fontsize=labelsize)
            ax.set_ylabel(r"$ \rm{best \ fitted \ temperature \ at \ v_p = (%i \pm 5) \ km/s}$"%(self.Proton_speed_eval), fontsize=labelsize)
            ax.set_title('Freeze-in temperatures for different '+elem+' ratios')
            fig.show()

            if save_figure==True:
                plt.savefig("%s"%figtitle)

    def plot_relative_element_abundances(self,literature_abundances="fast", labelsize=16,figx=7, figy=7,save_figure=False,figtitle="test_elements"):          
            """
            Note: Works only if the charge state abundances have been evaluated for all relevant SW ions directly before!
            """
            elems=[]
            elemabundances=[]
            elemabundances_unc_sq=[]
            ssabundances=[]
            FIPS = pd.read_csv("./Data/FIP.csv")
            plotfips=[]
            I=IonDist()
            I.add_ionlist(names=list(self.Ions_effcor),peakshapes="kappa_moyalpar_Easym",intensities=[1.]*len(list(self.Ions_effcor)),print_ions=True)

            Z=[]
            for i,ion in enumerate(I.Ions):  
                name = ion.name
                
                z=ion.atomnumber
                if z not in Z:
                    Z.append(z)
                elem=''
                for letter in name:
                    if not letter.isdigit() and letter!='+':
                        elem+=letter
                if not elem in elems and i!=0:
                    elems.append(elem)
                    elemabundances.append(elemabundance)
                    elemabundances_unc_sq.append(elemabundance_unc_sq)
                    plotfips.append(FIPS.iloc[ion.atomnumber-1,1])
                    elemabundance=self.abundances[i]
                    elemabundance_unc_sq=self.abundances_unc[i]**2
                    ssabundances.append(10**(FIPS.iloc[ion.atomnumber-1,2]-12))
                elif not elem in elems and i==0:
                    elems.append(elem)
                    plotfips.append(FIPS.iloc[ion.atomnumber-1,1])
                    elemabundance=self.abundances[i]
                    elemabundance_unc_sq=self.abundances_unc[i]**2
                    ssabundances.append(10**(FIPS.iloc[ion.atomnumber-1,2]-12))
                else:
                    elemabundance+=self.abundances[i]
                    elemabundance_unc_sq+=self.abundances_unc[i]**2
                #print elemabundance elemabundance_unc_sq
                
            elemabundances.append(elemabundance)
            elemabundances_unc_sq.append(elemabundance_unc_sq)
            elemabundances_unc=sqrt(elemabundances_unc_sq)
            print("elemabundances,elemabundances_unc:",elemabundances,elemabundances_unc) 
            abundance_O=elemabundances[2]
            abundance_O_unc=elemabundances_unc[2]
            elemabundances_ratio=elemabundances/abundance_O
            #elemabundances_ratio_unc=elemabundances_unc/abundance_O#correct this!
            elemabundances_ratio_unc=sqrt((elemabundances_unc/abundance_O)**2+((elemabundances/abundance_O**2)*abundance_O_unc)**2)
            ssabundance_O=ssabundances[2]
            ssabundances_ratio=ssabundances/ssabundance_O
            
            Z=array(Z)
            elinewidth=2
            capsize=5 
            capthick=2
            fig, ax = plt.subplots(figsize=(figx, figy))
            #ax.scatter(Z,elemabundances_ratio)
            ax.set_ylim(1e-3,2.0)
            ax.set_xlim(4,28)
            ax.set_yscale('log')
            ax.grid(which='both', axis='y')
            for i, txt in enumerate(elems):
                if i==0:
                    ax.errorbar(Z[i],elemabundances_ratio[i],yerr=elemabundances_ratio_unc[i],linestyle="None",marker="o",markersize=3,color="b",elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="this study")
                else:
                    ax.errorbar(Z[i],elemabundances_ratio[i],yerr=elemabundances_ratio_unc[i],linestyle="None",marker="o",markersize=3,color="b",elinewidth=elinewidth, capsize=capsize, capthick=capthick)
                ax.annotate(txt, (Z[i]+.5, elemabundances[i]), fontsize=20)
            ax.set_title(r'$\rm{elemental \ abundances \ at \ v_p = (%i \pm 5) \ km/s \ (relative \ to \ oxygen)}$'%
            (self.Proton_speed_eval),fontsize=labelsize+5)
            ax.set_xlabel(r"$ \rm{atomic \ number}$",fontsize=labelsize+5)
            ax.set_ylabel(r"$ \rm{rel.abundance}$",fontsize=labelsize+5)
            xticks=Z
            yticks=array([1e-3,1e-2,1e-1,1])
            ticklabelsize=labelsize-2
            ax.tick_params(axis="x", labelsize=ticklabelsize+3)
            ax.tick_params(axis="y", labelsize=ticklabelsize+3)
            ax.set_xticks(xticks, minor=False)
            ax.set_yticks(yticks)
            if literature_abundances=="slow":
                Z,ab_lit,aberr_lit=loadtxt("SW_Elemental_Abundances_BochslerReview2006",skiprows=1, usecols=(0,1,2),unpack=True)
                plt.errorbar(Z,ab_lit,yerr=aberr_lit,linestyle="None",marker="o",markersize=3,color="r",alpha=0.5,elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="other in-situ studies (with SOHO, ACE, Ulysses)\nafter Bochsler et al. - (Review 2006)")
            elif literature_abundances=="fast":
                Z,ab_lit,aberr_lit=loadtxt("SW_Elemental_Abundances_BochslerReview2006",skiprows=1,usecols=(0,3,4),unpack=True)
                plt.errorbar(Z,ab_lit,yerr=aberr_lit,linestyle="None",marker="o",markersize=3,color="r",alpha=0.5,elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="other in-situ studies (with SOHO, ACE, Ulysses)\nafter Bochsler et al. - (Review 2006)")
            else:
                pass
            plt.legend(loc="upper right",prop={'size': labelsize-3})
            if save_figure==True:
                plt.savefig("%s"%figtitle)
            self.elems=elems
            self.Z=Z
            self.fips=plotfips
            self.elemabundances_ratio=elemabundances_ratio
            self.abundance_ratio_ss=ssabundances_ratio
            self.elemabundances_ratio_unc=elemabundances_ratio_unc
            

    def get_FIP_enhancement(self, SW_type="fast",labelsize=16):       
            self.SW_elem_enhancements=self.elemabundances_ratio / self.abundance_ratio_ss 
            self.SW_elem_enhancements_unc=self.elemabundances_ratio_unc/self.abundance_ratio_ss#no uncertainties in solar system abundances in Bochsler Review -> check whether really irrelevant 
            fig, ax = plt.subplots()
            ax.scatter(self.fips, self.SW_elem_enhancements)
            for i, txt in enumerate(self.elems):
                ax.annotate(txt, (self.fips[i]+.25, self.SW_elem_enhancements[i]), fontsize=12)
            ax.set_yscale('log')
            ax.set_title('elem. abundance enhancement (%s solar wind vs photosphere)'%(SW_type))
            ax.set_xlabel(r"$ \rm{First \ ionization \ potential \ (FIP) \ [V]}$",fontsize=labelsize)
            ax.set_ylabel(r"$ \rm{[N_X/N_O]_{SW} / [N_X/N_O]_{PP}}$",fontsize=labelsize)
            plt.show()
            enh=self.SW_elem_enhancements
            enh_unc=self.SW_elem_enhancements_unc
            return enh,enh_unc 
    
        
    def plot_FIP_enhancement(self, enh_slow, enh_unc_slow, enh_fast, enh_unc_fast, labelsize=20,figx=9, figy=13,save_figure=False,figtitle="test_enhancements"):
            elinewidth=2
            capsize=0 
            capthick=0
            #enh_fast, enh_unc_fast
            fig, ax = plt.subplots(figsize=(figx,figy))
                
            for i, txt in enumerate(self.elems):
                if i==0:
                    ax.errorbar(self.fips[i],enh_fast[i],yerr=enh_unc_fast[i],linestyle="None",marker="s",markersize=8,color="blue",elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="fast wind",alpha=1.0)
                else:
                    ax.errorbar(self.fips[i],enh_fast[i],yerr=enh_unc_fast[i],linestyle="None",marker="s",markersize=8,color="blue",elinewidth=elinewidth, capsize=capsize, capthick=capthick, alpha=1.0)
            for i, txt in enumerate(self.elems):
                if i==0:
                    ax.errorbar(self.fips[i],enh_slow[i],yerr=enh_unc_slow[i],linestyle="None",marker="d",markersize=10,color="orange",elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="slow wind")
                else:
                    ax.errorbar(self.fips[i],enh_slow[i],yerr=enh_unc_slow[i],linestyle="None",marker="d",markersize=10,color="orange",elinewidth=elinewidth, capsize=capsize, capthick=capthick)
            
                if self.elems[i]=="Si":
                    ax.annotate(txt, (self.fips[i], 3.5), fontsize=15)
                elif self.elems[i]=="Mg":
                    ax.annotate(txt, (self.fips[i]-0.4, 4.5), fontsize=15)
                elif self.elems[i]=="Fe":
                    ax.annotate(txt, (self.fips[i]-0.2, 0.7), fontsize=15)
                else:
                    ax.annotate(txt, (self.fips[i]-0.2, 4.5), fontsize=15)
            
            ax.set_yscale('log')
            ax.set_xlim(3,25)
            ax.set_ylim(0.2,10**0.7)
            ax.plot([3,25],[1,1],linestyle="--",linewidth=1.5,color="k")
            
            #ax.set_xticks(xticks, minor=False)
            yticks=array([0.2,0.4,0.6,0.8,1.0,1.5,2.0,3.0,4.0])
            ax.set_yticks(yticks)
            #yticklabels=[0.6,0.7,0.8,0.9,1.0,2.0,3.0]
            #ax.set_yticklabels(yticklabels)
            labels = [item.get_text() for item in ax.get_yticklabels()]
            labels = ["0.2","0.4","0.6","0.8","1.0","1.5","2.0","3.0","4.0"]
            ax.set_yticklabels(labels)

            ticklabelsize=labelsize-2
            ax.tick_params(axis="x", labelsize=ticklabelsize)
            ax.tick_params(axis="y", labelsize=ticklabelsize)
            
            plt.legend(loc="lower right",prop={'size': labelsize-2})
            
            ax.set_title(r"$ \rm{elemental \ abundance \ enhancement \ (solar \ wind \ vs \ photosphere)}$",fontsize=labelsize)
            ax.set_xlabel(r"$ \rm{first \ ionization \ potential \ (FIP) \ [V]}$",fontsize=labelsize)
            ax.set_ylabel(r"$ \rm{[N_X/N_O]_{SW} \ / \ [N_X/N_O]_{PP}}$",fontsize=labelsize)
            plt.show()

            if save_figure==True:
                plt.savefig("%s"%figtitle,bbox_inches='tight')

        
#main()        
d=ctof_abundances(timeframe=[[174,177]],minute_frame=[0,1440],load_processed_PHAdata=True)

IMAGES_SAVING=True
PlotInLog=False
d.analyze_slowwind(velmin_select=310,velmax_select=360)
print('SLOW WIND')

#d.plot_chargestate_abundances_chianticomparison(elem='Fe',ions_plot=['Fe7+','Fe8+','Fe9+','Fe10+','Fe11+','Fe12+','Fe13+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="FeChiantiComparisonSLOW")
#d.plot_chargestate_abundances_chianticomparison(elem='Si',ions_plot=['Si7+','Si8+','Si9+','Si10+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="SiChiantiComparisonSLOW")
#d.plot_chargestate_abundances_chianticomparison(elem='O',ions_plot=["O6+",'O7+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="OChiantiComparisonSLOW")
#d.plot_chargestate_abundances_chianticomparison(elem='C',ions_plot=['C4+','C5+','C6+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="CChiantiComparisonSLOW")

#d.plot_chianticomparison_of_ratios(elem='Fe',ions_plot=['Fe8+','Fe9+','Fe10+','Fe11+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="FeRatiosTemperaturesSLOW")
#d.plot_chianticomparison_of_ratios(elem='Si',ions_plot=['Si7+','Si8+','Si9+','Si10+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="SiRatiosTemperaturesSLOW")
#d.plot_chianticomparison_of_ratios(elem='O',ions_plot=['O6+','O7+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="ORatiosTemperaturesSLOW")
#d.plot_chianticomparison_of_ratios(elem='C',ions_plot=['C4+','C5+','C6+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="CRatiosTemperaturesSLOW")

d.plot_relative_chargestate_abundances(ions_plot=d.Ions_effcor,norm='max',figx=21,save_figure=True,figtitle="all_chargestates_slowwind")
d.plot_relative_element_abundances(literature_abundances="slow",figx=10.5, figy=8, save_figure=True, figtitle="elemental_abundances_slow")
enh_slow,enh_slow_unc=d.get_FIP_enhancement(SW_type="slow")

d.analyze_fastwind(velmin_select=470,velmax_select=600)
print('FAST WIND')
#d.plot_chargestate_abundances_chianticomparison(elem='Fe',ions_plot=['Fe7+','Fe8+','Fe9+','Fe10+','Fe11+','Fe12+','Fe13+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="FeChiantiComparisonFAST")
#d.plot_chargestate_abundances_chianticomparison(elem='Si',ions_plot=['Si7+','Si8+','Si9+','Si10+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="SiChiantiComparisonFAST")
#d.plot_chargestate_abundances_chianticomparison(elem='O',ions_plot=["O6+",'O7+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="OChiantiComparisonFAST")
#d.plot_chargestate_abundances_chianticomparison(elem='C',ions_plot=['C4+','C5+','C6+'],norm='',figx=21,save_figure=IMAGES_SAVING,yscale_log=PlotInLog,figtitle="CChiantiComparisonFAST")

#d.plot_chianticomparison_of_ratios(elem='Fe',ions_plot=['Fe8+','Fe9+','Fe10+','Fe11+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="FeRatiosTemperaturesFAST")
#d.plot_chianticomparison_of_ratios(elem='Si',ions_plot=['Si7+','Si8+','Si9+','Si10+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="SiRatiosTemperaturesFAST")
#d.plot_chianticomparison_of_ratios(elem='O',ions_plot=['O6+','O7+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="ORatiosTemperaturesFAST")
#d.plot_chianticomparison_of_ratios(elem='C',ions_plot=['C4+','C5+','C6+'], yscale_log=True,labelsize=16,norm="sum",save_figure=IMAGES_SAVING,figtitle="CRatiosTemperaturesFAST")


d.plot_relative_chargestate_abundances(ions_plot=d.Ions_effcor,figx=21,norm='max',save_figure=True,figtitle="all_chargestates_fastwind")
d.plot_relative_element_abundances(literature_abundances="fast",figx=10.5, figy=8,save_figure=True, figtitle="elemental_abundances_fast")
enh_fast,enh_fast_unc=d.get_FIP_enhancement(SW_type="fast")

d.plot_FIP_enhancement(enh_fast,enh_fast_unc,enh_fast,enh_fast_unc,save_figure=True,figtitle="FIP_elemental_enhancement")











