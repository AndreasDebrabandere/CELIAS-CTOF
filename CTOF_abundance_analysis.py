from numpy import *
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


ionlist_analyze=["C4+","C5+","C6+","N5+","N6+","N7+","O6+","O7+","O8+","Ne8+","Mg6+","Mg7+","Mg8+","Mg9+","Mg10+","Si7+","Si8+","Si9+","Si10+","S7+","S8+","S9+","Ca10+","Ca11+","Fe7+","Fe8+","Fe9+","Fe10+","Fe11+","Fe12+","Fe13+"]

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
        figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False, Nboot=1000, plot_steps=False,Plot=True,figformat_autom=True, fitgauss=True, vth_fitguess=15, save_meanvels=False, filepath="/Test/",filename="Test", scale_ions=False,save_totalcounts=False, counts_filename="Test_counts")
        self.apply_efficiency_correction(velmin_select,velmax_select)


    def analyze_fastwind(self,ions_plot=ionlist_analyze,velmin_select=None,velmax_select=None): 
        f=open("fitdata_fastwind","rb")
        cf=pickle.load(f)
        Steps_fast=arange(18,75,1)
        vf=self.analyze_veldist(ionlist=cf[-1],Chi=cf,modelnumber=0,steps=Steps_fast,ions_plot=ionlist_analyze,cfracs=[0.61,0.32,0.14],velref=505.,runavg=5,MAX_velmin=400, MAX_velmax=2000,stopstep=68, cmult=True,plot_evalrange=True,Xrange=[350,720],Yrange=None,Xrange_log=[210,1150],Yrange_log=None,figx=13.9,figy=9,adjust_top=0.57,lgx=0.1,lgy=1.9,legsize=18,labelsize=20,ticklabelsize=16,vproton=None,figtitle="",savefigure=False,figpath="",figname="test",peakshape="kappa_moyalpar_Easym",plot_errorbars=False, Nboot=1000,plot_steps=False,Plot=True,figformat_autom=True, fitgauss=True, vth_fitguess=45, save_meanvels=False,filepath="/Test/",filename="Test",scale_ions=False,save_totalcounts=False, counts_filename="Test_counts")
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
            ax.bar(arange(len(ions_plot)), abundances)
            ax.errorbar(arange(len(ions_plot)), abundances, abundances_unc,linestyle="None",color="k")
            #for i, label in enumerate(self.element_names):
            #    plt.annotate(label, (arange(1,len(self.element_names)+1,1)[i], abundances[i]))
            #ax.set_xticks(np.add(element_charges,(0.8/2))) # set the position of the x ticks
            ax.set_xticks(arange(len(ions_plot)))
            #labels=(elementn[:-1] for elementn in ions_plot)
            labels=(elementn for elementn in ions_plot)
            ax.set_xticklabels(labels)
            #ax.tick_params(axis='y', which='minor')
            #ax[1].legend()
            #if lgx!=None and lgy!=None:
            ax.legend(loc="upper left")
            ax.set_ylim(0,1.5)
            if yscale_log==True:
                ax.set_ylim(min(abundances)/2.,1.5)
                ax.set_yscale('log')
            ax.grid(which='both', axis='y')
            ax.set_title('rel. charge state abundances')
            ax.set_xlabel(r"$ \rm{ion \ species}$",fontsize=labelsize)
            ax.set_ylabel(r"$ \rm{rel.abundance \ at \ v_p = (%i \pm 5) \ km/s}$"%(self.Proton_speed_eval), fontsize=labelsize)
            plt.show()
            self.abundances=abundances
            self.abundances_unc=abundances_unc
            
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
                print(name)
                
                z=ion.atomnumber
                if z not in Z:
                    Z.append(z)
                elem=''
                for letter in name:
                    print(letter)
                    if not letter.isdigit() and letter!='+':
                        elem+=letter
                if not elem in elems and i!=0:
                    print(elem)
                    elems.append(elem)
                    elemabundances.append(elemabundance)
                    elemabundances_unc_sq.append(elemabundance_unc_sq)
                    plotfips.append(FIPS.iloc[ion.atomnumber-1,1])
                    elemabundance=self.abundances[i]
                    elemabundance_unc_sq=self.abundances_unc[i]**2
                    ssabundances.append(10**(FIPS.iloc[ion.atomnumber-1,2]-12))
                elif not elem in elems and i==0:
                    print(elem)
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
                ax.annotate(txt, (Z[i]+.5, elemabundances[i]), fontsize=15)
            ax.set_title(r'$\rm{elemental \ abundances \ (relative \ to \ oxygen)}$',fontsize=labelsize)
            ax.set_xlabel(r"$ \rm{atomic \ number}$",fontsize=labelsize)
            ax.set_ylabel(r"$ \rm{rel.abundance \ at \ v_p = (%i \pm 5) \ km/s}$"%
            (self.Proton_speed_eval),fontsize=labelsize)
            xticks=Z
            yticks=array([1e-3,1e-2,1e-1,1])
            
            ticklabelsize=labelsize-2
            ax.tick_params(axis="x", labelsize=ticklabelsize)
            ax.tick_params(axis="y", labelsize=ticklabelsize)
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
                    ax.errorbar(self.fips[i],enh_fast[i],yerr=enh_unc_fast[i],linestyle="None",marker="s",markersize=8,color="dimgray",elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="fast wind",alpha=1.0)
                else:
                    ax.errorbar(self.fips[i],enh_fast[i],yerr=enh_unc_fast[i],linestyle="None",marker="s",markersize=8,color="dimgray",elinewidth=elinewidth, capsize=capsize, capthick=capthick, alpha=1.0)
            for i, txt in enumerate(self.elems):
                if i==0:
                    ax.errorbar(self.fips[i],enh_slow[i],yerr=enh_unc_slow[i],linestyle="None",marker="d",markersize=10,color="brown",elinewidth=elinewidth, capsize=capsize, capthick=capthick,label="slow wind")
                else:
                    ax.errorbar(self.fips[i],enh_slow[i],yerr=enh_unc_slow[i],linestyle="None",marker="d",markersize=10,color="brown",elinewidth=elinewidth, capsize=capsize, capthick=capthick)
            
                
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
d=ctof_abundances(timeframe=[[174,176]],minute_frame=[0,1440],load_processed_PHAdata=True)

d.analyze_slowwind(velmin_select=310,velmax_select=360)
d.plot_relative_chargestate_abundances(ions_plot=d.Ions_effcor,figx=21,save_figure=True,figtitle="all_chargestates_slowwind")
d.plot_relative_element_abundances(literature_abundances="slow",figx=10.5, figy=8, save_figure=True, figtitle="elemental_abundaes_slow")
enh_slow,enh_slow_unc=d.get_FIP_enhancement(SW_type="slow")

d.analyze_fastwind(velmin_select=470,velmax_select=600)
d.plot_relative_chargestate_abundances(ions_plot=d.Ions_effcor,figx=21,save_figure=True,figtitle="all_chargestates_fastwind")
d.plot_relative_element_abundances(literature_abundances="fast",figx=10.5, figy=8,save_figure=True, figtitle="elemental_abundaes_fast")
enh_fast,enh_fast_unc=d.get_FIP_enhancement(SW_type="fast")

d.plot_FIP_enhancement(enh_slow,enh_slow_unc,enh_fast,enh_fast_unc,save_figure=True,figtitle="FIP_elemental_enhancement")













