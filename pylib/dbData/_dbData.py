#{{{ Documentation and module imports
"""
Date: 15.08.2012
Author: Christian Drews <drews@physik.uni-kiel.de>, Lars Berger <berger@physik.uni-kiel.de>

KNOWN BUGS: #1 Memory leak if you call plot routine too often. Memory is not released. Matplotlib sucks
"""

import sys, os
from copy import deepcopy, copy
from numpy import array,histogram, sum,zeros,argmin,linspace,transpose,where,sqrt,exp, arange, max, min, nonzero, cos, sin, mean,std, isnan,isinf,inf, ones,NAN, histogram2d, ndarray,average,alltrue,greater_equal,equal,greater,less_equal,less,not_equal,logical_or,iterable,unique,meshgrid,ravel,swapaxes,reshape,asarray, histogramdd, hstack,vstack, searchsorted,seterr,log10
from matplotlib import animation
import matplotlib
from matplotlib.path import Path
from matplotlib.widgets import Cursor
import pylab
#from pylib.etMisc import loading_bar
import pickle as cPickle
import pdb
import functools
from scipy.optimize import leastsq
from scipy import ndimage
import numpy
from datetime import datetime

#}}}
def ArgParser(func):
	@functools.wraps(func)
	def funcDEC(self,*args,**kwargs):
		tmp = list(args)
		if isinstance(args[0],str) and args[0].lower()=="all":
			tmp[0]=self.mask.keys()
			#tmp[0].remove("Master")
			func(self,*tuple(tmp),**kwargs)
		elif isinstance(args[0],str):
			tmp[0]=[args[0]]
			func(self,*tuple(tmp),**kwargs)
		elif iterable(args[0]):
			func(self,*args,**kwargs)
		else:
			print('Invalid Arguments - Function call canceled!')
	return funcDEC
class show_mask(object):
	def __init__(self,dbData,smask,cmask):
		"""
		Class for vizualizing the current masks of dbData. Can be used to alter some 
		options and values set on any given mask
		INPUT: dbDATA; instance of dbData		       smask; list of masks masks that are drawn
		"""
		seterr(all="ignore")
		self.db = dbData
		self.smask = smask
		self.cmask = cmask
		self.ph = 0.8 #Panel Height
		self.axes = {}
	def _prepare_ax(self,kindex,key):
		"""
		Routine that prepares the figure
		"""
		#Nmask = sum(self.mask[key].get()*self.DM())
		ax = pylab.axes([0.1,0.1+kindex*self.ph,0.6,self.ph])
		ax2 = pylab.axes([0.7,0.1+kindex*self.ph,0.2,self.ph])
		ax.set_ylabel("%s"%self.db.mp[key].get("name"),\
		              backgroundcolor=self.db.mp[key].get("color"),color="white",\
			      weight="bold",fontsize=9)
		for i in ["bottom","top","left","right"]: 
			ax.spines[i].set_linewidth(3)
			ax2.spines[i].set_linewidth(3)
			ax.spines[i].set_zorder(50)
		ax.yaxis.set_ticks([])
		ax.yaxis.set_ticklabels([])
		ax.xaxis.set_ticklabels([])
		ax.set_xlim(-0.1,1.1)
		ax.set_ylim(-0.0,1.0)
		ax2.yaxis.set_ticks([])
		ax2.yaxis.set_ticklabels([])
		ax2.xaxis.set_ticklabels([])
		ax2.set_xlim(0,1.4)
		ax2.set_ylim(0,1)
		return ax,ax2
	def _get_cdata(self,prod,key):
		"""
		This routine returns the data product which is used for comparison in show_mask.
		If self.cmask == False this will simply return the current data product array
		INPUT: prod; data product as a string
		OUTPUT: Comparison Data array, Number of total events within that array
		"""
		if "MASK2D" in prod:
			cdata = zeros(10,dtype="S100")
			cdata[9]="User-Defined Submask: '%s'"%(prod)
			Nevents =10
		elif prod=="appliedmasks":
			cdata = self.db.mask[key].appliedmasks
			Nevents = sum(self.db.mask[self.cmask].get())
		elif prod=="directmasks":
			cdata = self.db.mask[key].directmasks.keys()
			Nevents = sum(self.db.mask[self.cmask].get())
		elif self.cmask==False:
			cdata = self.db.data[prod]
			Nevents = len(self.db.data[prod])
		else:
			if key!="Master":
				cdata = self.db.data[prod][self.db.mask[self.cmask].get()]
				Nevents = sum(self.db.mask[self.cmask].get())
			else:
				cdata = self.db.data[prod]
				Nevents = len(self.db.data[self.db.data.keys()[0]])
		return cdata, Nevents
	def _prepare_product_plot(self,pindex,prod,key):
		"""
		This routine plots the background bars for the show mask plot
		INPUT: TODO
		OUTPUT: TODO
		"""
		ax = self.axes[key][0]
		cdata = self.cdata
		if ("MASK2D" not in prod) and (prod!="appliedmasks") and (prod!="directmasks"):
			ax.text(0.01,self.ypos[pindex],self.db.dp[prod].get("name"),\
			horizontalalignment='left',verticalalignment="top",\
			fontsize=8,weight="bold",backgroundcolor="white",\
			rotation="horizontal",color=self.db.dp[prod].get("color"),zorder=12,clip_on=False)
		if not isinstance(cdata[0],str):
			minval=min(cdata[~isinf(cdata)*~isnan(cdata)])
			maxval=max(cdata[~isinf(cdata)*~isnan(cdata)])
			nanval=~alltrue(~isnan(cdata))
			infval=~alltrue(~isinf(cdata))
			rec2 = matplotlib.patches.Rectangle((0.,self.ypos[pindex]-self.dh),1.,self.dh,\
			                                     color=self.db.dp[prod].get("color"),\
							     alpha=0.3)
			ax.add_patch(rec2)
			ax.plot([0.,0.],[self.ypos[pindex]-self.dh,self.ypos[pindex]],\
			        "-",color=self.db.dp[prod].get("color"),lw=2)
			ax.plot([1.,1.],[self.ypos[pindex]-self.dh,self.ypos[pindex]],\
			        "-",color=self.db.dp[prod].get("color"),lw=2)
			ax.text(-0.007,self.ypos[pindex]-self.dh," %4.2f "%minval,\
			        horizontalalignment='right',verticalalignment="bottom",\
				fontsize=6,clip_on=True,rotation="horizontal",\
				zorder=11,color=self.db.dp[prod].get("color"),weight="bold")
			ax.text(1.005,self.ypos[pindex]," %4.2f "%maxval,\
			        horizontalalignment='left',verticalalignment="top",\
				fontsize=6,clip_on=True,rotation="horizontal",\
				zorder=11,color=self.db.dp[prod].get("color"),weight="bold")
			if nanval:
				ax.text(-0.007,self.ypos[pindex]," NAN ",\
				horizontalalignment='right',verticalalignment="top",\
				fontsize=6,clip_on=True,rotation="horizontal",weight="bold",\
				zorder=11,color="red")
			if infval:
				ax.text(1.005,self.ypos[pindex]-self.dh," INF ",\
				horizontalalignment='left',verticalalignment="bottom",\
				fontsize=6,clip_on=True,rotation="horizontal",weight="bold",\
				zorder=11,color="red")
		elif prod == "appliedmasks":
			rec2 = matplotlib.patches.Rectangle((0.,self.ypos[pindex]-self.dh),1.,self.dh,\
			                                     fill=False,ec="black",lw=2)
			ax.add_patch(rec2)
			vals=self.db.mask[key].appliedmasks
			ax.text(.5,self.ypos[pindex]-self.dh/3.,"Applied Masks : %i"%(len(vals)),\
				horizontalalignment='center',verticalalignment="center",\
				fontsize=6,clip_on=True,rotation="horizontal",\
				weight="bold",zorder=11)
			if len(vals)>9:
				for i,val in enumerate(vals[:8]):
					ax.text(0.1+i*0.1,self.ypos[pindex]-self.dh*2./3.,"%s"%val,\
						horizontalalignment='center',verticalalignment="center",\
						fontsize=6,clip_on=True,rotation="horizontal",\
						backgroundcolor=self.db.mp[val].get("color"),weight="bold",zorder=11,\
						color="white")
				ax.text(0.9,self.ypos[pindex]-self.dh*2./3.,"%i more"%(len(vals)-8),\
					horizontalalignment='center',verticalalignment="center",\
					fontsize=6,clip_on=True,rotation="horizontal",\
					weight="bold",zorder=11,\
					color="black")
			else:
				for i,val in enumerate(vals):
					ax.text(0.1+i*0.1,self.ypos[pindex]-self.dh*2./3.,"%s"%val,\
						horizontalalignment='center',verticalalignment="center",\
						fontsize=6,clip_on=True,rotation="horizontal",\
						backgroundcolor=self.db.mp[val].get("color"),weight="bold",zorder=11,\
						color="white")
		elif prod == "directmasks":
			rec2 = matplotlib.patches.Rectangle((0.,self.ypos[pindex]-self.dh),1.,self.dh,\
			                                     fill=False,ec="black",lw=2)
			ax.add_patch(rec2)
			vals=self.db.mask[key].directmasks.keys()
			ax.text(.5,self.ypos[pindex]-self.dh/3.,"Direct Masks : %i"%(len(vals)),\
				horizontalalignment='center',verticalalignment="center",\
				fontsize=6,clip_on=True,rotation="horizontal",\
				weight="bold",zorder=11)
			if len(vals)>9:
				for i,val in enumerate(vals[:8]):
					ax.text(0.1+i*0.1,self.ypos[pindex]-self.dh*2./3.,"%s"%val,\
						horizontalalignment='center',verticalalignment="center",\
						fontsize=6,clip_on=True,rotation="horizontal",\
						backgroundcolor="black",weight="bold",zorder=11,\
						color="white")
				ax.text(0.9,self.ypos[pindex]-self.dh*2./3.,"%i more"%(len(vals)-8),\
					horizontalalignment='center',verticalalignment="center",\
					fontsize=6,clip_on=True,rotation="horizontal",\
					weight="bold",zorder=11,\
					color="black")
			else:
				for i,val in enumerate(vals):
					ax.text(0.1+i*0.1,self.ypos[pindex]-self.dh*2./3.,"%s"%val,\
						horizontalalignment='center',verticalalignment="center",\
						fontsize=6,clip_on=True,rotation="horizontal",\
						backgroundcolor="black",weight="bold",zorder=11,\
						color="white")
		else:
			vals=unique(cdata)
			vals.sort()
			for i,val in enumerate(vals):
				ax.text(0.1+i*0.15,self.ypos[pindex]-self.dh/2.," %s "%val,\
				        horizontalalignment='left',verticalalignment="center",\
					fontsize=6,clip_on=True,rotation="horizontal",\
					weight="bold",zorder=11)
	def _plot_line(self,ax,x1,x2,y):
		ax.plot([x1,x2],[y]*2, linestyle="-",c="k",zorder=11)
	def _plot_marker(self,ax,marker,x,y,color="k"):
		"""
		Routine to plot markers in show mask routine
		"""
		if marker=="[":
			ax.plot([x,x],[y-self.dh/2.,y+self.dh/2.],lw=1.5,c=color,zorder=13)
			ax.plot([x,x-0.005],[y+self.dh/2.]*2,lw=1.5,c=color,zorder=13)
			ax.plot([x,x-0.005],[y-self.dh/2.]*2,lw=1.5,c=color,zorder=13)
		if marker=="]":
			ax.plot([x,x],[y-self.dh/2.,y+self.dh/2.],lw=1.5,c=color,zorder=13)
			ax.plot([x,x+0.005],[y+self.dh/2.]*2,lw=1.5,c=color,zorder=13)
			ax.plot([x,x+0.005],[y-self.dh/2.]*2,lw=1.5,c=color,zorder=13)
		if marker=="|":
			ax.plot([x,x],[y-self.dh/2.,y+self.dh/2.],lw=1.5,c=color,zorder=13)
		if marker==">":
			ax.plot([x],[y],linestyle="",c=color,marker=">",markersize=5,zorder=13)
		if marker=="<":
			ax.plot([x],[y],linestyle="",c=color,marker="<",markersize=5,zorder=13)
	def _plot_greater(self,ax,sm):
		"""
		Routine to plot "greater than / equal" operator in show mask
		"""
		minv,maxv = (sm.arg[0]-self.minval)/float(self.vr),1
		if minv<0: minv=0	
		self._plot_marker(ax,">",maxv,self.height)
		self._plot_line(ax,minv,maxv,self.height)
		if sm.operator in ["ge",">="]:
			self._plot_marker(ax,"]",minv,self.height)
		else:
			self._plot_marker(ax,"[",minv,self.height)

		ax.text(minv+0.005,self.height-self.dh/2.*0.9," %s "%sm.arg[0],\
		        horizontalalignment='left',verticalalignment="bottom",\
			fontsize=6,clip_on=True,rotation="horizontal",zorder=11,
			color="white",weight="bold")
		return minv,maxv
	def _plot_equal(self,ax,sm,prod):
		"""
		Routine to plot "greater than / equal" operator in show mask
		"""
		minv,maxv = (sm.arg[0]-self.minval)/float(self.vr),1
		if minv<0: minv=0	
		self._plot_marker(ax,"|",minv,self.height,color=self.db.dp[prod].get("color"))
		ax.text(minv+0.005,self.height," %s "%sm.arg[0],\
		        horizontalalignment='left',verticalalignment="center",\
			fontsize=6,clip_on=True,rotation="horizontal",zorder=11,
			color=self.db.dp[prod].get("color"),weight="bold")
		return 0,0
	def _plot_lesser(self,ax,sm):
		"""
		Routine to plot "lesser than / equal" operator in show mask
		"""
		minv,maxv = 0,(sm.arg[0]-self.minval)/float(self.vr)
		if maxv>1: maxv=1	
		self._plot_marker(ax,"<",minv,self.height)
		self._plot_line(ax,minv,maxv,self.height)
		if sm.operator in ["le","<="]:
			self._plot_marker(ax,"[",maxv,self.height)
		else:
			self._plot_marker(ax,"]",maxv,self.height)
		ax.text(maxv-0.005,self.height+self.dh/2.*0.9," %s "%sm.arg[0],\
		        horizontalalignment='right',verticalalignment="top",\
			fontsize=6,clip_on=True,rotation="horizontal",zorder=11,
			color="white",weight="bold")
		return minv,maxv
	def _plot_between(self,ax,sm):
		"""
		Routine to plot "between" operator in show mask
		"""
		minv,maxv = (sm.arg[0]-self.minval)/float(self.vr),(sm.arg[1]-self.minval)/float(self.vr)
		if maxv>1 or isnan(maxv): maxv=1	
		if minv<0: minv=0
		self._plot_line(ax,minv,maxv,self.height)
		if sm.operator in ["><","bn"]:
			self._plot_marker(ax,"]",maxv,self.height)
			self._plot_marker(ax,"[",minv,self.height)
		else:
			self._plot_marker(ax,"[",maxv,self.height)
			self._plot_marker(ax,"]",minv,self.height)
		ax.text(minv+0.005,self.height-self.dh/2.*0.9," %s "%sm.arg[0],\
		        horizontalalignment='left',verticalalignment="bottom",\
			fontsize=6,clip_on=True,rotation="horizontal",zorder=11,
			color="white",weight="bold")
		ax.text(maxv-0.005,self.height+self.dh/2.*0.9," %s "%sm.arg[1],\
		        horizontalalignment='right',verticalalignment="top",\
			fontsize=6,clip_on=True,rotation="horizontal",zorder=11,
			color="white",weight="bold")
		return minv,maxv
	def _plot_percent(self, ax, prod,key):
		"""
		This routine plots the percentage on the right site of show_mask plot
		"""
		if prod != "appliedmasks" and prod != "directmasks":
			perc = float(sum(self.db.mask[key].calc_submask(prod)*self.db.mask["Master"].get()))/self.Nevents
			ax.plot([0,perc],[self.height]*2,color=self.db.dp[prod].get("color"),lw=2)
			ax.plot([perc],[self.height],color=self.db.dp[prod].get("color"),ls="",marker="o",markersize=5)
			ax.text(perc+0.05,self.height,"%i"%(perc*100.)+"%",\
				horizontalalignment='left',verticalalignment="center",\
				fontsize=9,clip_on=True,rotation="horizontal",zorder=11)
		elif prod == "appliedmasks":
			if len(self.db.mask[key].appliedmasks)==0:
				perc=1.
			else:
				tmpma=ones(self.db.mask["Master"].get().shape,bool)*self.db.mask["Master"].get()
				for k in self.db.mask[key].appliedmasks:
					tmpma*=self.db.mask[k].get()
				perc = float(sum(tmpma))/self.Nevents
			ax.plot([0,perc],[self.height]*2,color="black",lw=2)
			ax.plot([perc],[self.height],color="black",ls="",marker="o",markersize=5)
			ax.text(perc+0.05,self.height,"%i"%(perc*100.)+"%",\
				horizontalalignment='left',verticalalignment="center",\
				fontsize=9,clip_on=True,rotation="horizontal",zorder=11)
		elif prod == "directmasks":
			if len(self.db.mask[key].directmasks.keys())>0:
				tmpma=ones(self.db.mask["Master"].get().shape,bool)*self.db.mask["Master"].get()
				for k in self.db.mask[key].directmasks.keys():
					tmpma*=self.db.mask[key].directmasks[k]
				perc = float(sum(tmpma))/self.Nevents
				ax.plot([0,perc],[self.height]*2,color="black",lw=2)
				ax.plot([perc],[self.height],color="black",ls="",marker="o",markersize=5)
				ax.text(perc+0.05,self.height,"%i"%(perc*100.)+"%",\
					horizontalalignment='left',verticalalignment="center",\
					fontsize=9,clip_on=True,rotation="horizontal",zorder=11)
			
	def _plot_string(self,ax,prod,key):
		"""
		This routine plots the equal operator for strings
		"""
		vals=unique(self.cdata)
		vals.sort()
		for mindex,sm in enumerate(self.db.mask[key].submasks[prod]): 
			try:
				pos=where(vals==sm.arg[0])[0][0]
			except:
				continue
			ax.text(0.1+pos*0.1,self.height," %s "%vals[pos],\
			        horizontalalignment='left',verticalalignment="center",\
				fontsize=6,clip_on=True,rotation="horizontal",weight="bold",\
				zorder=12,color=self.db.dp[prod].get("color"))
	def plot_ranges(self,pindex,prod,key):
		"""
		This routine is used to plot the particular ranges of the given 
		submask and data product
		INPUT: TODO
		OUTPUT: TODO
		"""
		cdata = self.cdata
		ax = self.axes[key][0]
		ax2 = self.axes[key][1]
		if prod != "appliedmasks" and prod != "directmasks":
			if not isinstance(cdata[0],str):
				self.minval=min(cdata[~isinf(cdata)*~isnan(cdata)])
				self.maxval=max(cdata[~isinf(cdata)*~isnan(cdata)])
				self.vr = self.maxval-self.minval
			self.height = self.ypos[pindex]-self.dh/2.
			for mindex,sm in enumerate(self.db.mask[key].submasks[prod]): 
				minv,maxv = 0, 0
				if sm.operator in [">","gt","ge",">="]:
					minv,maxv = self._plot_greater(ax,sm)
				if sm.operator in ["<","lt","le","<="]:
					minv,maxv = self._plot_lesser(ax,sm)
				if sm.operator in ["><","bn",">=<","be"]:
					minv,maxv = self._plot_between(ax,sm)
				if sm.operator in ["==","eq"] and not isinstance(cdata[0],str):
					minv,maxv = self._plot_equal(ax,sm,prod)
				if isinstance(cdata[0],str):
					self._plot_string(ax,prod,key)
				if "MASK2D" not in prod:
					rec = matplotlib.patches.Rectangle((minv,self.ypos[pindex]-self.dh),\
									   maxv-minv,self.dh, \
									   color=self.db.dp[prod].get("color"))
					ax.add_patch(rec)
					self._plot_percent(ax2,prod,key)
		elif prod == "appliedmasks" or prod == "directmasks":
			self.height = self.ypos[pindex]-self.dh/2.
			self._plot_percent(ax2,prod,key)
			
	def compute(self):
		"""
		This routine computes the mask and performs all necessary operations
		"""
		self.pylabsm2 = pylab.figure("Mask Overview")
		self.pylabsm2.clf()
		if self.smask==False:
			masks = self.db.mask.keys()
		else:
			masks = self.smask
		self.ph = 0.8/len(masks) #Panel height
		# Always show Master on top
		if "Master" in masks:
			masks.remove("Master")
			masks.append("Master")
		for kindex,key in enumerate(masks):
			self.axes[key] = self._prepare_ax(kindex,key)
			products = self.db.mask[key].submasks.keys()
			if len(self.db.mask[key].appliedmasks)>0:
				products.append("appliedmasks")   
			if len(self.db.mask[key].directmasks.keys())>0:
				products.append("directmasks")   
			if len(products)==0 and key=="Master":
				self.Nevents = float(self.db.data[self.db.data.keys()[0]].shape[0])
				Nmask = sum(self.db.get_mask(key))
				self.axes[key][1].set_ylabel("%i/%i \n (%3.1f"%(Nmask,self.Nevents,float(Nmask)/self.Nevents*100.)+"%)",fontsize=9)
				self.axes[key][1].yaxis.set_label_position("right")
				continue
			self.dh = .5/len(products) #Bar Height
			self.ypos=linspace(0.,1.,len(products)+2)[1:-1]+self.dh/2.
			for pindex,product in enumerate(products):
				self.cdata, self.Nevents = self._get_cdata(product,key)
				self._prepare_product_plot(pindex,product,key)
				self.plot_ranges(pindex,product,key)
			Nmask = sum(self.db.get_mask(key))
			self.axes[key][1].set_ylabel("%i/%i \n (%3.1f"%(Nmask,self.Nevents,float(Nmask)/self.Nevents*100.)+"%)",fontsize=9)
			self.axes[key][1].yaxis.set_label_position("right")
class plot_data(object):
	"""
	Class which stores the data generated during plotting, depending on the type of plot:	
		hist1d: Stores bins and histogram as returned by numpy.histogram for every plotted mask in a dictionary
		hist2d: Stores xbins, ybins and histogram as returned by numpy.histogram2d for every plotted mask in a dictionary
		timeseries: Stores xdata and a dictionary of plotted dataproducts for every plotted mask in a dictionary
	"""
	def __init__(self,plot_type, dbData = None):
		"""
		INPUT: plot_type - determined hte plot type of the data (hist1d, hist2d, ...)
		       dbData - instance of the dbData class
		"""
		self.plot_type = plot_type
		self.prodx = None
		self.prody = None
		self.prodz = None
		self.dbData = dbData
		self.data = {}
	def add_data(self,mask,data,prodx=False,prody=False,prodz=False,norm=False):
		"""
		Add data to to the plot_data class
		INPUT: MASK - name of the mask that is added
		       data - data of the mask
		       prodx - x data product label
		       prody - y data product label
		       prodz - z data product label
		"""
		self.prodx = prodx
		self.prody = prody
		self.prodz = prodz
		self.norm = norm
		if self.plot_type in ["hist1D","hist2D","timeseries"]:
			self.data.update({mask:data})

	def get_data(self,smask=False):
		if not smask:
			out = self.data
		elif isinstance(smask,list):
			out = {}
			for mask in smask:
				out.update({mask:self.data[mask]})
		else:
			out = self.data[smask]
		return out
	def save_ascii(self,filename="foobar",lower_lim=0,upper_lim=1e10):
		"""
		Routine to save the data of a given plot into ascii file ('filename')
		INPUT: upper_lim, lower_lim (float, float)
		          -> Data is only saved if lower_lim<value<upper_lim
		"""
		if self.plot_type == "timeseries":
			print('Writing Data into file %s'%(filename+"_"+self.plot_type+".dbData"))
			data_file = open(filename+"_"+self.plot_type+".dbData","w")
			data_file.write("### Header Start \n")
			data_file.write("## Mask Info \n")
			for key in self.prodz:
				data_file.write("Mask: "+self.dbData.mp[key].get("name")+" / Plottype: %s \n"%(self.plot_type))
				submask = self.dbData.mask[key]
				for SM in submask.submasks:
					data_file.write("# %s: "%(self.dbData.dp[SM].get("name")))
					for args in submask.submasks[SM]:
						data_file.write("[op='%s', arg=%s] / "%(args.operator,args.arg))
				data_file.write("\n")
			data_file.write("## Mask Info End \n")
			data_file.write("Prodx:%s   "%self.dbData.dp[self.prodx].get("name"))
			for key in sorted(self.data.keys()): data_file.write("Prody:%s   "%(key))
			data_file.write("\n")
			data_file.write("### Header End \n")
			time = self.data[self.data.keys()[0]][-1]
			for tindex, t in enumerate(time):
				data_file.write("%s   "%(t))
				for key in sorted(self.data.keys()):
					if lower_lim<self.data[key][0][tindex]<upper_lim:
						data_file.write("%s   "%(self.data[key][0][tindex]))
				data_file.write("\n")
			data_file.close()
			return
		for key in self.data:
			print('Writing Data into file %s'%(filename+"_"+key+".dbData"))
			data_file = open(filename+"_"+key+"_"+self.plot_type+".dbData","w")
			data_file.write("### Header Start \n")
			data_file.write("Mask: "+self.dbData.mp[key].get("name")+" / Plottype: %s \n"%(self.plot_type))
			data_file.write("## Mask Info \n")
			submask = self.dbData.mask[key]
			for SM in submask.submasks:
				data_file.write("# %s: "%(self.dbData.dp[SM].get("name")))
				for args in submask.submasks[SM]:
					data_file.write("[op='%s', arg=%s] / "%(args.operator,args.arg))
				data_file.write("\n")
			data_file.write("## Mask Info End \n")
			if self.plot_type == "hist2D":
				data_file.write("Prodx:%s     Prody:%s     Prodz:%s     Norm:%s \n"%(self.dbData.dp[self.prodx].get("name"),self.dbData.dp[self.prody].get("name"),self.prodz,self.norm))
				data_file.write("### Header End \n")
				C,X,Y = self.data[key]
				for xindex,x in enumerate(X[:-1]):
					for yindex,y in enumerate(Y[:-1]):
						if lower_lim<C[xindex,yindex]<upper_lim:
							data_file.write("%s %s %s \n"%(x,y,C[xindex,yindex]))
			if self.plot_type == "hist1D":
				data_file.write("Prodx:%s     Norm:%s \n"%(self.dbData.dp[self.prodx].get("name"),self.norm))
				data_file.write("### Header End \n")
				C,X = self.data[key]
				for xindex,x in enumerate(X[:-1]):
					if lower_lim<C[xindex]<upper_lim:
						data_file.write("%s %s \n"%(x,C[xindex]))
			data_file.close()
class plot_mod(object):
	def __init__(self,pylabfig,style="STD"):
		"""
		Init the plot modifier. This class is used to give the user a backend to modifiy plot created with
		dbData
		"""
		self.fig = pylabfig
		self.style = style
		self.oldobj = self._process_fig()
		self.data = None
		self.CMouse = False
	def connect_mouse(self):
		"""
		This routine connects Mouse clicks to the current active Canvas and prints them on Screen.
		If Mouse is already connected, connection is terminated
		"""
		def _enter(event):
			try:
				MSG = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+": B:%d, (X, Y): (%f, %f)"%(event.button, event.xdata, event.ydata)
				print(MSG)
			except:
				pass
		if not self.CMouse:
			self.CMouse=True
			self.C=self.fig.canvas.mpl_connect('button_press_event',lambda event:_enter(event))
		else:
			self.fig.canvas.mpl_disconnect(self.C)
	def _process_fig(self):
		self.axes = {}
		self.texts = {}
		self.lines = {}
		self.linelabels={}
		self.objects = {}
		for a,axes in enumerate(self.fig.axes):
			self.axes["A%i"%a]=axes
			self.objects["A%i"%a]={}
			self.texts["A%i"%a]={}
			self.lines["A%i"%a]={}
		for a, axes in enumerate(self.axes):
			for t,text in enumerate(self.axes[axes].texts):
				self.objects[axes]["T%i"%t]=text
				self.texts[axes]["T%i"%t]=text
		for a, axes in enumerate(self.axes):
			lines=self.axes[axes].lines
			for l,line in enumerate(lines):
				self.objects[axes]["L%i"%l]=line
				self.lines[axes]["L%i"%l]=line
		return self.objects
	def get_axis(self,keya):
		"""
		Return the choses axis. 
		INPUT: keyword of the axis (Can be looked up with self.show_axes)
		"""
		if keya in self.axes:
			return self.axes[keya]
		else:
			print("Can't find axis '%s'"%(keya))
	def get_object(self,keya,keyo):
		"""
		Return the choses object. 
		INPUT: keya: keyword of the axis (Can be looked up with self.show_axes)
		keyt: keyword of the object (Can be looked up with self.show_axes)
		"""
		if keya in self.objects:
			if keyo in self.objects[keya]:
				return self.objects[keya][keyo]
			else:
				print("Can't find object %s in axis '%s'"%(keya,keyo))
		else:
			print("Can't find axis '%s'"%(keya))
	def del_object(self,keya,keyo):
		"""
		Return the choses object. 
		INPUT: keya: keyword of the axis (Can be looked up with self.show_axes)
		keyt: keyword of the object (Can be looked up with self.show_axes)
		"""
		self.get_object(keya,keyo).remove()
		self._process_fig()
	def set_fontsize(self,size):
		"""
		Set all labels to the given fontsize	
		"""
		matplotlib.rcParams.update({'font.size': size})
	def show_axes(self):
		"""
		This routine shows all available axes with all printed labels in it.
		"""
		self._process_fig()
		print("Available Axes to modify on plot %s:"%(self.style))
		print("|")
		for keya in sorted(self.axes.keys()):
			print("Axis %s:"%(keya))
			print("---------")
			print("   | Labels")
			for keyt in sorted(self.texts[keya].keys()):
				text = self.objects[keya][keyt].get_text()
				print("   ----> %s: '%s'"%(keyt,text))
			print("   | Lines")
			for keyl in sorted(self.lines[keya].keys()):
				linec = self.objects[keya][keyl].get_color()
				lineds = self.objects[keya][keyl].get_drawstyle()
				linels = self.objects[keya][keyl].get_linestyle()
				print("   ----> %s: ls='%s', ds='%s', c='%s'"%(keyl,linels,lineds,linec))
	def set_line(self,*arg,**kwarg):
		"""
		This routine plots lines onto the given axes. It is used just like the normal 
		matplotlib plot command.
		INPUT: arg and kwarg can be the same arguments you would give to pylab.plot
		kwarg may contain one additional keyword 'axes', 
		which may be a list axes keys or an axis key string. If this keyword is not given,
		lineplot is performed on all axis
		"""
		if "axes" in kwarg:
			if isinstance(kwarg["axes"],str):
				Axes=[kwarg["axes"]]
			else:
				Axes=kwarg["axes"]
			kwarg.pop("axes")
		else:
			Axes=self.axes.keys()
		for A in Axes:
			xl=self.axes[A].get_xlim()
			yl=self.axes[A].get_ylim()
			self.axes[A].plot(*arg,**kwarg)
			self.axes[A].set_xlim(xl)
			self.axes[A].set_ylim(yl)
		self.update()
	def set_text(self,*arg,**kwarg):
		"""
		This routine plots text onto the given axes. It is used just like the normal 
		matplotlib text command.
		INPUT: arg and kwarg can be the same arguments you would give to pylab.plot
		kwarg may contain one additional keyword 'axes', 
		which may be a list axes keys or an axis key string. If this keyword is not given,
		lineplot is performed on all axis
		"""
		if "axes" in kwarg:
			if isinstance(kwarg["axes"],str):
				Axes=[kwarg["axes"]]
			else:
				Axes=kwarg["axes"]
			kwarg.pop("axes")
		else:
			Axes=self.axes.keys()
		for A in Axes:
			self.axes[A].text(*arg,**kwarg)
		self.update()
	def reset(self):
		"""
		This method resets the whole figure to its original state 
		"""
		for ax in self.axes:
			for o in self.objects[ax]:
				if not self.objects[ax][o] in self.oldobj[ax].values(): self.del_object(ax,o)
		self._process_fig()
		self.update()
	def update(self):
		"""
		Updates the plot
		"""
		self.fig.show()
	def set_data(self,data):
		"""
		Sets the corresponding plot_data object
		"""
		self.data = data
	def get_data(self):
		"""
		Returns the corresponding plot_data object, if it was added, else None
		"""
		return self.data
	def fit_peak1d(self,mask,lim1,lim2,fun,starts,sigma=None,plotit=True,showres=True,setmax=True,resname=False):     #Umschreiben fuer f(p,x)
		"""This routine fits a function to the data displayed in a two-dimensional plot (y=f(x)).
		INPUT: The mask(string) for which the data shall be fitted. lim1 and lim2(float) represent the lower and upper limit of the x area to fit.
		fun is the model function. It must have the form f(parameters,x), where parameters is a list. starts is a list with a guess of the function parameters.
		plotit=True/False states if the fitted function shall be displayed in the current plot. showres=True/False states if the residual shall be plotted.
		If setmax=True the starting value of the first parameter is set to the maximum value of the fit area. sigma is the standard deviation of the data.
		resname is the key, with that the data can be adressed in the plot_mod object.
		RETURNS the optimal set of parameters and the covariance matrix."""
		plotdata=self.get_data()
		ydat,xdat=plotdata.get_data()[mask]
		if len(xdat)!=len(ydat):
			xdat=xdat[0:-1]
		x=xdat[(xdat>lim1)*(xdat<lim2)]
		y=ydat[(xdat>lim1)*(xdat<lim2)]
		args=(x,y)
		if sigma==None:
			lsqr=lambda paras,xdata,ydata: ydata-fun(paras,xdata)
		else:
			sigma=sigma[(xdat>lim1)*(xdat<lim2)]
			sigma[sigma<=0]=1
			lsqr=lambda paras,xdata,ydata,weights: weights*(ydata-fun(paras,xdata))
			args += (1.0/asarray(sigma),)
		if setmax==True:
			starts[0]=max(ydat) 
		args=leastsq(lsqr,starts,args=args,maxfev=100000,full_output=1)
		res=args[0]
		pcov=args[1]
		yfit = fun(res,x)   # get predicted observations/calc errors
		SSE = sum((y-yfit)**2)
		sig2 = SSE/(len(y)-len(res))
		try:
			ecov = sig2*pcov
		except TypeError:
			ecov=None
		if plotit==True:
			self.set_line(xdat,fun(res,xdat),label="Fit")
			if showres==True:
				self.set_line(xdat,ydat-fun(res,xdat),ls="steps-post",label="Residual")
				if resname:
					self.data.add_data(resname,(ydat-fun(res,xdat),xdat))
			self.update()
		return res, ecov
	def fit_peak1d_spares(self,mask,lims,fun,starts,sigma=None,plotit=True,showres=True,setmax=True,resname=False,addargs=False):     
		"""This routine fits a function to the data displayed in a two-dimensional plot (y=f(x)).
		INPUT: The mask(string) for which the data shall be fitted. lims is list of lists, where the fit area can be specified.
		fun is the model function. It must have the form f(parameters,x), where parameters is a list. starts is a list with a guess of the function parameters.
		plotit=True/False states if the fitted function shall be displayed in the current plot. showres=True/False states if the residual shall be plotted.
		If setmax=True the starting value of the first parameter is set to the maximum value of the fit area. sigma is the standard deviation of the data. resname is the key, 
		with that the data can be adressed in the plot_mod object. Addargs is a tuple of additional arguments for the function.
		RETURNS the optimal set of parameters and the covariance matrix."""
		plotdata=self.get_data()
		ydat,xdat=plotdata.get_data()[mask]
		if len(xdat)!=len(ydat):
			xdat=xdat[0:-1]
		mask2=False
		for l in lims:
			mask2+=(xdat>l[0])*(xdat<l[1])
		
		x=xdat[mask2]
		y=ydat[mask2]
		args=(x,y)
		if sigma==None:
			lsqr=lambda paras,xdata,ydata: ydata-fun(paras,xdata)
		else:
			sigma=sigma[mask2]
			sigma[sigma<=0]=1
			lsqr=lambda paras,xdata,ydata,weights: weights*(ydata-fun(paras,xdata))
			args += (1.0/asarray(sigma),)
			print(args)		
		if addargs and sigma==None:
			lsqr=lambda paras,xdata,ydata,adds: ydata-fun(paras,xdata,adds)
			args+=addargs
			print(args)		
		elif addargs:
			lsqr=lambda paras,xdata,ydata,weights,adds: weights*(ydata-fun(paras,xdata,adds))
			args+=addargs
			print(args)	
		#print "ydata:"
		#print ydata
		#print "fun(paras,xdata,adds):"	
		#print fun(paras,xdata,adds)		
		if setmax==True:
			starts[0]=max(ydat) 
		args=leastsq(lsqr,starts,args=args,maxfev=100000,full_output=1)
		res=args[0]
		pcov=args[1]
		if addargs:
			yfit = fun(res,x,addargs[0])   # get predicted observations/calc errors
		else: yfit = fun(res,x)   
		SSE = sum((y-yfit)**2)
		sig2 = SSE/(len(y)-len(res))
		try:
			ecov = sig2*pcov
		except TypeError:
			ecov=None
		if plotit==True:
			if addargs:
				self.set_line(xdat,fun(res,xdat,addargs[0]),label="Fit")
			else:
				self.set_line(xdat,fun(res,xdat),label="Fit")
			if showres==True:
				if addargs:
					self.set_line(xdat,ydat-fun(res,xdat,addargs[0]),ls="steps-post",label="Residual")
				else:
					self.set_line(xdat,ydat-fun(res,xdat),ls="steps-post",label="Residual")
				if resname:
					if addargs:
						self.data.add_data(resname,(ydat-fun(res,xdat,addargs[0]),xdat))
					else:
						self.data.add_data(resname,(ydat-fun(res,xdat),xdat))
			self.update()
		return res, ecov

	def fit_peak2d(self,mask,key1r,key2r,fun,starts,sigma=None,plotit=True,setmax=True,cb=False):   #----> fun(parameter,x,y)
		"""This routine can fit a distribution function to the data displayed in a three-dimensional plot (z=f(x,y)).
		INPUT: The mask(string) for which the data shall be fitted. key1r and key2r are lists containing the two limits of the xdata and ydata, respectively [lowerlimit,upperlimit].
		fun is a function of the form f(parameters,x,y), where parameters is a list. starts(list) is a guess for the parameters. If setmax=True the starting value of the first parameter is set
		to the maximum value of the fit area. sigma (len(sigma)=len(data in plot)) is the standard deviation of the data. cb states if a colorbar for the contour shall be plotted. It's not that beautiful...
		RETURNS the optimal set of parameters and the covariance matrix."""
		plotdata=self.get_data()
		hdat,xdat,ydat=plotdata.get_data()[mask]
		x=xdat[0:-1]
		y=ydat[0:-1]
		tempmaskx=(x>key1r[0])*(x<key1r[1])
		tempmasky=(y>key2r[0])*(y<key2r[1])
		x=x[tempmaskx]
		y=y[tempmasky]
		hdat=hdat[tempmaskx]
		hist=hdat[::,tempmasky]
		l=len(x)
		ygrid,xgrid=meshgrid(y,x)
		xg,yg=ravel(xgrid),ravel(ygrid)
		h=ravel(hist)
		args=(xg,yg,h)
		print(args)
		if sigma is None:
			lsqr=lambda paras,xdata,ydata,hdata: hdata-fun(paras,xdata,ydata)
		else:
			sigma=sigma[tempmaskx]
			sigma=sigma[::,tempmasky]
			sigma=ravel(sigma)
			sigma[sigma==0]=1
			lsqr=lambda paras,xdata,ydata,hdata,weights: weights*(hdata-fun(paras,xdata,ydata))
			args += (1.0/asarray(sigma),)
			print(args)
		if setmax==True:
			starts[0]=max(h)   
		args=leastsq(lsqr,starts,args=args,maxfev=100000,ftol=1.e-6,full_output=1)
		res=args[0]
		pcov=args[1]
		hfit = fun(res,xg,yg)   # get predicted observations/calc errors
		SSE = sum((h-hfit)**2)
		sig2 = SSE/(len(h)-len(res))
		try:
			ecov = sig2*pcov
		except TypeError:
			ecov=None
		if plotit==True:
			hfit=reshape(hfit,(l,-1)) 
			hfit=swapaxes(hfit,0,1)
			for i in self.axes:
				if "T0" in self.objects[i]:
					lab=self.get_object(i,"T0")
					if lab.get_text()==mask:
						ax=self.get_axis(i) 
						ax.contour(x,y,hfit,4,cmap=pylab.cm.RdYlBu)
						if cb==True:
							axcb = pylab.axes([0.127,0.1,0.02,0.8])  #HERE!!! THESE TWO LINES WERE AT HEIGHT IF LAB.GET_TEXT==MASK
							cb2 = matplotlib.colorbar.ColorbarBase(axcb,cmap=pylab.cm.RdYlBu,orientation="vertical",)
			self.update()
		return res,ecov	


	def fit_peak2d_spares(self,mask,key1r,key2r,fun,starts,sigma=None,plotit=True,setmax=True,cb=False,addargs=False):   #----> fun(parameter,x,y)
		"""This routine can fit a distribution function to the data displayed in a three-dimensional plot (z=f(x,y)).
		INPUT: The mask(string) for which the data shall be fitted. key1r and key2r are lists containing the two limits of the xdata and ydata, respectively [lowerlimit,upperlimit].
		fun is a function of the form f(parameters,x,y), where parameters is a list. starts(list) is a guess for the parameters. If setmax=True the starting value of the first parameter is set
		to the maximum value of the fit area. sigma (len(sigma)=len(data in plot)) is the standard deviation of the data. cb states if a colorbar for the contour shall be plotted. It's not that beautiful...
		RETURNS the optimal set of parameters and the covariance matrix."""
		plotdata=self.get_data()
		hdat,xdat,ydat=plotdata.get_data()[mask]
		x=xdat[0:-1]
		y=ydat[0:-1]
		tempmaskx=(x>key1r[0])*(x<key1r[1])
		tempmasky=(y>key2r[0])*(y<key2r[1])
		x=x[tempmaskx]
		y=y[tempmasky]
		hdat=hdat[tempmaskx]
		hist=hdat[::,tempmasky]
		l=len(x)
		ygrid,xgrid=meshgrid(y,x)
		xg,yg=ravel(xgrid),ravel(ygrid)
		h=ravel(hist)
		args=(xg,yg,h)
		if sigma is None:
			lsqr=lambda paras,xdata,ydata,hdata: hdata-fun(paras,xdata,ydata)
		else:
			sigma=sigma[tempmaskx]
			sigma=sigma[::,tempmasky]
			sigma=ravel(sigma)
			sigma[sigma==0]=1
			lsqr=lambda paras,xdata,ydata,hdata,weights: weights*(hdata-fun(paras,xdata,ydata))
			args += (1.0/asarray(sigma),)
			#print args			
		if addargs and sigma==None:
			lsqr=lambda paras,xdata,ydata,hdata,adds: hdata-fun(paras,xdata,ydata,adds)
			args+=addargs
			#print args			
		elif addargs:
			lsqr=lambda paras,xdata,ydata,hdata,weights,adds: weights*(hdata-fun(paras,xdata,ydata,adds))
			args+=addargs
			#print args	
		if setmax==True:
			starts[0]=max(h)   
		args=leastsq(lsqr,starts,args=args,maxfev=100000,ftol=1.e-6,full_output=1)
		res=args[0]
		#print "final fitparameters"
		#print res
		pcov=args[1]
		if addargs:
			hfit = fun(res,xg,yg,addargs[0])   # get predicted observations/calc errors
		else: hfit = fun(res,xg,yg)   # get predicted observations/calc errors
		SSE = sum((h-hfit)**2)
		sig2 = SSE/(len(h)-len(res))
		try:
			ecov = sig2*pcov
		except TypeError:
			ecov=None
		if plotit==True:
			hfit=reshape(hfit,(l,-1)) 
			hfit=swapaxes(hfit,0,1)
			for i in self.axes:
				if "T0" in self.objects[i]:
					lab=self.get_object(i,"T0")
					if lab.get_text()==mask:
						ax=self.get_axis(i) 
						ax.contour(x,y,hfit,4,cmap=pylab.cm.RdYlBu)
						if cb==True:
							axcb = pylab.axes([0.127,0.1,0.02,0.8])
							cb2 = matplotlib.mpl.colorbar.ColorbarBase(axcb,cmap=pylab.cm.RdYlBu,orientation="vertical",)
			self.update()
		return res,ecov	






		
	def smooth(self,mask,wl,bins):
		"""Routine that smoothes the data in the plot. This means that for every bin the mean of the the neighboring bins is calculated.
		mask is the key of the data to be smoothed.
		wl stand for the windowlength. For example if the windowlength is 3, the mean of the bin and its direct lower and upper neighboring bins is calculated
		as the value of the bin.
		bins is the bin width of the original histogram to be smoothed.
		No return value, but the data is stored in the plot_data object with the key "smooth"."""
		counts, data=self.get_data().get_data()[mask]
		data=data[0:-1]
		newcounts=[]
		if wl%2==0:
			'''for i in arange(wl/2-1,len(counts)-wl/2-1):
		newcounts.append(sum(counts[(i-wl/2+1):i+wl/2+1]))
			newcounts.append(sum(counts[len(counts)-wl/2-1::]))
			newdata=data[(wl/2-1):(-wl/2)]+bins/2.'''
			for i in arange(wl/2-1,len(counts)-wl/2-1):
				newcounts.append(sum(counts[(i-wl/2+1):i+wl/2+1]))
			newcounts.append(sum(counts[len(counts)-wl/2-1::]))
			newdata=data[(wl/2-1):(-wl/2)]+bins/2.
		else:
			for i in arange(wl/2,len(counts)-wl/2-1):
				newcounts.append(sum(counts[(i-wl/2):i+wl/2+1]))
			newcounts.append(float(sum(counts[-wl::])))
			newdata=data[(wl/2):(-wl/2+1)]
		res=(array(newcounts)/float(wl), newdata)
		self.get_data().add_data("smooth",res)
		self.set_line(res[1],res[0],ls="steps-post",label="Smoothed data")
		return 

class plot_properties(object):
	def __init__(self,prod):
		self.props = {}
		self.props["color"]="red"
		self.props["name"]=prod
		self.props["linewidth"]=1
		self.props["linestyle"]='-'
		self.props["marker"]='None'
		self.props["markersize"]=1
		#self.props["contourcolor"]="spectral"
		#self.props["contourcolor"]="jet"
		self.props["contourcolor"]="rainbow"
		self.props["shadeazimuth"]=45
		self.props["shadepolar"]=45
		#self.props["contourcolor2"]="binary"
		self.props["label_loc"]=(0.05,0.05)
	def set_name(self,name):
		self.name=name
	def set(self,prop,val):
		if prop in self.props:
			self.props[prop]=val
		else:
			print("Property %s not available!"%(prop))
	def get_all(self):
		return self.props
	def get(self,prop):
		if prop in self.props:
			return self.props[prop]
		else:
			return False
	def get_name(self):
		return self.name
	def get_hist1d(self):
		tmp = deepcopy(self.props)
		if "name" in tmp: tmp.pop("name")
		tmp["drawstyle"]="steps"
		print(tmp)
		return tmp
class submask(object):
	def __init__(self,key,op,*arg):
		self.key = key
		self.operator = op
		self.arg = arg
	def calc(self,data):
		if isinstance(data,ndarray):
			return self.get_operator()(data,*self.arg)
		else:
			return self.get_operator()(*self.arg)
	def get_operator(self):
		if isinstance(self.operator,str):
			if self.operator.lower() in [">=","ge"]:
				return greater_equal
			if self.operator.lower() in [">","gt"]:
				return greater
			if self.operator.lower() in ["==","eq"]:
				return self._eq
			if self.operator.lower() in ["<=","le"]:
				return less_equal
			if self.operator.lower() in ["<","lt"]:
				return less
			if self.operator.lower() in ["!=","ne"]:
				return not_equal
			if self.operator.lower() in ["><","bn"]:
				return self._bn
			if self.operator.lower() in [">=<","be"]:
				return self._be
			if self.operator.lower() in ["m2d", ]:
				return mask2D.op
		else:
			return self.operator
	def _eq(self,data,arg1):
		return data==arg1
	def _bn(self,data,arg1,arg2):
		return greater(data,arg1)*less(data,arg2)
	def _be(self,data,arg1,arg2):
		return greater_equal(data,arg1)*less_equal(data,arg2)
	def get_range(self):
		if self.operator in [">=","ge"]:
			return self.arg[0],None,True,False,False
		if self.operator in [">","gt"]:
			return self.arg[0],None,False,False,False
		if self.operator in ["==","eq"]:
			return self.arg[0],None,False,False,True
		if self.operator in ["<=","le"]:
			return None,self.arg[0],False,True,False
		if self.operator in ["<","lt"]:
			return None,self.arg[0],False, False,False
		if self.operator in ["><","bn"]:
			return self.arg[0],self.arg[1],False, False,False
		if self.operator in [">=<","be"]:
			return self.arg[0],self.arg[1],True, True,False
		else:
			return None,None,False,False,False
class mask(object):
	"""
	TODO !!!
	"""
	def __init__(self,dbd,name="Default"):
		self.dbd = dbd
		self.ma = ones(self.dbd.data[list(self.dbd.data.keys())[0]].shape[0],dtype=bool)
		self.submasks = {}
		if name == "Master":
			self.appliedmasks = []
		else:
			self.appliedmasks = ["Master"]
		self.directmasks ={}
		self.name = name
	def apply_mask(self,key):
		self.ma=self.ma*self.dbd.get_mask(key)
		return True
	def add_directmask(self,key,dmask):
		if key in self.directmasks.keys():
			print("Direct mask with the name "+key+" already exists! No action performed!")
			return False
		if (type(dmask)!=numpy.ndarray):
			print("Given directmask should be a numpy array (dtype=bool)! No action performed!")
			return False
		elif dmask.dtype!=bool:
			print("Given directmask should be have dtype=bool! No action performed!")
			return False
		elif self.ma.shape[0]!=dmask.shape[0]:
			print("Direct mask has correct type but different lenght than data! No action performed!")
			return False
		else:
			self.directmasks[key]=dmask
			self.calc_mask()
			return True
	def remove_directmask(self,key):
		if key=="all":
			self.directmasks={}
			self.calc_mask()
		elif key in self.directmasks:
			self.directmasks.pop(key,False)
			self.calc_mask()
		else:
			print("No directmask with name "+key+" found! No action performed!")
		return True
	def add_appliedmask(self,key):
		if key in self.appliedmasks:
			print(key+" is already applied! No action performed!")
		elif key == self.name:
			print("Mask cannot be applied to itself! No action performed!")
		elif key not in self.dbd.mask.keys():
			print("Mask "+key+" does not exist! No action performed!")
		elif self.name == "Master":
			print("Master Mask must not have any applied Masks! No action performed!")
		else:
			if not self.check_infinite_loop(key):
				self.appliedmasks.append(key)
				self.calc_mask()
			else:
				print("Mask "+key+" not added to applied masks!")
		return True
	def check_infinite_loop(self,key):
		"""
		This routine checks if appliedmasks would end in infinite loop (or rather an undefined mask setting)! 
		"""
		applied_list=[key]
		stop=False
		while not stop:
			stop=True
			for k in set(applied_list):
				for kk in self.dbd.mask[k].appliedmasks:
					if kk==self.name:
						print(self.name+".check_infinite_loop("+key+"): Mask "+k+" would be applied in Mask "+self.name+" but applies "+self.name+" itself! This would cause infinite loop or undefined state of at least one Mask!")
						return True
					elif kk!="Master":
						applied_list
						while k in applied_list:
							applied_list.remove(k)
						applied_list.append(kk)
						stop=False
					else:
						applied_list.remove(k)
		return False
	def remove_appliedmask(self,key):
		if key in self.appliedmasks:
			if key == "Master":
				print("Master Mask can not be removed from appliedmasks! No action performed!")
			else:
				self.appliedmasks.remove(key)
				self.calc_mask()
		elif key=="all":
			if self.name == "Master":
				self.appliedmasks=[]
			else:
				self.appliedmasks=["Master"]
			self.calc_mask()
		else:
			print(key+" is not in appliedmasks! No action performed!")
		return True
	def add_submask(self,key,operator,*arg,**kwarg):
		if "reset" in kwarg and kwarg["reset"]==True:
			self.remove_submask(key)
		if key not in self.dbd.data  and "MASK2D" not in key:
			print("add_submask : Invalid Data Product (wrong key)")
			return False
		if key in self.submasks:
			self.submasks[key].append(submask(key,operator,*arg))
		else:
			self.submasks[key]=[submask(key,operator,*arg)]
		self.calc_mask()
	def remove_submask(self,key,N=False):
		if key!='all' and N==False:
			self.submasks.pop(key,False)
		elif key!='all':
			self.submasks[key].pop(N)
		else:
			self.submasks={}
		self.calc_mask()
	def calc_mask(self):
		self.ma = ones(self.dbd.data[self.dbd.data.keys()[0]].shape[0],dtype=bool)
		for key in self.submasks:
			tmp_ma = zeros(self.ma.shape[0],dtype=bool)
			for subm in self.submasks[key]:
				if key in self.dbd.data.keys(): #If key is found in self.dbd.data, pass data prod to calc
					tmp_ma = logical_or(tmp_ma,subm.calc(self.dbd.data[key]))
				elif "MASK2D" in key: #If it's a MASK2D mask, pass data[key_1] and data[key_2] to calc
					key_1, key_2 = key.split()[1].split('/')
					tmp_ma = logical_or(tmp_ma,subm.calc(array([self.dbd.data[key_1], self.dbd.data[key_2]])))
				else: # If key not found, pass False (for user defined functions)
					tmp_ma = logical_or(tmp_ma,subm.calc(False))
			self.ma*=tmp_ma
		# Apply all Masks that are in appliedmasks
		for key in self.appliedmasks:
			self.apply_mask(key)
		# Apply all directmasks
		for key in self.directmasks.keys():
			self.ma*=self.directmasks[key]
		# Mask is now up to date now finaly check if Mask is applied in any other Mask and update if applicable
		for key in self.dbd.mask.keys():
			if self.name in self.dbd.mask[key].appliedmasks:
				self.dbd.mask[key].calc_mask()
	def calc_submask(self,prod):
		if prod in self.submasks:
			tmp_ma = zeros(self.ma.shape[0],dtype=bool)
			for subm in self.submasks[prod]:
				tmp_ma = logical_or(tmp_ma,subm.calc(self.dbd.data[prod]))
			return tmp_ma
		else:
			return zeros(self.ma.shape[0],dtype=bool)
	def cleanup_mask(self):
		nkeys=self.dbd.data.keys()
		okeys=self.submasks.keys()
		for sm in okeys:
			if not sm in nkeys:
				self.submasks.pop(sm)
	def update_mask(self):
		self.cleanup_mask()
		self.calc_mask()
	def get(self):
		return self.ma
	def set(self,ma):
		self.ma=ma
class mask1D(object):
	def __init__(self,dbData,mask,prodx,prody):
		"""
		INPUT ###
		dbData: instance of dbData
		prodx: xval for the 1D mask
		prody: prodys used for the timeseries plot 
		"""
		self.db = dbData
		self.mask = mask
		self.prodx = prodx
		self.prody = prody
		self.pX = self.db.data[self.prodx]
		self.M1D = []
		self.state = "NOT APPLIED"
		self.button_state = "SPOINT"
	def create_mask1D(self,**kwargs):
		"""
		Thos routine takes the same **kwargs as timeseries from dbData class
		"""
		self.P = self.db.timeseries(self.prody,prodx=self.prodx,smask=[self.mask],**kwargs)
		self.P.fig.axes[0].set_title("Mask1D of '%s' on mask '%s' (NOT APPLIED)"%(self.prodx,self.mask),backgroundcolor="red")
		self.C2=self.P.fig.canvas.mpl_connect('key_press_event',lambda event: self._enter(event))
	def _enter(self,event):
		xd,yd = event.xdata,event.ydata
		if event.key=="enter":
			print("Processing Filter %s on mask %s - Please Wait"%(self.prodx,self.mask))
			for M in self.M1D:
				self.db.set_mask(self.mask,self.prodx,M[0],M[1])
			print("Processing finished - Filter applied!")
			self.P.fig.axes[0].set_title("Mask2D of '%s' on mask '%s' (Applied)"%(self.prodx,self.mask),backgroundcolor="green")
			self.P.fig.canvas.mpl_disconnect(self.C2)
			self.P.update()
			self.state="APPLIED"
		elif event.key=="y" and self.button_state=="SPOINT":
			self.M1D.append([xd])
			self.button_state="EPOINT"
		elif event.key=="x" and self.button_state=="EPOINT" and xd>self.M1D[-1][0]:
			self.M1D[-1].append(xd)
			self._draw()
			self.button_state="SPOINT"
		elif event.key=="v":
			self.M1D.pop()
			self._draw()
		elif event.key=="c":
			self.P.fig.canvas.mpl_disconnect(self.C2)
			self.P.update()
		else:
			pass
	def _clear(self):
		for ax in self.P.fig.axes:
			ax.patches=[]
		self.P.update()
	def _draw(self):
		self._clear()
		for ax in self.P.fig.axes:
			for M in self.M1D:
				ax.axvspan(M[0],M[1],facecolor="blue",alpha=0.3)
		self.P.update()
class mask2D(object):
	def __init__(self,dbData,mask,prodx,prody):
		"""
		INPUT ###
		dbData: instance of dbData
		prodx: xval for the 2D mask
		prody: yval for the 2D mask
		"""
		self.db = dbData
		self.mask = mask
		self.prodx = prodx
		self.prody = prody
		self.M2D = []
		self.state = "NOT APPLIED"
	def _draw(self):
		for patch in self.P.fig.gca().patches: patch.remove()
		self.P.fig.gca().add_patch(matplotlib.patches.Polygon(self.M2D,facecolor="red",alpha=0.5,fill=True))
		self.P.update()
	def _enter(self,event):
		xd,yd = event.xdata,event.ydata
		if event.key=="enter":
			print("Processing Filter %s/%s on mask %s - Please Wait"%(self.prodx,self.prody,self.mask))
			self.db.set_mask(self.mask,"MASK2D %s/%s"%(self.prodx,self.prody),self.M2D,op="m2d")
			print("Processing finished - Filter applied!")
			self.P.fig.gca().set_title("Mask2D of '%s'/'%s' on mask '%s' (Applied)"%(self.prodx,self.prody,self.mask),backgroundcolor="green")
			self.P.fig.canvas.mpl_disconnect(self.C2)
			self.P.update()
			self.state="APPLIED"
		elif event.key=="a":
			self.M2D.append((xd,yd))
			self._draw()
		elif event.key=="r":
			self.M2D.pop()
			self._draw()
		elif event.key=="c":
			self.P.fig.canvas.mpl_disconnect(self.C2)
			self.P.update()
		else:
			pass
	def create_mask2D(self,**kwargs):
		"""
		Thos routine takes the same **kwargs as 
 from dbData class
		"""
		self.P = self.db.hist2d(self.prodx,self.prody,smask=[self.mask],**kwargs)
		self.P.fig.gca().set_title("Mask2D of '%s'/'%s' on mask '%s' (NOT APPLIED)"%(self.prodx,self.prody,self.mask),backgroundcolor="red")
		self.C2=self.P.fig.canvas.mpl_connect('key_press_event',lambda event: self._enter(event))
	@staticmethod
	def op(data, M2D):
		"""
		This operator returns a 2D mask on data according to the points contained in M2D
		"""
		pX = data[0]
		pY = data[1]
		try:
			return matplotlib.nxutils.points_inside_poly(array((pX,pY)).T,M2D)
		except:
			P = Path(array(M2D))
			return P.contains_points(array([pX,pY]).T)
class DataDict(dict):
	"""
	This class can be used to tinker with the dbData.data dictionary. 
	"""
	def __init__(self,*arg,**kwarg):
		self.linked = {}
		super(DataDict,self).__init__(*arg,**kwarg)
	def __setitem__(self,key,value):
		dict.__setitem__(self,key,value)
	def __getitem__(self,key):
		value = dict.__getitem__(self,key)
		return value
	def compress_data(self,key,linkedKey):
		"""
		AddData = self.__getitem__(key)
		linkedData, linkedIndex = unique(self.__getitem__(linkedKey),return_index=True)
		dict.__setitem__(self,key,AddData[linkedIndex])
		self.linked[key]=linkedKey
		"""
		pass
	def get_unique_data(self,key):
		return dict.__getitem__(self,key)
class dbData(object):
	def __init__(self,*args,**kwargs):
		self.mask = {}
		self.data = DataDict({})
		self.dp ={}
		self.mp ={}
		self.load_data(*args,**kwargs)
		for key in self.data:
			self.data[key]=array(self.data[key])
		if self.data:
			self.add_mask("Master")
		else:
			print("Implement load_data method first!")
			exit()
		self._init_plot_properties()
	def _init_plot_properties(self):
		"""
		Initialize Plot properties
		"""
		for i,key in enumerate(sorted(self.data.keys())):
			self.dp[key] = plot_properties(key)
			self.dp[key].set("color",self._get_color(i))
		for i,key in enumerate(sorted(self.mask.keys())):
			self.mp[key] = plot_properties(key)
			self.mp[key].set("color",self._get_color(i))
	def _get_color(self,i):
		colors = ["red","blue","green","orange","coral","navy","darkgoldenrod","DarkRed","Fuchsia","Maroon","pink","OrangeRed","Olive","Indigo","Firebrick","DimGray","Purple","Plum","PaleGreen","brown","DarkBlue"]*10
		return colors[i-1]
	def __configure_axes(self,kindex,ax,validmasks,Nx,xlabel,ylabel,rot_xticks,prodx,prody):
		"""
		Internal routine to configure axes of hist2d and animate2d plot
		"""
		if kindex>=len(validmasks)-Nx: 
			if not xlabel:
				ax.set_xlabel(self.dp[prodx].get("name"))
			else:
				ax.set_xlabel(xlabel)
			ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
			if rot_xticks:
				for tick in ax.xaxis.get_major_ticks():
					tick.label.set_rotation('vertical')
		else:
			for tick in ax.xaxis.get_major_ticks():
				tick.label.set_visible(False)
			ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
		if (kindex)%Nx==0: 
			if not ylabel:
				ax.set_ylabel(self.dp[prody].get("name"))
			else:
				ax.set_ylabel(ylabel)
			ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
		else:
			ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(prune='both'))
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_visible(False)
	def __create_bins(self,binx,prodx,valx):
		"""
		Internal routine to create bin arrays
		"""
		if not isinstance(binx,ndarray):
			P=self.data[prodx][self.DM()]
			P=P[(P>-1e99)*(P<1e99)]
			if alltrue(valx%1==0):
				Nbins = max(P)-min(P)+2
				if Nbins>binx:Nbins=binx
				binx = linspace(min(P),max(P)+1,Nbins)
			else:
				Nbins = binx
				binx = linspace(min(P),max(P),Nbins+1)
		return binx
	def __process_norm(self,C,norm):
		"""
		Internal routine to processes normalization for hist2d and animate2d
		"""
		def func(C,norm):
			if norm=="ysum":
				for i in range(C.shape[1]):
					if sum(C[:,i])>0:
						C[:,i]/=float(sum(C[:,i]))
			elif norm=="ymax":
				for i in range(C.shape[1]):
					if max(C[:,i])>0:
						C[:,i]/=float(max(C[:,i]))
			elif norm=="xsum":
				for i in range(C.shape[0]):
					if sum(C[i,:])>0:
						C[i,:]/=float(sum(C[i,:]))
			elif norm=="xmax":
				for i in range(C.shape[0]):
					if max(C[i,:])>0:
						C[i,:]/=float(max(C[i,:]))
			elif norm=="sqrt":
				for i in range(C.shape[0]):
					C[i,:]=sqrt((C[i,:]))
			elif norm=="max":
				C=C/float(max(C))
			elif norm=="sum":
				C=C/float(sum(C))
			elif norm=="log":
				C=log10(C)
			return C
		if len(C.shape)==3:
			for i in range(0,C.shape[2]):
				C[:,:,i]=func(C[:,:,i],norm)
		else:
			C=func(C,norm)
		return C
	def __calculate_panels(self,ncols,validmasks):
		"""
		Internal routine to calculate to panel number and format
		"""
		NPlots = len(validmasks)
		if ncols:
			Nx = ncols
			if len(validmasks)%int(ncols)==0:
				Ny = int(len(validmasks)/ncols)
			else:
				Ny = int(len(validmasks)/ncols)+1 
		else:
			if NPlots==1:
				Nx,Ny = 1,1
			elif 2>=NPlots>0:
				Nx,Ny = 2,1
			elif 4>=NPlots>2:
				Nx,Ny = 2,2
			elif 6>=NPlots>4:
				Nx,Ny = 3,2
			elif 10>NPlots>6:
				Nx,Ny = 3,3
			elif 12>=NPlots>=10:
				Nx,Ny = 4,3
			elif 16>=NPlots>12:
				Nx,Ny = 4,4
			else:
				print("Too Many Mask or no Mask - Use less than 10 or at least 1 Mask(s) for histogram2d")
				return
		return Nx,Ny
	def __init_anim_axes(self,ax_anim,zlabel,prodz):
		"""
		Internal routine to create animation control axes
		"""
		ax_anim.xaxis.set_ticks_position("top")
		for tick in ax_anim.get_xticklabels():
			tick.set_fontsize(9)
		if not zlabel:
			ax_anim.set_xlabel(self.dp[prodz].get("name"),size=9)
		else:
			ax_anim.set_xlabel(zlabel,size=9)
		for tick in ax_anim.yaxis.get_major_ticks():
			tick.label.set_visible(False)
			tick.set_visible(False)
	def __prepare_timeseries(self,P,smask,Master):
		"""
		Internal Routine to prepare the timeseries plot. Calculated number panels, processes
		product and mask list to proper format
		"""
		if smask:
			validmasks=smask
			NPlots = len(validmasks)
			if isinstance(P,list): 
				Counter=len(P)
				for product in P:
					if product[0]=='*': Counter-=1
				NPlots*=Counter
				NPlots+=len(P)-Counter
			else:
				P=[P]
		else:
			validmasks = self.mask.keys()
			validmasks.remove("Master")
			if Master!=False:
				validmasks.append("Master")
			NPlots = len(validmasks)
			if isinstance(P,list): 
				Counter=len(P)
				for product in P:
					if product[0]=='*': Counter-=1
				NPlots*=Counter
				NPlots+=len(P)-Counter
			else:
				P=[P]
		return NPlots,P,validmasks
	def __init_anim_control(self):
		def _enter(event):
			if event.key=="8":
				self.Splay=True
			elif event.key=="2":
				self.Splay=False
			elif event.key=="4":
				self.ANIMATION_C-=1
			elif event.key=="6":
				self.ANIMATION_C+=1
		def _enterB(event,Z):
			try:
				if event.inaxes.get_position().x0<0.05001:
					X=event.xdata
					self.ANIMATION_C=searchsorted(Z,X)
			except:
				pass
		return _enter, _enterB
	def get_data(self,mask,prod):
		"""
		Returns an array of prods of a given mask. Note that the Master mask is also applied!
		INPUT: prod - Data product you want to get (a string)
		mask - applied mask for which you get your data product (a string)
		"""
		return self.data[prod][self.get_mask(mask)]
	def get_mask(self,mask):
		"""
		Returns an the current mask. Note that the Master mask is also applied!
		INPUT: 
		mask - mask you want to get (a string)
		"""
		if (isinstance(mask,str)):
			return self.mask[mask].get()
		elif (isinstance(mask,list)):
			ma=self.DM()
			for m in mask:
				ma=ma*self.mask[m].get()
			return ma
		else:
			print("no valid mask!")
	def DM(self):
		""" Returns the Master Mask"""
		return self.mask["Master"].get()
	def set_data_property(self,prod,prop,val):
		"""
		Assigns plot properties to the respective data products 
		"""
		if prod=="all":
			prods = self.data.keys()
		else:
			prods = [prod]
		for prod in prods:
			self.dp[prod].set(prop,val)
	def set_mask_property(self,mask,prop,val):
		"""
		Assigns plot properties to the respective masks
		"""
		if mask=="all":
			masks = self.mask.keys()
			masks.remove("Master")
		else:
			masks = [mask]
		for mask in masks:
			self.mp[mask].set(prop,val)
	def set_global_property(self,prop,val):
		"""
		This Changes global properties of the plot. Here you can change the fontsize 
		of all labels for instance. 
		"""
		try:
			matplotlib.rcParams[prop]=val
		except:
			for key in sorted(matplotlib.rcParams.keys()):
				print(key,": ",matplotlib.rcParams[key])
			print("")
			print("################## !!! ERROR !!! #######################")
			print("Wrong Property used. See the list above for all possible properties and currently set values")
	@ArgParser
	def add_mask(self,name):
		"""
		This routine adds a (or a list of) new mask(s) to the total 
		INPUT : name --> (list of) str that specifies the added masks ; This argument will converted by self._get_namelist
		"""
		for key in name:
			if not key in self.mask.keys():
				self.mask[key] = mask(self,name=key)
				self.mp[key] = plot_properties(key)
				self.mp[key].set("color",self._get_color(len(self.mask)-1))
			else:
				print("add_mask: Mask ",key," already exists, no action performed")
	def add_data(self,key,data):
		"""
		This method adds an additional dataproduct to self.data with keyword 'key'
		"""
		if not key in self.data.keys():
			if len(self.data[self.data.keys()[0]])==len(data):
				self.data[key]=array(data)
				self.dp[key]=plot_properties(key)
				self.dp[key].set("color",self._get_color(len(self.data)-1))
			else:
				print("Supplied data has not the same length (%s) as entries in self.data (%s)"%(len(data),len(self.data[self.data.keys()[0]])))
		else:
			print("Keyword '%s' already exists!"%(key))
			if raw_input("Proceed? [Y/N]").lower() in ["yes","y","jo","ja","j"]:
				print("You have to add data to all keys to maintain a valid data class.")
				if raw_input(" Do you really really want to do this? [Y/N]").lower() in ["yes","y","jo","ja","j"]:
					self.data[key]=append(self.data[key],data)
				else:
					print("That is really very sensible of you. Thanks you for not endangering my structure.")
			else:
				print("That is very sensible of you. Thank you for not endangering my structure.")
	def copy_mask(self,nmask,omask):
		"""
		Copies mask 'osmask' into new mask 'nmask'! If nmask already exists nmask will be overwritten!
		"""
		self.add_mask(nmask)
		self.mask[nmask].submasks=deepcopy(self.mask[omask].submasks)
		if omask=="Master":
			self.mask[nmask].appliedmasks=["Master"]
		else:
			self.mask[nmask].appliedmasks=deepcopy(self.mask[omask].appliedmasks)
		self.mask[nmask].directmasks=deepcopy(self.mask[omask].directmasks)
		self.mask[nmask].calc_mask()	
		self.mp[nmask] = plot_properties(nmask)
		self.mp[nmask].set("color",self._get_color(len(self.mask)-1))
	@ArgParser
	def remove_mask(self,name):
		"""
		This routine removes specified masks. Beware Master mask must not and cannot be removed!
		INPUT : name --> (list of) str that specifies the remnoved masks ; This argument will converted by self._get_namelist
		"""
		if "Master" in name:
			print("Master mask must not be removed!")
			name.remove("Master")
		for mask in name:
			self.mask.pop(mask)
			self.mp.pop(mask)
	@ArgParser
	def remove_submask(self,name,subname):
		"""
		This routine removes specified submask of masks.
		INPUT : name --> (list of) str that specifies the masks on which submask will be removed
		subname --> name of the submask to be removed
		"""
		for mask in name:
			self.mask[mask].remove_submask(subname)
	@ArgParser
	def remove_appliedmask(self,name,subname):
		"""
		This routine removes specified submask of masks.
		INPUT : name --> (list of) str that specifies the masks on which submask will be removed
		subname --> name of the submask to be removed
		"""
		for mask in name:
			self.mask[mask].remove_appliedmask(subname)
	@ArgParser
	def remove_directmask(self,name,subname):
		"""
		This routine removes specified submask of masks.
		INPUT : name --> (list of) str that specifies the masks on which submask will be removed
		subname --> name of the submask to be removed
		"""
		for mask in name:
			self.mask[mask].remove_directmask(subname)
	@ArgParser
	def reset_mask(self,name):
		"""
		This routine resets specified masks.
		INPUT : name --> (list of) str that specifies the reseted masks ; This argument will converted by self._get_namelist
		"""
		for mask in name:
			self.mask[mask].remove_submask('all')
			self.mask[mask].remove_appliedmask('all')
			self.mask[mask].remove_directmask('all')
	def show_mask(self,smask=False,cmask="Master"):
		SM = show_mask(self,smask=smask,cmask=cmask)
		SM.compute()
	@ArgParser
	def set_mask(self,name,prod,*arg,**kwarg):
		"""
		This Routine sets the mask
		INPUT: name: name of the mask
		prod: dataproduct key or maskname -> maskname must be accompanied with kwarg applied=True otherwise it prod is handled as dataproduct key
		*arg (list of arguments for the operator)
		**kwarg:
		op: Standard is set to 'between two values'
		possible operators are in general every logic comparators or 
		built-in/user-defined functions
		reset: If set to True all previous masks of the given prod are deleted (has no effect if direct or applied is True
		direct: If set True adds given boolean mask 'arg[0]' with label 'prod]' to name.directmasks
		applied: If set True adds given mask 'prod' to name.appliedmasks
		"""
		for mask in name:
			if "applied" in kwarg:
				if kwarg["applied"]==True:
					self.mask[mask].add_appliedmask(prod)
			elif "direct" in kwarg:
				if kwarg["direct"]==True:
					self.mask[mask].add_directmask(prod,arg[0])
			else:
				if "op" not in kwarg:
					self.mask[mask].add_submask(prod,"be",*arg,**kwarg)
				else:
					self.mask[mask].add_submask(prod,kwarg["op"],*arg,**kwarg)
	def set_mask2D(self,name,prodx,prody,*arg,**kwarg):
		"""
		This Routine sets a 2Dimensional mask, which is achieved by drawing a polygron onto a 2d contour
		INPUT: name: name of the mask
		prodx: 1st dataproduct key
		prody: 2nd dataproduct key
		arg,kwarg: additional keywords for 2dhist plot
		"""
		print("HowTO")
		print("----------")
		print("'a': Add Point")
		print("'r': Remove Last Point")
		print("'c': Cancel")
		print("'Enter': Apply Mask")
		M = mask2D(self,name,prodx,prody)
		M.create_mask2D(*arg,**kwarg)
	def set_mask1D(self,name,prody,prodx="time",*arg,**kwarg):
		"""
		This Routine sets a 2Dimensional mask, which is achieved by drawing a polygron onto a 2d contour
		INPUT: name: name of the mask
		prodx: 1st dataproduct key
		prody: dataproducts used for timeseries plot (can be a list)
		arg,kwarg: additional keywords for timeseries plot
		"""
		print("HowTO")
		print("----------")
		print("'y': Add Start Point of Time Frame")
		print("'x': Add End Point of Time Frame")
		print("'v': Remove Last Time Frame")
		print("c': Cancel")
		print("'Enter': Apply Mask")
		M = mask1D(self,name,prodx,prody)
		M.create_mask1D(*arg,**kwarg)
	def save_all(self, filename, overwrite=False):
		"""
		This routine saves the complete dbData instance including data, masks etc.
		INPUT:     (/path/to/)filename as string
		overwrite: enable with True if you wish to overwrite a potentially existing file with the 
		           same name
		"""
		if not overwrite and os.path.isfile(filename):
			print("File '{}' already exists; choose another filename or enable overwrite.".format(filename))
		else:
			of = open(filename, 'wb')
			cPickle.dump(self, of, -1)
			of.close()
			print("Everything successfully saved to file '{}'.".format(filename))
		return
	def save_masks(self, filename, overwrite=False):
		"""
		This routine saves dbData masks except the 'Master' masks.
		INPUT:     (/path/to/)filename as string
		overwrite: enable with True if you wish to overwrite a potentially existing file with the 
		           same name
		"""
		if not overwrite and os.path.isfile(filename):
			print("File '{}' already exists; choose another filename or enable overwrite.".format(filename))
		else:
			mask_dict = deepcopy(self.mask)
			del(mask_dict['Master'])
			for key in mask_dict.keys():
				del(mask_dict[key].dbd)
				del(mask_dict[key].ma)
				mask_dict[key].appliedmasks.remove('Master')
			of = open(filename, 'wb')
			cPickle.dump(mask_dict, of, -1)
			of.close()
			print("Masks successfully saved to file '{}'.".format(filename))
		return
	def save_subset(self,mask,filename="tmp.dat",prods="all"):
		"""
		This routine save all data of a current mask into a given file. 
		It can be loaded later to work with a reduced data set
		INPUT: mask - the mask that should be saved as a subset
		filename - filepath+filename to which the subset is saved
		prods - either "all" or a list of products
		"""
		subset = {}
		if prods=="all":
			for prod in self.data:
				subset[prod] = self.data[prod][self.get_mask(mask)]
		else:
			for prod in prods:
				subset[prod] = self.data[prod][self.get_mask(mask)]
		cPickle.dump(subset,open(filename,"w"))
	@staticmethod
	def load_all(filename):
		"""
		This routine loads a complete dbData instance including data, masks etc.
		It is a staticmethod, meaning that you call it with dbData.load_all(...)
		In case you inherited dbData in one of your own classes, you'll need to use that
		class's name instead.
		INPUT:  (/path/to/)filename as string
		OUTPUT: dbData instance
		"""
		if os.path.isfile(filename):
			with open(filename, 'rb') as of:
				return cPickle.load(of)
		else:
			print("'{}' doesn't exist.".format(filename))
		return
	def load_masks(self, filename, overwrite=False):
		"""
		This routine loads previously saved masks.
		INPUT:  (/path/to/)filename as string
		overwrite: enable with True if you wish to overwrite a potentially existing mask with the 
		           same name
		"""
		if os.path.isfile(filename):
			of = open(filename, 'rb')
			mask_dict = cPickle.load(of)
			of.close()
			existing_masks = []
			if type(mask_dict) != dict:
				print("'{}' does not contain masks, load aborted.".format(filename))
				return
			for key in mask_dict:
				if key not in self.mask.keys():
					self.add_mask(key)
					self.mask[key].submasks = mask_dict[key].submasks
					self.mask[key].directmasks = mask_dict[key].directmasks
					self.mask[key].appliedmasks += mask_dict[key].appliedmasks
				elif overwrite:
					self.remove_mask(key)
					self.add_mask(key)
					self.mask[key].submasks = mask_dict[key].submasks
					self.mask[key].directmasks = mask_dict[key].directmasks
					self.mask[key].appliedmasks += mask_dict[key].appliedmasks
					existing_masks.append(key)
				else:
					existing_masks.append(key)
			if existing_masks != []:
				print("The masks:")
				for mask in existing_masks:
					print("  '{}'".format(mask))
				if overwrite:
					print("already existed and have been overwritten.")
				else:
					print("already exist and have not been replaced. If you wish to load all masks from '{}', enable overwrite or rename your existing masks (call copy_mask and then remove_mask on the masks in question).".format(filename))
			dir_mask = []
			dir_mask_len_nok = []
			for key in self.mask:
				if self.mask[key].directmasks:
					dir_mask.append(key)
					for key2 in self.mask[key].directmasks:
						if len(self.mask[key].directmasks[key2]) != len(self.mask[key].dbd.data.values()[0]):
							dir_mask_len_nok += [[key, key2]]
			if dir_mask:
				print("WARNING: The masks:")
				for mask in dir_mask:
					print("\t'{}'".format(mask))
				print("contain direct (binary) masks; be sure you know what you do.")
				if dir_mask_len_nok:
					print("ERROR: The direct masks:")
					for m1, m2 in dir_mask_len_nok:
						print("\t'{}' in '{}'".format(m2, m1))
					print("do not correspond to the length of your dataset and therefore cannot be used.\nPlease also check your applied masks, if they depend on any of these direct masks\nan error may only be thrown if your dataset is shorter in length than\nthe binary mask.")
			calc_mask_nok = False
			for key in self.mask:
				try:
					if key not in dir_mask:
						self.mask[key].calc_mask()
				except:
					print("ERROR: Calculating mask '{}' failed. Check your direct masks.".format(key))
		else:
			print("'{}' doesn't exist.".format(filename))
		return
	def load_subset(self,filename="tmp.dat",force=False):
		"""
		Loads a subset that is used for further analysis
		!!! This overrides all data loaded by method self.load_data() !!!
		INPUT: mask: 
		"""
		subset=cPickle.load(open(filename,"r"))
		print("!!! Warning !!! - If you Proceed current data set will be overwritten!")
		if force==True or raw_input("Proceed? [Y/N]").lower() in ["yes","y","jo","ja","j"]:
			keys=self.data.keys()
			# Replace old self.data with the loaded data
			for key in keys:
				self.data.pop(key)
			for key in subset:
				self.data[key]=subset[key]
			# Now deal with masks
			for mask in self.mask:
				self.mask[mask].update_mask()
			pkeys=self.dp.keys()
			dkeys=self.data.keys()
			for key in pkeys:
				if not key in dkeys:
					self.dp.pop(key)
			pkeys=self.dp.keys()
			for key in dkeys:
				if not key in pkeys:
					self.dp[key] = plot_properties(key)
					self.dp[key].set("color",self._get_color(len(self.dp)-1))
				#self.mask[mask].calc_mask()
			return True
		else:
			print("Loading subset has been canceled!")
			return False
	def timeseries(self,prody,prodx="time",time=None,smask=False,avg=False,weights=False,mode="mean",Master=True):
		"""
		This routine creates timeseries of various data products
		INPUT:

		prodx: key of self.data over which the series is plotted (prob. 'time')
		prody: key of self.data that is plotted over time, can also be a list of products.
		       If prody starts with '*' this dataproduct is ONLY plotted for the first entry
		       in smask. E.g prody="vsw" means vsw is plotted for all masks, while for prody='*vsw'
		       vsw is only plotted for the first mask in smask
		time: time bins used for time series - must a ndarray
		smask: list of masks that should be plotted
		weights: weights for the timeseries
		mode: Either a string or a list of strings -->
			'mean': mean of prody inside time bin is plotted over time
			'sum' : sum of prody inside time bin is plotted over time 
			'freq' : number of occurences of prody inside time bin is plotted over time 
		avg: number of bins (to the left AND right) used for a slindig average
		sliding average is only plotted if avg is passed

		"""

		
		pylabts = pylab.figure()
		pylab.subplots_adjust(hspace=0,wspace=0)
		pdata = plot_data("timeseries",dbData=self)
		P = prody
		if time == None:
			untimes = unique(self.data[prodx])
			time = append(untimes,untimes[-1]+(untimes[-1]-untimes[-2]))
		NPlots,P,validmasks = self.__prepare_timeseries(P,smask,Master)
		if isinstance(mode,list):
			if len(mode)!=len(P): 
				print("Not enough Modes (N=%s) supplied for timeseries of Prods (N=%s)"%(len(mode),len(P)))
				return
		else:
			mode=len(P)*[mode]
		PCounter = 0
		axtmp = False
		for kindex,key in enumerate(validmasks):
			for pindex, prody in enumerate(P):
				if prody[0]=="*":
					prody=prody[1:]
					if kindex!=0: continue
				mL = matplotlib.ticker.AutoMinorLocator()
				PCounter+=1
				if axtmp:
					ax = pylabts.add_subplot(NPlots,1,PCounter,sharex=axtmp)
				else:
					ax = pylabts.add_subplot(NPlots,1,PCounter)
				axtmp=ax
				if PCounter==NPlots:
					ax.set_xlabel(self.dp[prodx].get("name"))
				else:
					for tick in ax.xaxis.get_major_ticks():
						tick.label.set_visible(False)
				ax.set_ylabel(self.dp[prody].get("name"),color="white",backgroundcolor=self.dp[prody].get("color"))
				valx = self.data[prodx][self.get_mask(key)]
				valy = self.data[prody][self.get_mask(key)]
				if weights:
					valw = self.data[weights][self.get_mask(key)]
					Counts,binedges = histogram(valx,bins=time,weights=valw)
				else:
					Counts,binedges = histogram(valx,bins=time)
				ProdY,binedges = histogram(valx,bins=time,weights=valy)
				if mode[pindex]=="mean":
					plotres = ProdY/(Counts*1.)
				elif mode[pindex]=="sum":
					plotres = ProdY
				elif mode[pindex]=="freq":
					plotres = Counts
				else:
					print("Please enter valid mode!")
					return
				pdata.add_data(key+":"+self.dp[prody].get("name"),(plotres,time[:-1]),prodx=prodx,prody=P,prodz=validmasks,norm=(mode,avg))
				ticks=ax.yaxis.get_major_ticks()
				ticks[0].label.set_visible(False)
				if avg:
					avg_arr = zeros(len(plotres))
					for tindex in range(avg,len(time)-avg):
						dat = plotres[tindex-avg:tindex+avg]
						avg_arr[tindex] = average(dat[isnan(dat)==False])
					ax.plot(time[:-1],plotres,color=self.dp[prody].get("color"),marker="o",markersize=self.dp[prody].get("markersize"),linestyle="")
					ax.plot(time[:-1],avg_arr,c=self.dp[prody].get("color"),lw = self.dp[prody].get("linewidth"),ls = self.dp[prody].get("linestyle"))
				else:
					ax.plot(time[:-1],plotres,c=self.dp[prody].get("color"),lw = self.dp[prody].get("linewidth"),ls = self.dp[prody].get("linestyle"),marker=self.dp[prody].get("marker"),markersize=self.dp[prody].get("markersize"))
				ax.text(1.02,0.5,self.mp[key].get("name"),horizontalalignment='center',verticalalignment="center",transform=ax.transAxes,backgroundcolor=self.mp[key].get("color"),color="white",rotation="vertical")
				ax.set_xlim(time[0],time[-1])
				ax.xaxis.set_minor_locator(mL)
				ax.xaxis.grid(True,which="both")
				ax.yaxis.grid(True)
		pmod=plot_mod(pylabts,style="timeseries")
		pmod.set_data(pdata)
		return pmod
	
	def hist2d(self,prodx,prody,prodz=False,binx=50,biny=50,norm=False,cb=False,weights=False,smask=False,ncols=False,Master=True,xlabel=False,ylabel=False,rot_xticks=True, style = "both",shade=False,clines=1,show_contourlines=True):
		"""
		This routine creates 2D Histograms of two different data products
		TODO: Use proper automatic bins
		INPUT:
		prodx: key of self.data to be histogrammed onto x-axis
		prody: key of self.data to be histogrammed onto y-axis
		prodz: if a prod is given, the averaged product prodz over prodx
		       and prody will be plotted
		binx: xbins for the histogram (number or ndarray)
		biny: ybins for the histogram (number or ndarray)
		norm: can be either 'xmax','xsum','ymax', 'ysum', 'max', 'sum', or 'log'
		smask: list of masks that should be plotted
		rot_xticks: If True xticks are rotated for plot
		weights: weights for the histogram
		cb: a string for the colorbar - The colrbar is only plotted if a non empty string is passed
		ncols: number of columns for the 2d histogram
		style: can be 'contour', 'contourline', or 'both'
		shade: True or False, Turns on/off the shading plot option, which illuminates the contour.
		       Parameters 'shadeazimuth' and 'shadepolar' in mask properties determine the
		       direction from which the contour is illuminated
		Master: If True, the Master mask is plotted
		clines: number of contourlines for the contour plot

		"""
		pylab2dh = pylab.figure()
		pdata = plot_data("hist2D",self)
		pylab.subplots_adjust(hspace=0,wspace=0)
		if smask:
			validmasks = smask
		else:
			validmasks = self.mask.keys()
			validmasks.remove("Master")
			if Master!=False:
				validmasks.append("Master")
		Nx,Ny=self.__calculate_panels(ncols,validmasks)
		MINX, MAXX = False, False
		MINY, MAXY = False, False
		axtmp = False
		for kindex,key in enumerate(validmasks):
			if axtmp:
				ax = pylab2dh.add_subplot(Ny,Nx,kindex+1,sharex=axtmp,sharey=axtmp)
			else:
				ax = pylab2dh.add_subplot(Ny,Nx,kindex+1)
			axtmp = ax
			self.__configure_axes(kindex,ax,validmasks,Nx,xlabel,ylabel,rot_xticks,prodx,prody)
			
			valx = self.data[prodx][self.get_mask(key)]
			valy = self.data[prody][self.get_mask(key)]
			
			binx=self.__create_bins(binx,prodx,valx)
			biny=self.__create_bins(biny,prody,valy)
			
			if weights:
				C,X,Y = histogram2d(valx,valy,bins=[binx,biny],weights=self.data[weights][self.get_mask(key)])
			elif prodz:
				C,X,Y = histogram2d(valx,valy,bins=[binx,biny],weights=self.data[prodz][self.get_mask(key)])
				CFREQ,X,Y = histogram2d(valx,valy,bins=[binx,biny])
				C=C/CFREQ
			elif prodz and weights:
				C,X,Y = histogram2d(valx,valy,bins=[binx,biny],weights=self.data[prodz][self.get_mask(key)]*self.data[prodz][self.get_mask(key)])
				CFREQ,X,Y = histogram2d(valx,valy,bins=[binx,biny])
				C=C/CFREQ
			else:
				C,X,Y = histogram2d(valx,valy,bins=[binx,biny])
			pdata.add_data(key,(C,X,Y),prodx,prody,prodz,norm)
			if MINX==False or MINX>min(X): MINX=min(X)
			if MAXX==False or MAXX<max(X): MAXX=max(X)
			if MINY==False or MINY>min(Y): MINY=min(Y)
			if MAXY==False or MAXY<max(Y): MAXY=max(Y)
			C[isnan(C)]=-1
			C=1.*C
			#print "check C"
			#print C, max(C)
			if norm:
				C=self.__process_norm(C,norm)
			colormap = pylab.cm.get_cmap(self.mp[key].get("contourcolor"),1024*16)
			
			if style=="contour" or style=="both":
				if shade:
					ls = matplotlib.colors.LightSource(azdeg=self.mp[key].get("shadeazimuth"),altdeg=self.mp[key].get("shadepolar"))
					rgb = ls.shade(C.T,colormap)
					ax.imshow(rgb,extent=(X[0],X[-1],Y[-1],Y[0]),aspect="auto",alpha=0.7)
					vmin=min(C[C>0])
					vmax=max(C[C>0])
					#print "vmin, vmax:"
					#print vmin, vmax
					ax.pcolormesh(X,Y,C.T,cmap=colormap,vmin=min(C[C>0.1]),vmax=max(C[C>1.1]),zorder=0)
					colormap.set_under('white')
					cs.cmap.set_under('white')	
				else:
					#print "check vmin, vmax"
					#print min(C[C>0]), max(C)
					if max(C)<=1.0:
						MC=2.0
						cs=ax.pcolormesh(X,Y,C.T,cmap=colormap,vmin=min(C[C>0]),vmax=MC)
					else:
						cs=ax.pcolormesh(X,Y,C.T,cmap=colormap,vmin=min(C[C>0]),vmax=max(C))
					#cs.colormap.set_under('white')	
					#print dir(cs.cmap)
					cs.cmap.set_under('white')		
			if show_contourlines==True:
				if style=="both" or style=="contourline":
					C[C<0]=0
					G = ndimage.gaussian_filter(C,2)
				if style=="both":
					MeshL=ax.contour(X[:-1], Y[:-1], G.T,clines,colors="white",alpha=0.5,vmin=min(C[G>0]),vmax=max(G))
					pylab.clabel(MeshL,fontsize=9,inline=True,alpha=1.0,fmt="%4.1f")
				if style=="contourline":
					MeshL=ax.contour(X[:-1], Y[:-1], G.T,clines,cmap=colormap,alpha=1.0,vmin=min(G[G>0]),vmax=max(G))
					pylab.clabel(MeshL,fontsize=9,inline=True,alpha=1.0,fmt="%4.1f")
			else:
				pass		
			colormap.set_under('white')
			ax.text(self.mp[key].get("label_loc")[0],self.mp[key].get("label_loc")[1],self.mp[key].get("name"),horizontalalignment='left',verticalalignment="bottom",transform=ax.transAxes,backgroundcolor=self.mp[key].get("color"),color="white")
			ax.xaxis.grid(True)
		ax.yaxis.grid(True)
		for ax in pylab2dh.axes:
			ax.set_xlim(min(X),max(X))
			ax.set_ylim(min(Y),max(Y))
			ax.tick_params(axis='both', which='major', labelsize=20)
			ax.set_xlabel(xlabel,fontsize=25)
			ax.set_ylabel(ylabel,fontsize=25)
		
		if cb:
			axcb = pylab.axes([0.9,0.1,0.02,0.8])
			if prodz:
				MINZ, MAXZ = min(C),max(C)
				cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,orientation="vertical",norm=matplotlib.colors.Normalize(vmin=MINZ,vmax=MAXZ))
			else:
				#print min(C[C>0]), max(C)
				pass
		if norm=="max":		
			cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,orientation="vertical")
		else:
			cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,norm=matplotlib.colors.Normalize(vmin=min(C[C>0]), vmax=max(C)),orientation="vertical")
			cb1.set_label("%s"%(cb), fontsize=25)
		if max(C)<=1.0:
			cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,norm=matplotlib.colors.Normalize(vmin=min(C[C>0]), vmax=MC),orientation="vertical")
			cb1.set_label("%s"%(cb), fontsize=25)		
		for t in cb1.ax.get_yticklabels():
			t.set_fontsize(20)
		pmod = plot_mod(pylab2dh,style="hist2D")
		pmod.set_data(pdata)
		return pmod

	def animate2d(self,prodx,prody,prodz,binx=50,biny=50,binz=50,speed=100,norm=False,cb=False,weights=False,smask=False,ncols=False,Master=True,xlabel=False,ylabel=False, zlabel=False,rot_xticks=True,save='',fps=5):
		"""
		This routine creates animated 2D Histograms of two different data products over a third
		Data Product
		TODO: Use proper automatic bins
		INPUT:
		prodx: key of self.data to be histogrammed onto x-axis
		prody: key of self.data to be histogrammed onto y-axis
		prodz: key of self.data to be histogrammed used for the animation frames
		binx: xbins for the histogram (number or ndarray)
		biny: ybins for the histogram (number or ndarray)
		binz: zbins for the aimation
		speed: Animation speed (small = Fast)
		norm: can be either 'xmax','xsum','ymax', 'ysum', 'max', 'sum', or 'log'
		smask: list of masks that should be plotted
		rot_xticks: If True xticks are rotated for plot
		weights: weights for the histogram
		cb: a string for the colorbar - The colrbar is only plotted if a non empty string is passed
		ncols: number of columns for the 2d histogram
		Master: If True, the Master mask is plotted

		"""
		pylab2da = pylab.figure()
		ax_anim = pylab.axes([0.05,0.94,0.9,0.02])
		self.__init_anim_axes(ax_anim,zlabel,prodz)
		pylab.subplots_adjust(hspace=0,wspace=0)
		if smask:
			validmasks = smask
		else:
			validmasks = self.mask.keys()
			validmasks.remove("Master")
			if Master!=False:
				validmasks.append("Master")
		Nx,Ny=self.__calculate_panels(ncols,validmasks)
		MINX, MAXX = False, False
		MINY, MAXY = False, False
		axtmp = False
		DATACONT = {}
		DATA = {}
		for kindex,key in enumerate(validmasks):
			if axtmp:
				ax = pylab2da.add_subplot(Ny,Nx,kindex+1,sharex=axtmp,sharey=axtmp)
			else:
				ax = pylab2da.add_subplot(Ny,Nx,kindex+1)
			axtmp = ax
			self.__configure_axes(kindex,ax,validmasks,Nx,xlabel,ylabel,rot_xticks,prodx,prody)
			
			valx = self.data[prodx][self.get_mask(key)]
			valy = self.data[prody][self.get_mask(key)]
			valz = self.data[prodz][self.get_mask(key)]
			
			binx=self.__create_bins(binx,prodx,valx)
			biny=self.__create_bins(biny,prody,valy)

			C,edges = histogramdd([valx,valy,valz],bins=[binx,biny,binz])
			X,Y,Z = edges
			C[isnan(C)]=-1
			C=1.*C
			if MINX==False or MINX>min(X): MINX=min(X)
			if MAXX==False or MAXX<max(X): MAXX=max(X)
			if MINY==False or MINY>min(Y): MINY=min(Y)
			if MAXY==False or MAXY<max(Y): MAXY=max(Y)
			C[isnan(C)]=-1
			C=1.*C
			if norm:
				C=self.__process_norm(C,norm)
			DATA[key] = C
			colormap = pylab.cm.get_cmap(self.mp[key].get("contourcolor"),1024*16)
			DATACONT[key]=ax.pcolormesh(X,Y,C[:,:,0].T,cmap=colormap,vmin=min(C[C>0]),vmax=max(C))
			colormap.set_under('white')
			ax.text(self.mp[key].get("label_loc")[0],self.mp[key].get("label_loc")[1],self.mp[key].get("name"),horizontalalignment='left',verticalalignment="bottom",transform=ax.transAxes,backgroundcolor=self.mp[key].get("color"),color="white")
			ax.xaxis.grid(True)
			ax.yaxis.grid(True)
		for ax in pylab2da.axes:
			ax.set_xlim(min(X),max(X))
			ax.set_ylim(min(Y),max(Y))
		ax_anim.set_xlim(min(Z),max(Z))
		ax_anim.set_ylim(0,1)
		ANIMCONT=ax_anim.plot([min(Z)],[0.5],linestyle="None",marker="o",markersize=6)
		if cb:
			axcb = pylab.axes([0.9,0.1,0.02,0.8])
			if prodz:
				MINZ, MAXZ = min(C),max(C)
				cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,orientation="vertical",norm=matplotlib.colors.Normalize(vmin=MINZ,vmax=MAXZ))
			else:
				cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,orientation="vertical",)
			cb1.set_label("%s"%(cb))
		
		# Here the Animation logic starts
		self.ANIMATION_C = -1
		ZW = (Z[1]-Z[0])/2.
		self.Splay = True
		_enter,_enterB=self.__init_anim_control()
		def animate(i):
			if self.Splay:
				self.ANIMATION_C+=1
			self.ANIMATION_C=self.ANIMATION_C%(len(Z)-1)
			ANIMCONT[0].set_data([Z[self.ANIMATION_C]],[0.5])
			for key in validmasks:
				z = DATA[key][:,:,self.ANIMATION_C].T
				DATACONT[key].set_array(z.ravel())
			return 0,
		self.C2=pylab2da.canvas.mpl_connect('key_press_event',lambda event: _enter(event))
		self.C2=pylab2da.canvas.mpl_connect('motion_notify_event',lambda event: _enterB(event,Z))
		print("SMALL HOWTO: ANIMATED2D")
		print("------------------------")
		print("Use NUMPAD arrowkeys to control animation.")
		print("Hover Mouse over animation bar, to jump to that location!")
		anim = matplotlib.animation.FuncAnimation(pylab2da,animate,frames=Z.shape[0]-1,interval=speed,blit=False)
		if save:
			anim.save(save,fps=fps,codec="mpeg4")
		return anim
	def quicklook(self,prods=False,cb=False,mask=False,bins=50):
		"""
		This routine creates a quicklook of all available data products
		INPUT:
		prods: List of Products to be plotted
		cb: a string for the colorbar - The colrbar is only plotted if a non empty string is passed
		mask: a string of the mask used for quicklook - if False the Master mask is used
		bins: maximum number of bins for histogram plotting

		"""
		if prods:
			pass
		else:
			prods=self.data.keys()
		N = len(prods)
		pylabql = pylab.figure(figsize=(N,N+1))
		pylab.subplots_adjust(hspace=0,wspace=0)
		pC = 0
		print("Processing Data")
		for yN,ykey in enumerate(prods):
			for xN,xkey in enumerate(prods):
				pC+=1
				#loading_bar(N*N,pC,">")
				ax = pylabql.add_subplot(N+1,N,pC)
				if pC<=N:
					ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
					ax.xaxis.set_ticks_position('top')
					ax.xaxis.set_label_position('top')
					for tick in ax.xaxis.get_ticklabels():
						tick.set_rotation('vertical')
						tick.set_size(6)

				else:
					ax.xaxis.set_ticklabels([])
					ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
				if N-xN==N: 
					ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
					for tick in ax.yaxis.get_ticklabels():
						tick.set_size(6)
				elif xN+1==N and not cb:
					ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
					ax.yaxis.set_ticks_position('right')
					ax.yaxis.set_label_position('right')
					for tick in ax.yaxis.get_ticklabels():
						tick.set_size(6)
				else:
					ax.yaxis.set_ticklabels([])
					ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
				if mask:
					valx = self.data[xkey][self.get_mask(mask)]
					valy = self.data[ykey][self.get_mask(mask)]
				else:
					valx = self.data[xkey][self.DM()]
					valy = self.data[ykey][self.DM()]
				try:
					binx, biny = bins, bins
					if alltrue(valx%1==0):
						Nbins = max(valx)-min(valx)+2
						if Nbins>bins:Nbins=bins
						binx = linspace(min(valx),max(valx)+1,Nbins)
					if alltrue(valy%1==0):
						Nbins = max(valy)-min(valy)+2
						if Nbins>bins:Nbins=bins
						biny = linspace(min(valy),max(valy)+1,Nbins)
					C,X,Y = histogram2d(valx,valy,bins=(binx,biny))
					C=1.*C
					C = C/sum(C)
					colormap = pylab.cm.get_cmap(self.dp[ykey].get("contourcolor"),1024)
					if yN<xN:
						ax.pcolormesh(X,Y,C.T,cmap=colormap,vmin=min(C[C>0]),vmax=max(C))
						ax.xaxis.grid(True,ls="solid")
						ax.yaxis.grid(True,ls="solid")
					elif yN==xN:
						ax.text(0.5,0.5,self.dp[xkey].get("name"),horizontalalignment='center',verticalalignment="center",transform=ax.transAxes,backgroundcolor=self.dp[xkey].get("color"),color="white")
					else:
						pass
				except:
					pass
				colormap.set_under('white',0)
				ax.set_xlim(min(X),max(X))
				ax.set_ylim(min(Y),max(Y))
				if pC>=N**2-N:
					ax = pylabql.add_subplot(N+1,N,pC+N)
					ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10,prune='both'))
					for tick in ax.xaxis.get_ticklabels():
						tick.set_rotation('vertical')
						tick.set_size(6)
					ax.yaxis.set_ticklabels([])
					ax.plot(X[:-1],sum(C,axis=1),color=self.dp[xkey].get("color"),lw = self.dp[xkey].get("linewidth"),ls = "steps"+self.dp[xkey].get("linestyle"),marker=self.dp[xkey].get("marker"),markersize=self.dp[xkey].get("markersize"))
		if cb:
			axcb = pylab.axes([0.9,0.1,0.02,0.8])
			cb1 = matplotlib.colorbar.ColorbarBase(axcb,cmap=colormap,orientation="vertical")
			cb1.set_label("%s"%(cb))
	
	def hist1d(self,prod,binx=50,norm=False,smask=False,weights=False,Master=True,xlabel=False,ylabel=False,legend=True,legendbox=True):
		"""This routine makes a 1d histogram of the current data
		input: prod: key of self.data to be histogrammed
		binx: bins for the histogram (number or ndarray)
		norm: 'ymax' (normalized to max of hist) or 'ysum' (normalized to sum of hist)
		smask: list of masks that should be plotted
		weights: weights for the histogram (e.g. 'counts')
		Master: If True, the Master mask is plotted
		"""
		pylabh = pylab.figure()
		pdata = plot_data("hist1D",self)
		ax = pylab.gca()
		if not xlabel:
			ax.set_xlabel(self.dp[prod].get("name"))
		else:
			ax.set_xlabel(xlabel)
		if not ylabel:
			ax.set_ylabel('N')
		else:
			ax.set_ylabel(ylabel)
		if smask:
			validmasks = smask
		else:
			validmasks = self.mask.keys()
			validmasks.remove("Master")
			if Master!=False:
				validmasks.append("Master")
		valy = self.data[prod][self.DM()]
		
		binx=self.__create_bins(binx,prod,valy)
		pylab.xlim(binx[0],binx[-1])
		for mask in validmasks:
			valy = self.data[prod][self.get_mask(mask)]
			if weights:
				hist, xvals = histogram(valy,bins=binx,weights=self.data[weights][self.get_mask(mask)])
			else:
				hist, xvals = histogram(valy,bins=binx)
			pdata.add_data(mask,(hist,xvals),prodx=prod,norm=norm)
			hist=1.*hist
			if norm=='ymax':
				hist/=float(max(hist))
			elif norm=="ysum":
				hist/=float(sum(hist))
			elif norm:
				print("Unrecognized norm mode!")
			hist=append(hist,0.)
			ax.plot(xvals[:], hist,label=self.mp[mask].get("name"),color=self.mp[mask].get("color"),lw = self.mp[mask].get("linewidth"),ls = "steps-post"+self.mp[mask].get("linestyle"),marker=self.mp[mask].get("marker"))
		if legend and legendbox:
			pylab.legend()
		elif legend:
			pylab.legend().get_frame().set_visible(False)
		pmod = plot_mod(pylabh,style="hist1D")
		pmod.set_data(pdata)
		return pmod
		
	def animate1d(self,prod,prodz,binx=50,binz=50,speed=100,norm=False,smask=False,weights=False,Master=True,xlabel=False,ylabel=False, zlabel=False, legend=True,legendbox=True,save='',fps=5):
		"""This routine makes an animation of 1d histograms of the current data
		input: prod: key of self.data to be histogrammed
		binx: bins for the histogram (number or ndarray)
		binz: zbins for the aimation
		prodz: key of self.data to be histogrammed used for the animation frames
		speed: Animation speed (small = Fast)
		norm: 'ymax' (normalized to max of hist) or 'ysum' (normalized to sum of hist)
		smask: list of masks that should be plotted
		weights: weights for the histogram (e.g. 'counts')
		Master: If True, the Master mask is plotted
		"""
		pylaba = pylab.figure()
		ax = pylab.gca()
		ax_anim = pylab.axes([0.05,0.94,0.9,0.02])
		self.__init_anim_axes(ax_anim,zlabel,prodz)
		if not xlabel:
			ax.set_xlabel(self.dp[prod].get("name"))
		else:
			ax.set_xlabel(xlabel)
		if not ylabel:
			ax.set_ylabel('N')
		else:
			ax.set_ylabel(ylabel)
		if smask:
			validmasks = smask
		else:
			validmasks = self.mask.keys()
			validmasks.remove("Master")
			if Master!=False:
				validmasks.append("Master")
		valx = self.data[prod][self.DM()]
		valz = self.data[prodz][self.DM()]
		binx=self.__create_bins(binx,prod,valx)
		binz=self.__create_bins(binz,prodz,valz)
		pylab.xlim(binx[0],binx[-1])
		DATACONT = {}
		DATA = {}
		for mask in validmasks:
			valx = self.data[prod][self.get_mask(mask)]
			valz = self.data[prodz][self.get_mask(mask)]
			if weights:
				C,X,Z = histogram2d(valx,valz,bins=[binx,binz],weights=self.data[weights][self.get_mask(mask)])
			else:
				C,X,Z = histogram2d(valx,valz,bins=[binx,binz])
			C=1.*C
			if norm:
				C=self.__process_norm(C,norm)
			DATA[mask]=C
			DATACONT[mask]=ax.plot(binx, append(C[:,0],0),label=self.mp[mask].get("name"),color=self.mp[mask].get("color"),lw = self.mp[mask].get("linewidth"),ls = "steps-post"+self.mp[mask].get("linestyle"),marker=self.mp[mask].get("marker"))[0]
		ax_anim.set_xlim(min(Z),max(Z))
		ax_anim.set_ylim(0,1)
		if legend and legendbox:
			ax.legend()
		elif legend:
			ax.legend().get_frame().set_visible(False)
		ANIMCONT=ax_anim.plot([min(Z)],[0.5],linestyle="None",marker="o",markersize=6)
		
		# Here the Animation logic starts
		self.ANIMATION_C = -1
		ZW = (Z[1]-Z[0])/2.
		self.Splay = True
		_enter,_enterB=self.__init_anim_control()
		def animate(i):
			if self.Splay:
				self.ANIMATION_C+=1
			self.ANIMATION_C=self.ANIMATION_C%(len(Z)-1)
			ANIMCONT[0].set_data([Z[self.ANIMATION_C]],[0.5])
			for key in validmasks:
				z = append(DATA[key][:,self.ANIMATION_C],0)
				DATACONT[key].set_data(binx,z)
			return 0,
		self.C2=pylaba.canvas.mpl_connect('key_press_event',lambda event: _enter(event))
		self.C2=pylaba.canvas.mpl_connect('motion_notify_event',lambda event: _enterB(event,Z))
		print("SMALL HOWTO: ANIMATED1D")
		print("------------------------")
		print("Use NUMPAD arrowkeys to control animation.")
		print("Hover Mouse over animation bar, to jump to that location!")
		anim = matplotlib.animation.FuncAnimation(pylaba,animate,frames=Z.shape[0]-1,interval=speed,blit=False)
		if save:
			anim.save(save,fps=fps,codec="mpeg4")
		return anim
	def load_data(self):
		"""
		Must be implemented by USER
		"""
		return True
		
	def add_subset(self, file_U, prodx=False, prody=False, prodz=False, norm=False):
		'''
		This method extends the current date set with a previously saved subset. The keys of the current data set and those of the subset must be the same
		File_U: path were the subset is saved
		'''
		o = open(file_U)
		subset = cPickle.load(o)
		o.close()
		old_keys, new_keys = self.data.keys(), subset.keys()
		keys_fit = numpy.intersect1d(old_keys, new_keys)
		if len(keys_fit) != len(old_keys):
			print('keys do not fit, cannot add this subset to current data, shame on you')
		elif len(keys_fit) == len(old_keys):
			for key in old_keys:
				self.data[key] = numpy.append(self.data[key], subset[key])
			for mask in self.mask:
				self.mask[mask].update_mask()
			print('done, subset added')
		
		
from numpy import append
import scipy.cluster.vq

def generatekmeansmasks(data,k,name="Default",prodlist=[]):
	"""
	For a given data set and expected number of clusters k generatekmeansmasks returns a dictionary of k masks for the products given in prodlist using the k-means algorithm. standard k means assumes that each cluster is a Gaussian
	The dictionary keys are constructed as "name k".
	"""
	traindata=[]
	traindata=numpy.array([data[i] for i in prodlist])
	centeroid, classmask=scipy.cluster.vq.kmeans2(traindata.transpose(),k)
	masks={}
	for i in range(0,k):
		masks[name+' '+str(i+1)]=zeros(data[data.keys()[0]].shape[0],dtype=bool)
	for j in range (1,len(data[data.keys()[0]])):
		masks[name+' '+str(classmask[j]+1)][j] = 1
	return masks

#For debugging and testi
if __name__=="__main__":
	
	from numpy.random import normal
	import pylab
	pylab.ion()
	# EXAMPLE 

	# Here a new class MyData is created that inherits from dbData
	class MyData(dbData):
		# A load_data method is defined that saves data products into the dictionary self.data
		def load_data(self):
			# This small code snippet creates 200000 samples of 2 2D Gaussians at position (3,2) and (14,8)
			self.data["Detector_A"] = append(normal(3,3,size=900000),normal(14,2,size=900000))
			self.data["Detector_B"] = append(normal(2,4,size=900000),normal(8,3,size=900000))
			# Each events gets a time stamp
			self.data["time"]=linspace(0,100,self.data["Detector_A"].shape[0])
	# Create an instance of your class
	d = MyData()
	# Adds 2 masks that separates the 2 2D Gaussians
	d.add_mask("Ion_1")
	d.set_mask("Ion_1","time",0,50)
	d.add_mask("Ion_2")
	d.set_mask("Ion_2","time",50,100)

	# Show the Masks (broken for "direct masks" as from generatekmeansmask because it currently abuses the master mask)
	d.show_mask()

	# Create a 2d Histogram of all masks
	d.hist2d('Detector_A',"Detector_B",binx=linspace(-20,20,101),biny=linspace(-20,20,101),style="contour",shade=False)
	d.hist2d('Detector_A',"Detector_B",binx=linspace(-20,20,101),biny=linspace(-20,20,101),style="contour",shade=True)
	#Create a 1d Histogram of Detector A for all masks (normalized to the sum of the data)
	d.hist1d("Detector_A",binx=100,norm="ysum")
	#Plot a time series of the Detector A of masks Ion 1 and Ion 2
	d.timeseries("Detector_A",time=linspace(0,100,1001),mode="mean",smask=["Ion_1","Ion_2"])

	"""
	# example for applied masks:
	d.reset_mask("Ion_1")
	d.set_mask("Ion_1","time", direct=True)
	d.reset_mask("Ion_2")
	d.set_mask("Ion_2","time", direct=True)
	d.timeseries("Detector_A",time=linspace(0,100,1001),mode="mean",smask=["Ion_1","Ion_2"])
	"""
