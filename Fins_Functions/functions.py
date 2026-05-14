import sys
sys.path.insert(0, 'D:\VSCodeWorkspace\LBE_Project')
import numpy as np
import h5py as h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

import illustris_python as il
import Fins_Functions.constants as fc

from scipy import stats

# --- Physics functions ---
def Recentralise(P0 ,Coords, L):
    """This function centralises all coordinates on the object at P0 using a periodic boundary condition of a box of side length L"""

    New_Coords = Coords - P0     # New coordinates for all other objects
    
    BC_Check1 = np.where(New_Coords>L/2)
    BC_Check2 = np.where(New_Coords<-L/2)

    New_Coords[BC_Check1] = New_Coords[BC_Check1] - L
    New_Coords[BC_Check2] = New_Coords[BC_Check2] + L
        
    return np.array(New_Coords)

# Im not actually sure what this function is for
def Running_Average_1D(array1x,array2x,array1y,array2y, x0,xf):  
    """ A function which calculates 10 data points which can be plotted to make a running average"""

    Data_range = xf-x0
    binsize = Data_range/10.0

    bin_left_edge = x0

    Average_diff = np.zeros(10)

    for i in range(10):     # only needs in range 10 because 10 bins
        bin_right_edge = bin_left_edge + binsize

        w1 = np.where((array1x >= bin_left_edge) & (array1x < bin_right_edge)) #shift + alt + down
        w2 = np.where((array2x >= bin_left_edge) & (array2x < bin_right_edge)) 

        if (len(w1[0]) < 20) or (len(w2[0]) < 20):
            Average_diff[i] = np.nan
            bin_left_edge = bin_right_edge
            
        
        else:
            array1y_in_bin = array1y[w1]
            array2y_in_bin = array2y[w2]

            # averages of the y value in the bin
            array1y_average = np.median(array1y_in_bin) 
            array2y_average = np.median(array2y_in_bin)

            # i dont need a bin_centers list because i dont care for it, only the averages
            Average_diff[i] = array2y_average - array1y_average

            bin_left_edge = bin_right_edge
    

    isnan = np.where(np.isnan(Average_diff))
    Average_diff = np.delete(Average_diff, isnan)
    # delete the nan entries, they cant be used to find an average anyway

    return Average_diff

def Add_Zero_Scatter(Data,SourceRange: list,TargetRange : list):

    """This function replaces data within a specified range with values randomly drawn from a uniform distribution within a target range."""    
    
    Dummy = np.where((Data >= SourceRange[0]) & (Data < SourceRange[1]))    
    Dummy_vals = np.random.uniform(low=TargetRange[0], high=TargetRange[1], size=len(Dummy[0]))

    Data[Dummy] = Dummy_vals
    return Data

# This function is currently very slow running, so for now i will just use the standard function
def Variable_Binned_Statistic(x, y, x_minlim=None, x_maxlim=None, Bin_lowerlim = None, Bin_Upperlim = None, percentile=16):
    """
    Compute binned statistics for y values grouped by x intervals.

    Parameters:
    x: array-like, x-values for binning
    y: array-like, y-values to compute statistics on
    x_minlim: float, minimum x limit for binning (default: min(x))
    x_maxlim: float, maximum x limit for binning (default: max(x))
    Bin_lowerlim: float, lower limit for drawing a bin (default: 0.02 * max(x))
    Bin_Upperlim: float, upper limit for drawing a bin (default: 0.1 * max(x))
    percentile: float, percentile level for lower/upper percentile bounds (default: 16)

    Returns:
    dict with keys: 'bin_edges', 'mean', 'median', 'std', 'sum', 'percentile_lower', 'percentile_upper', 'bin_centers', 'x_minlim', 'x_maxlim'
    """

    Bin_lowerlim = Bin_lowerlim if Bin_lowerlim is not None else 0.02 * len(x)
    Bin_Upperlim = Bin_Upperlim if Bin_Upperlim is not None else 0.1 * len(x)

    if x_minlim is None:
        x_minlim = np.min(x)
    if x_maxlim is None:
        x_maxlim = np.max(x)



    def get_bins_from_histogram(Var_bin_histogram):
        numbins = np.size(Var_bin_histogram) - 1
        Var_bins = np.zeros(numbins)
        for i in range(0,numbins):
            Var_bins[i] = (Var_bin_histogram[i] + Var_bin_histogram[i+1])/2

        return(Var_bins)

    def get_bin_edges(x, Bin_lowerlim, Bin_Upperlim, x_minlim, x_maxlim):
        bin_edges = [x_minlim]
        binsize = 0.1 * (x_maxlim - x_minlim)
        i = 0

        while bin_edges[i] < x_maxlim:
            Bin_vals = np.where((x >= bin_edges[i]) & (x < bin_edges[i] + binsize))[0]
            
            if len(Bin_vals) < Bin_lowerlim:
                binsize = binsize * 1.1
            elif len(Bin_vals) > Bin_Upperlim:
                binsize = binsize * 0.9
            else:
                if bin_edges[i] + binsize > x_maxlim:
                    bin_edges.append(x_maxlim)
                else:
                    bin_edges.append(bin_edges[i] + binsize)
                i += 1

        return np.array(bin_edges)



    output = {}

    output['bin_edges'] = get_bin_edges(x, Bin_lowerlim, Bin_Upperlim, x_minlim, x_maxlim)


    output['mean'], _, _ = stats.binned_statistic(x, y, statistic=np.nanmean, bins=output['bin_edges'])
    output['median'], _, _ = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=output['bin_edges'])
    output['std'], _, _ = stats.binned_statistic(x, y, statistic=np.nanstd, bins=output['bin_edges'])
    output['sum'], _, _ = stats.binned_statistic(x, y, statistic=np.nansum, bins=output['bin_edges'])
    output['percentile_lower'], _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, percentile), bins=output['bin_edges'])
    output['percentile_upper'], _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 100-percentile), bins=output['bin_edges'])
    output['bin_centers'] = get_bins_from_histogram(output['bin_edges'])
    output['x_minlim'] = x_minlim
    output['x_maxlim'] = x_maxlim
    return output

# --- hdf5 functions ---
def hdf5_append(file_loc, new_data,  header: str, make = False, overide = False):
    """This function appends data to a hdf5 file. If make == True then
    create a new dataset in the hdf5 file with header."""
    
    import numpy as np
    import h5py

    new_data = np.array(new_data)

    with h5py.File(file_loc, "a") as hdf:
        if header not in hdf:
            if make == False:
                print(hdf.keys())
                raise KeyError(f"Field '{header}' not found as a header within file.")
            elif make == True:
                hdf.create_dataset(header, data=new_data, maxshape=(None,))
        if overide == True:
            del hdf[header]
            new_data = np.atleast_1d(new_data)
            hdf.create_dataset(header, data=new_data, maxshape=(None,))

        else:
            dset = hdf[header]
            dset.resize((dset.shape[0] + len(new_data),))
            dset[-len(new_data):] = np.atleast_1d(new_data)

def hdf5_del(file_loc, header: str):
    import h5py

    with h5py.File(file_loc, "a") as hdf:
        if header not in hdf:
            return
        else:
            del hdf[header]

def hdf5_open(file_loc, header, printheaders: bool = False):
    import h5py

    Results = {}
    with h5py.File(file_loc, "r") as hdf:

        keys = header if header is not None else hdf.keys()

        for key in keys:
            
            if key not in hdf:
                raise KeyError(f"Field '{key}' not found as a header within file.")
            
            Loaded = hdf[key]
            if isinstance(Loaded, h5py.Dataset):
                Results[key] = Loaded[()]
            
            elif isinstance(Loaded, h5py.Group):
                Results[key] = hdf5_open(Loaded, None, printheaders)

            if printheaders and key == 'Header':
                print("Header:")
            for attr_name, attr_value in Loaded.attrs.items():
                print(f"  {attr_name}: {attr_value}")

        return Results
 

# --- Plotting Functions ---
def format_axes(ax, title="", xlabel="", ylabel="", ylim=[], xlim=[]):
    """
    Format a Matplotlib axes with common options, used in my 2d histograms.
    """

    ax.set_title(title, fontsize = "24")
    ax.set_xlabel(xlabel, fontsize = "20")
    ax.set_ylabel(ylabel, fontsize = "20")
    
    ax.tick_params(axis='both', labelsize=20) 

    ax.spines['top'].set_linewidth(1.25) 
    ax.spines['right'].set_linewidth(1.25)
    ax.spines['bottom'].set_linewidth(1.25)
    ax.spines['left'].set_linewidth(1.25) 

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(xlim[0],xlim[1])
    #ax.set_yscale('log')

def format_axes_1d(ax, xlabel="", ylabel="", ylog = False, xlog = False, 
                   legendloc = None, n_cols = 1):
    """
    Format a Matplotlib axes with common options.

    Parameters:
        ax (matplotlib.axes._subplots.Axes): The axes to be formatted.
        xlabel (str, optional): Label for the x-axis. Default is "".
        ylabel (str, optional): Label for the y-axis. Default is "".
        ylog (bool, optional): Whether to display the y-axis in log10, default is False
        xlog (bool, optional): Whether to display the x-axis in log10, default is False
        legendloc (str, optional): Where to display the legend. Default is undisplayed.
        n_cols (int, optional): Number of columns desplayed in the legend.
        xlim (list, optional): x-axis limits, default is [min_val,max_val]
        ylim (list, optional): y-axis limits, default is [min_val,max_val]
    """
    ax.set_xlabel(xlabel, fontsize = "20")
    ax.set_ylabel(ylabel, fontsize = "20")
    
    ax.tick_params(axis='both', labelsize=20) 

    ax.spines['top'].set_linewidth(1.25) 
    ax.spines['right'].set_linewidth(1.25)
    ax.spines['bottom'].set_linewidth(1.25)
    ax.spines['left'].set_linewidth(1.25) 

    if legendloc is not None:
        ax.legend(loc = legendloc, ncol = n_cols ,frameon=False, fontsize = "15")

    if ylog == True:
        ax.set_yscale('log')
    if xlog == True:
        ax.set_xscale('log')

def matplot_1d(xdata, ydata, markers = True, binned_statistic = True, statistic_method = "median",
               xlabel = "", ylabel = "", ylog = False, xlog = False, legendloc = None,
                 xlim = None, ylim = None, savepath = None, savetitle = None, nbins = 10,
                 percentile = 68):
    """
    Format a Matplotlib axes with common options and optionally calculate a running average with percentiles.

    Parameters:
        xdata (array-like, float): x-axis data.
        ydata (array-like, float): y-axis data.
        markers (bool, optional): Whether to display markers. Default is True.
        binned_statistic (bool, optional): Whether to calculate and display a running average with shaded percentile regions. Default is True.
        statistic_method (str, optional): Which average should be used, options in Reza_binned_statistic. Default is "median"
        xlabel (str, optional): Label for the x-axis. Default is "".
        ylabel (str, optional): Label for the y-axis. Default is "".
        ylog (bool, optional): Whether to display the y-axis in log10. Default is False.
        xlog (bool, optional): Whether to display the x-axis in log10. Default is False.
        legendloc (str, optional): Where to display the legend. Default is undisplayed.
        n_cols (int, optional): Number of columns desplayed in the legend.
        xlim (list, optional): x-axis limits. Default is [min_xdata,max_xdata].
        ylim (list, optional): y-axis limits. Default is [min_ydata,max_ydata].
        savepath (str, optional): root path for where to save the figure.
        savetitle (str, optional): title for the jpg file.
        nbins (int, optional): number of bins for the running average. Default is 10.
        percentile (float, optional): percentile level for lower/upper percentile bounds. Default is 16
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig = plt.figure(figsize = fc.figsize)
    ax = plt.axes()
    ax.set_rasterized(True)

    if markers is True:
        ax.plot(xdata,ydata, "." , color = "black", markersize = 1, alpha = 0.5, rasterized=True)      
    
    if binned_statistic is True:
        Statistic = Reza_binned_statistic(xdata,ydata, nbins=nbins, percentile=percentile)

        ax.plot(Statistic["bin_centers"],Statistic[statistic_method], color = "red",linewidth = 2.5,linestyle = "-", alpha = 1)          
        ax.fill_between(Statistic["bin_centers"], Statistic["percentile_lower"] , Statistic["percentile_upper"], color = "red", alpha = 0.2)

    if xlim is None:
        xlim = [np.min(xdata),np.max(xdata)]
    if ylim is None:
        ylim = [np.min(ydata),np.max(ydata)]

    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    format_axes_1d(ax, xlabel, ylabel, ylog, xlog, legendloc)
    plt.tight_layout()


    if savepath and savetitle is not None:
        plt.savefig("{}/{}.pdf".format(savepath,savetitle))


# --- Reza Functions ---

def Reza_binned_statistic(x, y, nbins=10, x_minlim=None, x_maxlim=None, percentile=16, bins_edges=None):
  """
  Compute binned statistics for y values grouped by x intervals.
  
  Parameters:
    x: array-like, x-values for binning
    y: array-like, y-values to compute statistics on
    nbins: int, number of bins (default: 10)
    x_minlim: float, minimum x limit for binning (default: min(x))
    x_maxlim: float, maximum x limit for binning (default: max(x))
    percentile: float, percentile level for lower/upper percentile bounds (default: 16)
    bins_edges: array-like, custom bin edges (overrides nbins if provided)
  
  Returns:
    dict with keys: 'bin_edges', 'mean', 'median', 'std', 'sum', 'percentile_lower', 'percentile_upper', 'bin_centers', 'x_minlim', 'x_maxlim'
  """


  def get_bins_from_histogram(Var_bin_histogram):
    numbins = np.size(Var_bin_histogram) - 1
    Var_bins = np.zeros(numbins)
    for i in range(0,numbins):
      Var_bins[i] = (Var_bin_histogram[i] + Var_bin_histogram[i+1])/2
      
    return(Var_bins)
  
  if x_minlim is None:
    x_minlim = np.min(x)
  if x_maxlim is None:
    x_maxlim = np.max(x)
  output = {}
  if bins_edges is not None:
    output['bin_edges'] = bins_edges
  else:
    output['bin_edges'] = np.linspace(x_minlim, x_maxlim, nbins + 1)
  output['mean'], _, _ = stats.binned_statistic(x, y, statistic=np.nanmean, bins=output['bin_edges'])
  output['median'], _, _ = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=output['bin_edges'])
  output['std'], _, _ = stats.binned_statistic(x, y, statistic=np.nanstd, bins=output['bin_edges'])
  output['sum'], _, _ = stats.binned_statistic(x, y, statistic=np.nansum, bins=output['bin_edges'])
  output['percentile_lower'], _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, percentile), bins=output['bin_edges'])
  output['percentile_upper'], _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, 100-percentile), bins=output['bin_edges'])
  output['bin_centers'] = get_bins_from_histogram(output['bin_edges'])
  output['x_minlim'] = x_minlim
  output['x_maxlim'] = x_maxlim
  return output