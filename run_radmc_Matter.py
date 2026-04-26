# Import the radmc3dPy module
import radmc3dPy
import os
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits 
from scipy import interpolate
from radmc3dPy import *
import shutil
import datetime
import logging
import time
Name="Matter2016_AB_Aur"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("sim", exist_ok=True)

# Dann wird der run_dir-Ordner innerhalb von "sim" erstellt
run_dir = os.path.join("sim", f"run_{timestamp}_{Name}")
os.makedirs(run_dir, exist_ok=True)
#run_dir = f"run_{timestamp}_{Name}"
#os.makedirs(run_dir, exist_ok=True)

#Logging
logging.basicConfig(filename=os.path.join(run_dir, "log.txt"),
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# some constants
au           = 149.6e+11              #astronomical unit in cm
mas2rad      = np.pi/180/3600./1000. # mas to rad conversion
spec1 = None
spec2 = None
spec3 = None
Set_multiple=False
Make_images=False
logging.info(f"Making multiple SEDs = {Set_multiple}")
logging.info(f"Making Images = {Make_images}")

# List of the available dust opacities for our model
dustkappas=[
    "['Astrosilcarbgra']", 
    "['Astrosilcarb0012']", 
    "['Astrosilcarb001']",
    "['Astrosilcarboptool']",
    "['Draine84-Astrosil-01micron']",
    "['Draine84Astrosil095-carbon005-01micron']"]         

#Stellar Parameters
istar_sphere=0
tstar   =  10000       # Effective temp
rstar   = '2.0*rs'       # stellar radius in To
mstar   = '2.5*ms'     # stellar mass   in Mo 
# Disk Parameters
mdisk = '0.01*ms'       # Total mass in the disk. Dust mass is 1% of this mass
rin = '0.5*au'          # Disk inner radius in au
r_in=0.5*au
r1 = 3*au
rdisk = '200.0*au'      # Disk outer radius in au 
hrdisk =  0.09        # Pressure-scale-height over radius ratio at 100.0au
plsig1 = -1  #1,5           # Power exponent of the surface density law
plh =  0.25#0,25            # Disk flaring index
hpr_prim_rout = 0#0.025    # Pressure scale height at rin
prim_rout=0#3 #Outer boundary of puffed um inner rim
srim_rout = 0#3 #Outer boundary if inner rims smoothing
srim_plsig = 0 #power exponent of density reduction in the inner rim
hrpivot='100.0*au'
sigma_type=0
gsmax=0.1
gsmin=0.1
mixabun=[0.95, 0.05]
# Dust parameters
dustkappa = dustkappas[0]

# Grid/Other Parameters
pc = 144                   # Distance in parsec 
incl = 45                  # Inclination angle in degrees (for SED & Images)
xbound=['0.02*au', '2*au','220*au']
nx     = [200,200]          
wbound = [0.1, 0.9, 2, 7.0, 25., 1e4]   
nw     = [60, 30, 20, 30,30]    
ny     = [100,100,150,150]
# Computational Parameters
nphot  = '1e+8'  
nphot_scat = '1e+5'
nphot_spec = '1e+5'
threads = 32
modified_random_walk = 1
scattering_mode_max =1
tgas_eq_tdust=0

#Create parameter file
analyze.writeDefaultParfile('ppdisk1')

def run_radmc3d_simulation():
    global spec1, spec2, spec3, Set_Multiple, Make_images, star, g
    # Read initial density data
    #data = analyze.readData(ddens=True)
    
    # Setup the problem
    setup.problemSetupDust('ppdisk1',r_in=r_in, r1=r1, h_in=1, rstar=rstar,mstar=mstar,tstar=tstar, mdisk=mdisk,rin=rin,rdisk=rdisk,hrdisk=hrdisk, dustkappa_ext=dustkappa,plsig1=plsig1,plh=plh, nx=nx,nphot=nphot,xbound=xbound,wbound= wbound,nw= nw, binary=False, modified_random_walk=modified_random_walk, nphot_scat=nphot_scat, nphot_spec=nphot_spec, hpr_prim_rout=hpr_prim_rout, prim_rout=prim_rout, srim_rout=srim_rout, srim_plsig=srim_plsig, hrpivot=hrpivot, sigma_type=sigma_type, gsmax=gsmax, mixabun=mixabun, gsmin=gsmin, scattering_mode_max=scattering_mode_max, istar_sphere=istar_sphere, tgas_eq_tdust=tgas_eq_tdust, ny=ny)
    
    logging.info("Starting Radmc-3D")
    logging.info(f"Using incl={incl} and pc={pc}")
    logging.info("Parameters used:")
    
    # Modify radmc3d.inp file
    with open("radmc3d.inp", "a") as f:
        f.write("mc_scat_maxtauabs         = 30\n")  # Photon destroyed when tau=30
        f.close()
    logging.info(f"Setting maxtauabs=15")
    # Read parameters
    #par = analyze.readParams()
    
    # Run the RADMC-3D thermal Monte Carlo simulation
    os.system(f'radmc3d mctherm setthreads {threads} sloppy')
    logging.info(f"Starting Monte Carlo Simulation with {threads} threads sloppy")
    os.system(f'radmc3d sed incl {incl} setthreads {threads} sloppy')
    
    if Make_images is True:
        os.system(f'radmc3d image phi 0 lambda 10 incl 0 setthreads {threads} npix 500')
        shutil.copy("image.out", os.path.join(run_dir, "image_incl0.out"))
        os.system(f'radmc3d image phi 0 lambda 10 incl 90 setthreads {threads} npix 500')
        shutil.copy("image.out", os.path.join(run_dir, "image_incl90.out"))
        os.system(f'radmc3d image phi 0 lambda 10 incl 65 setthreads {threads} npix 500')
        shutil.copy("image.out", os.path.join(run_dir, "image_incl65.out"))
        

    if spec1 is None:  # Wenn spec1 noch nicht definiert ist
        spec1 = analyze.readSpectrum(fname='spectrum.out')
        shutil.copy("spectrum.out", os.path.join(run_dir, f"spectrum1_{timestamp}.out"))
    elif spec2 is None and Set_multiple is True:
        spec2 = analyze.readSpectrum(fname='spectrum.out')
        shutil.copy("spectrum.out", os.path.join(run_dir, f"spectrum2_{timestamp}.out"))
    elif spec3 is None and Set_multiple is True:
        spec3 = analyze.readSpectrum(fname='spectrum.out')
        shutil.copy("spectrum.out", os.path.join(run_dir, f"spectrum3_{timestamp}.out"))
    
    star = analyze.readStars()
    g = analyze.readGrid()
    
    return spec1, spec2, spec3, star, g

#Parameter for sim

#Run the simulation
start_time = time.time()
spec1, spec2, spec3, star, g = run_radmc3d_simulation()
with open("problem_params.inp", "r") as f:
        params = f.read()
        logging.info("\n" + params)
#Save input
shutil.copy("problem_params.inp", os.path.join(run_dir, f"problem_params_{timestamp}.inp"))
end_time = time.time()
runtime=(end_time-start_time)/60

if Set_multiple is True:
    #Parameter for sim
    hrdisk =  0.1
    plh =  0.25
    hrpivot='200.0*au'
    prim_rout=2.0
    srim_rout=3
    #Run the simulation
    spec1, spec2, spec3, star, g = run_radmc3d_simulation()

    #Parameter for sim
    hrdisk =  0.1
    plh =  0.26
    hrpivot='200.0*au'
    prim_rout=2.0
    srim_rout=3
    #Run the simulation
    spec1, spec2, spec3, star, g = run_radmc3d_simulation()

flux1=spec1[:, 1] / ((pc)**2)
if Set_multiple is True:
    flux2=spec2[:, 1] / ((pc)**2)
    flux3=spec3[:, 1] / ((pc)**2)

#Saving Input
shutil.copy("radmc3d.inp", os.path.join(run_dir, f"radmc3d_{timestamp}.inp"))

#Plotting the SED
plt.figure()
plb.title(r'SED of AB Aur')
plt.xscale('log')
plt.yscale('log')
data = np.loadtxt("ABAurDullemond.txt")
x = data[:, 0]
y = data[:, 1]*1.55*0.1
analyze.plotSpectrum(a=spec1, nufnu=True, micron=True, xlg=True, ylg=True, dpc=144.0, oplot=True, label=r'Modified H')

if Set_multiple is True:
    analyze.plotSpectrum(a=spec2, nufnu=True, micron=True, xlg=True, ylg=True, dpc=144.0, oplot=True, label=r'plh =  0.19')
    analyze.plotSpectrum(a=spec3, nufnu=True, micron=True, xlg=True, ylg=True, dpc=144.0, oplot=True, label=r'plh =  0.25')
star = analyze.readStars()
g = analyze.readGrid()
flux=star.fnustar/((pc)**2)
flux=np.reshape(flux,len(spec1))
plt.plot(g.wav, g.freq*flux, label='Stellar contribution')
plt.xlabel(r'$\lambda$ [$\mu$m]')
plt.ylabel(r'log $\nu F_{\nu}$ [erg.s$^{-1}$.cm$^{-2}$]')
plt.plot(x,y, label='AbAur-Dullemond')
plt.xlim(0.1,3000)
plt.ylim(1e-15,10**-6)
plt.legend(loc='lower left')
plt.savefig(f"SED_{Name}_{timestamp}.png", dpi=250, bbox_inches='tight')
shutil.move(f"SED_{Name}_{timestamp}.png", os.path.join(run_dir, f"SED_{Name}_{timestamp}.png"))
###
shutil.copy("run_radmc_Matter.py", os.path.join(run_dir, f"run_radmc_Dominik_{timestamp}.py"))
logging.info(f"Runtime: {runtime:.2f} minutes while using {threads} threads")

#Read Data
data = analyze.readData(dtemp=True, ddens=True)
opac = analyze.readOpac(ext=['Astrosilcarbgra'])
data.getTau(wav=0.55) 

#Plotting dust density
plb.figure()
plb.title(r'Dust density contours with $\tau=1$')
c1 = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, np.log10(data.rhodust[:,:,0,0].T), 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c1)
cb.ax.yaxis.labelpad = 20
cb.set_label(r'$\log_{10}{\rho}$', rotation=270.)
c2 = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.taux[:,:,0].T, [1.0],  colors='w', linestyles='solid')
#plb.clabel(c2, inline=1, fontsize=10, fmt='%g')
plb.savefig(f"dust_density_contours_{Name}_{timestamp}.png", dpi=300)
shutil.move(f"dust_density_contours_{Name}_{timestamp}.png", os.path.join(run_dir, f"dust_density_contours_{Name}_{timestamp}.png"))
    
#Plotting dust temp
plb.figure()
plb.title(r'Dust temperature contours')
c3 = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c3)
cb.set_label('T [K]', rotation=270.)
c4 = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.dusttemp[:,:,0,0].T, 10,  colors='k', linestyles='solid')
plb.clabel(c4, inline=1, fontsize=10)
cb.ax.yaxis.labelpad = 20
plb.savefig(f"dust_temperature_contours_{Name}_{timestamp}.png", dpi=300)    
shutil.move(f"dust_temperature_contours_{Name}_{timestamp}.png", os.path.join(run_dir, f"dust_temperature_contours_{Name}_{timestamp}.png"))
    
#Temp Dens Contours    
plb.figure()
plb.title(r'Temperature and density structure')
plt.xlim(0.5,200)
plt.ylim(0,0.4)
data_dens = analyze.readData(ddens=True)
c5 = plb.contourf(data_dens.grid.x/natconst.au, np.pi/2 - data_dens.grid.y, np.log10(data_dens.rhodust[:,:,0,0].T), 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
plb.colorbar(c5, label=r'$\log_{10}(\tau)$')  # oder passe die Beschriftung an
#c6 = plb.contourf(data.grid.x/natconst.au, np.pi/2 - data.grid.y, data.dusttemp[:,:,0,0].T, 30, alpha=0.2, cmap='gray')
# Optional: Colorbar für den Temperaturplot
# Zusätzlich Konturlinien für die Temperatur
c_lines = plb.contour(data.grid.x/natconst.au, np.pi/2 - data.grid.y, data.dusttemp[:,:,0,0].T, 30, linestyles='solid')
plb.clabel(c_lines, inline=1, fontsize=10)  
plb.savefig(f"temperature_density_contours_{Name}_{timestamp}.png", dpi=300)    
shutil.move(f"temperature_density_contours_{Name}_{timestamp}.png", os.path.join(run_dir, f"temperature_density_contours_{Name}_{timestamp}.png"))

#Density with tau=1
plb.figure()
plb.title(r'Dust density contours with $\tau=1$')
c7 = plb.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y, np.log10(data.rhodust[:,:,0,0].T), 30)
plb.xlabel('r [AU]')
plb.ylabel(r'$\pi/2-\theta$')
plb.xscale('log')
cb = plb.colorbar(c7)
cb.ax.yaxis.labelpad = 20
cb.set_label(r'$\log_{10}{\rho}$', rotation=270.)
c8 = plb.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y, data.taux[:,:,0].T, [1.0],  colors='w', linestyles='solid')
#plb.clabel(c8, inline=1, fontsize=10, fmt='%g')
plt.ylim(0,0.5)
plt.xlim(0.5,)
plb.savefig(f"density_zoom_{Name}_{timestamp}.png", dpi=300)  
shutil.move(f"density_zoom_{Name}_{timestamp}.png", os.path.join(run_dir, f"density_zoom_{Name}_{timestamp}.png"))
    
    
    