import numpy as np
import matplotlib.pyplot as plt
import sys

def calc_deformation(time,head,Kv,Sskv,Sske,claythick,nclay,sandthick=-1,Nt_const=70): # this calculates deformation for a single clay layer of user=defined thickness
    # Use whatever units for time and length as desired, but they need to stay consistent
    # time - a vector of same lenght as head with the times that head measurements are taken. Numeric (years or days, typically)
    # head - a vector of same length as time with head measurements.
    # Kv - vertical hydraulic conductivity
    # Sske - Skeletal specific storage (elastic)
    # Sskv - skeletalt specific storage (inelastic)
    # clay_thick - thickness of single clay layer modeled
    # nclay - number of clay layers
    # sandthick - total thickness of sand. If negative, assume that sand thickness = clay thickness
    # t - vector containing the time at each time step. Needs to be sampled very dense. A dt of around 0.1 days usually is sufficient
    # Nz - number of layers in z direction, within the single clay layer modeled. Higher Nz requires higher dt, and vice versa

    # outputs are interpolated time, total surface deformation, interpolated head, and inelastic surface deformation

    Sske_sand_mult=0.5 # assuem sand sske 
    Sske_sand=Sske*Sske_sand_mult
    if sandthick<0:
        sand_thick=nclay*claythick
    else:
        sand_thick=sandthick
    Ske_sand=Sske_sand*sand_thick
    
    D=Kv/Sske
    num_yrs=time[-1]-time[0]
    Nz=5 # discretization of clay layer. Higher discretization in z requires higher discretization in time (longer simulation)
    
    clay_thick=claythick/2 # simulate doubly draining clay
    z = np.linspace(0, clay_thick, Nz+2)    # mesh points in space
    dz = clay_thick/Nz
    dz_full=np.ones(np.shape(z))*dz;dz_full[0]=dz/2;dz_full[-1]=dz/2
    tau=np.square(claythick/2)*Sske/Kv

    if tau<.1: # assume instantaneous equilibration of clay if tau <.1
        D=Kv/Sskv
        Nt=int(num_yrs*24+D*Nt_const) # set optimum number of time steps based on diffusivity. Minimum number is 4 per year
        t=np.linspace(time[0],time[-1],int(Nt+1))

        boundary=np.interp(t,time,head)

        dt = t[1] - t[0]
        h = boundary[0]*np.ones((Nz+2,Nt+1))  
        h[0,:]=boundary
        
        precons_head=h[:,0].copy()
    
        deformation=np.zeros(np.shape(h))
        deformation_v=np.zeros(np.shape(h))
        
        for n in range(0, Nt):
            # Compute u at inner mesh points
            for i in range(1, Nz+1):
                dz1=dz/(int(i==1)+1)
                dz2=dz
                dz_all=np.mean([dz1,dz2])
                h_new = h[0,n+1]
                dh=(h_new-h[i,n])
                defm=dh*Sske*dz*2
                deformation_v[i,n+1]=deformation_v[i,n]
                if np.logical_and(h_new<precons_head[i],dh<0): # if head drops below preconsolidation head, then Ss changes to Sskv
                    h_new = h[i,n] + ((Kv/Sskv)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                    precons_head[i]=h_new
                    dh=(h_new-h[i,n])
                    defm=dh*Sskv*dz*2
                    deformation_v[i,n+1]=defm+np.min(deformation_v[i,0:(n+1)])
                h[i,n+1]=h_new
                deformation[i,n+1]=defm+deformation[i,n]
            h[-1,n+1]=h_new
    else:    
        Nt=int(num_yrs*24+D*Nt_const) # set optimum number of time steps based on diffusivity. Minimum number is 4 per year

        t=np.linspace(time[0],time[-1],int(Nt+1))

        boundary=np.interp(t,time,head)

        dt = t[1] - t[0]
        h = boundary[0]*np.ones((Nz+2,Nt+1))  
        h[0,:]=boundary
        
        precons_head=h[:,0].copy()
    
        deformation=np.zeros(np.shape(h))
        deformation_v=np.zeros(np.shape(h))
        
        for n in range(0, Nt):
            # Compute u at inner mesh points
            for i in range(1, Nz+1):
                dz1=dz/(int(i==1)+1)
                dz2=dz
                dz_all=np.mean([dz1,dz2])
                h_new = h[i,n] + ((Kv/Sske)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                dh=(h_new-h[i,n])
                defm=dh*Sske*dz*2
                deformation_v[i,n+1]=deformation_v[i,n]
                if np.logical_and(h_new<precons_head[i],dh<0): # if head drops below preconsolidation head, then Ss changes to Sskv
                    h_new = h[i,n] + ((Kv/Sskv)*dt/dz_all)*((h[i-1,n] - h[i,n])/dz1+( - h[i,n] + h[i+1,n])/dz2)
                    precons_head[i]=h_new
                    dh=(h_new-h[i,n])
                    defm=dh*Sskv*dz*2
                    deformation_v[i,n+1]=defm+np.min(deformation_v[i,0:(n+1)])
                h[i,n+1]=h_new
                deformation[i,n+1]=defm+deformation[i,n]
            h[-1,n+1]=h_new
    deformation=np.sum(deformation,axis=0)*nclay
    deformation_v=np.sum(deformation_v,axis=0)*nclay
    boundary0=boundary-boundary[0]
    deformation=deformation+boundary0*Ske_sand
    return(t,deformation,boundary,deformation_v)



from scipy.optimize import minimize


def inference(ref_time, head, obs, max_iter=10):
    it_per_n_clay_value = int(max_iter / 5)
    Kv_init, Sskv_init, Sske_init = -5.0, -3.0, -5.0
    best_rsme = None
    best_params = [Kv_init, Sskv_init, Sske_init]
    best_n_clay = 0
    for n_clay_init in range(5, 10):

        def calc_residual(params, claythick=5):
            time, defm, _, _ = calc_deformation(ref_time, head, 10 ** params[0], 10 ** params[1], 10 ** params[2],
                                                claythick, n_clay_init)
            mod_def = np.interp(ref_time, time, defm)
            # mod_def = np.random.normal(loc=mod_def, scale=2, size=mod_def.shape)
            residual = mod_def - obs
            rmse = np.sqrt(np.mean(np.square(residual)))
            return rmse

        initial_parameters = [Kv_init, Sskv_init, Sske_init]
        parameters=minimize(calc_residual,initial_parameters, options={'maxiter':20}) # setting the maximum iterations to a low number so it doesn't take forever
        rmse = calc_residual(parameters.x)

        if best_rsme is None or rmse < best_rsme:
            best_rsme = rmse
            best_params = parameters.x
            best_n_clay = n_clay_init
            print("Better parameters found: ", best_params, ", ", n_clay_init)
    return best_params, best_n_clay


if __name__ == "__main__":
    f = open(sys.argv[1], "r")
    line_1 = f.readline()
    line_2 = f.readline()
    line_3 = f.readline()
    it = sys.argv[2]
    ref_time = [float(x) for x in line_1.split()]
    head = [float(x) for x in line_2.split()]
    obs_def = [float(x) for x in line_3.split()]
    params, n_clay = inference(ref_time, head, obs_def, max_iter=10)
    time, defm, _, _ = calc_deformation(ref_time, head, 10 ** params[0], 10 ** params[1], 10 ** params[2],
                                        5, n_clay)
    # compare with synthetic 'true' data
    plt.figure();
    plt.plot(time, defm);
    plt.scatter(ref_time, obs_def, s=1, c='r')
    plt.legend(['estimated deformation', 'observed'])
    plt.show()

