using PyCall
using Distributions
using Gen


@pyimport numpy

print("heyy")


py"""
import numpy as np
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
"""

# np = pyimport("numpy")
# # print(np.linspace(2000,2020,1000))
# t=np.linspace(2000,2020,10)
# head=(t.-t[1])*(-1)+5*np.cos(np.pi*2*t)
# print("np")
# print(t)
# print(head)



@gen function geo_model(head, reference_time, observed_deformation=[-0.0, 0.06656669398346529, 0.15390143625382938, 0.1676900960863535, 0.13702130338593635, 0.1673178782642764, 0.26523515379600787, 0.3275075677243084, 0.31217034659917625, 0.3065887878443767])
    Kv ~ cauchy(-5, 3)# m/yr
    # Sskv ~ cauchy(-3.5, 3) #m-1
    # Sske ~ cauchy(-5, 3) #m-1
    # nclay ~ uniform(5, .., 10)
    # claythick=5 # m
    # Kv ~ normal(0, 1)
    Sskv ~ normal(0, 1)
    Sske ~ normal(0, 1)
    nclay ~ normal(0, 1)
    claythick=5 # m

    # run simulation
    t,defm,head,defm_v=py"calc_deformation"(reference_time,head,10^Kv,10^Sskv,10^Sske,claythick,nclay)
    aligned_deformation=numpy.interp(reference_time,t,defm)
    print("aligned")
    print(aligned_deformation)

    # The following loop represents this method:
    # observe(aligned_deformation, Normal(observed_deformation, 2))
    # Since deformation is an array, we have to iterate over all variables.
    # @trace(aligned_deformation, :a)
    # for (index, value) in enumerate(aligned_deformation)
    #     @trace(aligned_deformation[index],"deformation_$index")
    # end
end 