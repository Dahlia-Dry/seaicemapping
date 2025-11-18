# -*- coding: utf-8 -*-
import numpy as np

def nasa(b18v, b18h, b37v, tiepts=None):
    """NASA-Team ice concetration algorithm
    """
    
    if tiepts == None:
        tiepts_old = [202.00, 182.00, 246.00, 0, 0, 0, 179.00, 217.64, 253.00,\
                  100.00, 170.00, 242.00]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]
     
    ow18v = tiepts[6]
    ow18h = tiepts[9]
    ow37v = tiepts[0]
    fy18v = tiepts[8]
    fy18h = tiepts[11]
    fy37v = tiepts[2]
    my18v = tiepts[7]
    my18h = tiepts[10]
    my37v = tiepts[1]
    #print "ow18v,ow18h,ow37v,fy18v,fy18h,fy37v,my18v,my18h,my37v"
    #print ow18v,ow18h,ow37v,fy18v,fy18h,fy37v,my18v,my18h,my37v

    a0 = - ow18v + ow18h
    a1 =   ow18v + ow18h
    a2 =   my18v - my18h - ow18v + ow18h
    a3 = - my18v - my18h + ow18v + ow18h
    a4 =   fy18v - fy18h - ow18v + ow18h
    a5 = - fy18v - fy18h + ow18v + ow18h

    b0 = - ow37v + ow18v
    b1 =   ow37v + ow18v
    b2 =   my37v - my18v - ow37v + ow18v
    b3 = - my37v - my18v + ow37v + ow18v
    b4 =   fy37v - fy18v - ow37v + ow18v
    b5 = - fy37v - fy18v + ow37v + ow18v

    gr = (b37v - b18v)/(b37v + b18v)
    pr = (b18v - b18h)/(b18v + b18h)

    d0 = (-a2*b4) + (a4*b2)
    d1 = (-a3*b4) + (a5*b2)
    d2 = (-a2*b5) + (a4*b3)
    d3 = (-a3*b5) + (a5*b3)

    dd = d0 + d1*pr + d2*gr + d3*pr*gr
    
    f0 = (a0*b2) - (a2*b0)
    f1 = (a1*b2) - (a3*b0)
    f2 = (a0*b3) - (a2*b1)
    f3 = (a1*b3) - (a3*b1)
    m0 = (-a0*b4) + (a4*b0)
    m1 = (-a1*b4) + (a5*b0)
    m2 = (-a0*b5) + (a4*b1)
    m3 = (-a1*b5) + (a5*b1)

    cf = (f0 + f1*pr + f2*gr + f3*pr*gr)/dd
    cm = (m0 + m1*pr + m2*gr + m3*pr*gr)/dd

    cf = cf
    cm = cm
    ct = cm + cf
    return ct

def bristol(tb18v, tb37v, tb37h, tiepts=None):
    """Bristol ice concentration algorithm
    """

    if tiepts == None:
        # Bootstrap winter tie points (Comiso, 1997)
        # (Ice tie points represent slopes and offsets only)
        tiepts_old = [202.00, 182.00, 246.00, 130.00, 170.00, 234.00, 179.00,\
                  217.64, 253.00, 0.0, 0.0, 0.0]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    tw18v = tiepts[6]
    tw37h = tiepts[3]
    tw37v = tiepts[0]
    tfy18v = tiepts[8]
    tfy37h = tiepts[5]
    tfy37v = tiepts[2]
    tmy18v = tiepts[7]
    tmy37h = tiepts[4]
    tmy37v = tiepts[1]
    
    xa = tmy37v + (1.045*tmy37h) + (0.525*tmy18v)
    xd = tfy37v + (1.045*tfy37h) + (0.525*tfy18v)
    xh = tw37v + (1.045*tw37h) + (0.525*tw18v)
    xt = tb37v +(1.045*tb37h) + (0.525*tb18v)

    ya = (0.9164*tmy18v) - tmy37v + (0.4965*tmy37h)
    yd = (0.9164*tfy18v) - tfy37v + (0.4965*tfy37h)
    yh = (0.9164*tw18v) - tw37v + (0.4965*tw37h)
    yt = (0.9164*tb18v)- tb37v + (0.4965*tb37h)

    a_ht = (yt - yh)/(xt - xh)
    b_ht = yh - (a_ht*xh)
    a_da = (ya - yd)/(xa - xd)
    b_da = yd - (a_da*xd)

    xi = (b_da - b_ht)/(a_ht - a_da)
    cf = (xt - xh)/(xi - xh)
    c = cf
    return c

def fcomiso(tb18v, tb37v, tiepts=None):
    """Comiso ice concentration algorithm
    """

    if tiepts == None:
        # bootstrap winter tie points (Comiso, 1997)
        # (ice tie points represent slopes and offsets only)
        tiepts_old = [202.00, 182.00, 246.00, 0, 0, 0, 179.00,\
                  217.64, 253.00, 0, 0, 0]

        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]
    tw18v = tiepts[6]
    tw37v = tiepts[0]
    tfy18v = tiepts[8]
    tfy37v = tiepts[2]
    tmy18v = tiepts[7]
    tmy37v = tiepts[1]
    
    af = (tfy37v - tmy37v)/(tfy18v - tmy18v)
    bf = (tmy37v - af*tmy18v)
    qf = (tb37v - tw37v)/(tb18v - tw18v)
    wf = (tw37v - qf*tw18v)
    ti18vf = (bf - wf)/(qf - af)
    cf = (tb18v - tw18v)/(ti18vf - tw18v)
    return cf
    
def pcomiso(tb37v, tb37h, tiepts=None):
    """Comiso ice concentration algorithm
    """
    if tiepts == None:
        # bootstrap winter tie points (Comiso, 1997)
        # (ice tie points represent slopes and offsets only)
        tiepts_old = [202.00, 182.00, 246.00, 130.00, 170.00, 234.00, 179.00,\
                  217.64, 253.00, 0.0, 0.0, 0.0]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    tw18v = tiepts[6]
    tw37h = tiepts[3]
    tw37v = tiepts[0]
    tfy18v = tiepts[8]
    tfy37h = tiepts[5]
    tfy37v = tiepts[2]
    tmy18v = tiepts[7]
    tmy37h = tiepts[4]
    tmy37v = tiepts[1]

    ap   = (tfy37v - tmy37v) / (tfy37h - tmy37h)
    bp   = (tmy37v - ap * tmy37h)
    qp   = (tb37v - tw37v) / (tb37h - tw37h)
    wp   = (tw37v - qp * tw37h)
    ti37hp = (bp - wp) / (qp - ap)
    ti37vp =  ap * ti37hp + bp
    cp = 0
    if (ti37vp - tw37v) != 0:
        cp = (tb37v - tw37v) / (ti37vp - tw37v)
    return cp

def tud(tb18v, tb37v, tb85v, tb85h, tiepts=None):
    """TUD ice concentration alogorithm
    """

    if tiepts == None:
        # bootstrap winter tie points (Comiso, 1997)
        # (ice tie points represent slopes and offsets only)
        tiepts_old = [202.00, 182.00, 246.00, 130.00, 170.00, 234.00, 179.00,\
                  217.64, 253.00, 0, 0, 0]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    a1 = 1.35
    a2 = -1.0/40.0
    a3 = -0.03

    tw18v = tiepts[6]
    tw37v = tiepts[0]
    tfy18v = tiepts[8]
    tfy37v = tiepts[2]
    tmy18v = tiepts[7]
    tmy37v = tiepts[1]

    af = (tfy37v - tmy37v)/(tfy18v - tmy18v)
    bf = (tmy37v - af*tmy18v)
    qf = (tb37v - tw37v)/(tb18v - tw18v)
    wf = (tw37v - qf*tw18v)
    ti18vf = (bf - wf)/(qf - af)
    cf = (tb18v - tw18v)/(ti18vf - tw18v)
    c=cf
    c85 = a1 + (tb85v - tb85h)*a2
    #if 100.0*cf > 10.0:
    if cf*c85 > 0: c = np.sqrt(cf*c85)+a3
    return c

def asi(tb85v, tb85h):
    """The asi ice concentration algorithm
    """
    P0 = 47.0
    P1 = 7.5
    P = tb85v - tb85h
    """method coefficients:"""
    d3=1.64/100000.0
    d2=-0.0016
    d1=0.0192
    d0_coeff=0.971
    """concentrations calculation:"""
    ct = d3 * P**3.0 + d2 * P**2.0 + d1 * P + d0_coeff
    return ct
    

def near90(tb85v, tb85h):
    """Svendsen et al. 1987"""
    Tw=272.0
    SAT=267.0
    m=1
    """ Emissivities from Andersen 1998"""
    ew85v=[0.84, 0.85, 0.86, 0.86, 0.87, 0.88, 0.89, 0.89, 0.88, 0.87, 0.86, 0.86]
    emy85v=[0.67, 0.69, 0.72, 0.75, 0.77, 0.80, 0.77, 0.74, 0.74, 0.73, 0.75, 0.70]
    efy85v=[0.84, 0.89, 0.84, 0.84, 0.86, 0.87, 0.85, 0.82, 0.84, 0.84, 0.88, 0.82]
    ew85h=[0.70, 0.69, 0.69, 0.70, 0.70, 0.73, 0.75, 0.76, 0.75, 0.72, 0.71, 0.69]
    emy85h=[0.65, 0.66, 0.66, 0.71, 0.74, 0.74, 0.75, 0.72, 0.72, 0.71, 0.72, 0.67]
    efy85h=[0.82, 0.84, 0.83, 0.82, 0.84, 0.83, 0.83, 0.77, 0.74, 0.76, 0.83, 0.78]

    """from NORSEX"""
    #T_ice = 0.4 * SAT + 0.6 * Tw

    #tw85v  = ew85v[m-1] *  Tw
    #tmy85v = emy85v[m-1] * T_ice
    #tfy85v = efy85v[m-1] * T_ice
    #tw85h  = ew85h[m-1] *  Tw
    #tmy85h = emy85h[m-1] * T_ice
    #tfy85h = efy85h[m-1] * T_ice

    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
               88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    tmy85v = tiepts[30]
    tfy85v = tiepts[31]
    tmy85h = tiepts[32]
    tfy85h = tiepts[33]
    tw85v  = tiepts[34]
    tw85h  = tiepts[35]

    P = tb85v - tb85h
    P0 = tw85v - tw85h
    P1 = tfy85v - tfy85h
    #P0 = 47.0 # as in asi
    #P1 = 19.0 #using an emissivity diff of 0.07 as in Matzler et al. 1984

    A=np.matrix([[P1**3.0, P1**2.0, P1, 1.0],\
                [P0**3.0, P0**2.0, P0, 1.0],\
                [3.0*P1**3.0, 2.0*P1**2.0, P1, 0.0],\
                [3.0*P0**3.0, 2.0*P0**2.0, P0, 0.0]])
    b=np.matrix([1.0, 0.0, -0.14, -1.14])

    d=A.I * b.T
    C = d[0] * P**3 + d[1] * P**2 + d[2] * P + d[3]
    return np.float(C)
    
def near90_linear(tb85v, tb85h):
    ct=1.22673-0.02652*(tb85v-tb85h)
    return ct

def UMass(tb19v,tb37v,tiepts=None):
#The code is based on C. T. Swift, L. S. Fedor, and R. O. Ramseier, ?An Algorithm 
#to Measure Sea Ice Concentration With Microwave Radiometers,? Journal of Geophysical 
#Research, vol. 90, no. C1, pages 1087 - 1099, 1985.
#Emissivities are taken from NORSEX SSM/I
    
    if tiepts == None:
        # bootstrap winter tie points (Comiso, 1997)
        # (ice tie points represent slopes and offsets only)
        tiepts = [202.00, 182.00, 246.00, 130.00, 170.00, 234.00, 179.00,\
                  217.64, 253.00, 0, 0, 0]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    tw19v = tiepts[6]
    tw37v = tiepts[0]
    tfy19v = tiepts[8]
    tfy37v = tiepts[2]
    tmy19v = tiepts[7]
    tmy37v = tiepts[1]

    """solution of the equations (11)-(12) in Swift et al 1985"""
    """here we use brightness temperatures instead!! (Rasmus 2012)"""
    """e19v=(TB19v-13)./(Ts-12);
       e37v=(TB37v-26)./(Ts-26);"""

    a1 = (tfy19v - tb19v) / (tfy19v - tw19v)
    a2 = (tfy19v - tmy19v) / (tfy19v - tw19v)
    a3 = (tfy37v - tb37v) / (tfy37v - tmy37v)
    a4 = (tfy37v - tw37v) / (tfy37v - tmy37v)
    fw = (a1 - a2 * a3) / (1.0 - a2 * a4)

    Cmy = a3 - fw * a4
    Cfy = 1.0 - fw - Cmy
    CT = Cmy + Cfy
    return CT

def norsex(TB19v,TB37v,tiepts=None):
    #index for satellite smmr and ssmi 0/1
    idx = 1
    #some coefficients for SMMR and SSMI
    SAT = 260.0
    T_sa = [270.0, 270.0]
    T_a = [250.0, 250.0]
    To = [272.0, 272.0]
    ew19v = [0.6500, 0.6210]
    ew37v = [0.7500, 0.7120]
    efy19v = [0.9700, 0.9700]
    efy37v = [0.9700, 0.9700]
    emy19v = [0.8200, 0.8200]
    emy37v = [0.7400, 0.7400]
    tau_sa19v = [0.0610, 0.0610]
    tau_sa37v = [0.1000, 0.1000]
    tau_a19v = [0.0440, 0.0440]
    tau_a37v = [0.0700, 0.0700]
    
    if tiepts == None:
        # bootstrap winter tie points (Comiso, 1997)
        # (ice tie points represent slopes and offsets only)
        tiepts_old = [202.00, 182.00, 246.00, 130.00, 170.00, 234.00, 179.00,\
                  217.64, 253.00, 0, 0, 0]
        tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
                  219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
                  251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
                   88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
                  192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    TB_w_19v = tiepts[6]
    TB_w_37v = tiepts[0]
    TB_fy_19v = tiepts[8]
    TB_fy_37v = tiepts[2]
    TB_my_19v = tiepts[7]
    TB_my_37v = tiepts[1]

    #the sea ice temperature
    T_ice = 0.4*SAT + 0.6*To[1]

    #Constants to be used in computing ice concentrations:
    a11 = TB_fy_19v - TB_w_19v
    a21 = TB_fy_37v - TB_w_37v
    a12 = TB_my_19v - TB_w_19v
    a22 = TB_my_37v - TB_w_37v
    d_coef = a11 * a22 - a12 * a21

    #Initialize atmospheric surface temperature: here it is 260K
    t_atm_surf=SAT

    #interpolate opacity between arctic and subarctic values:
    tau19v = tau_a19v[idx] + (t_atm_surf - T_a[idx]) * (tau_sa19v[idx] - tau_a19v[idx]) / (T_sa[idx] - T_a[idx])
    tau37v = tau_a37v[idx] + (t_atm_surf - T_a[idx]) * (tau_sa37v[idx] - tau_a37v[idx]) / (T_sa[idx] - T_a[idx])
    
    #find emitted brightness temperature at the surface by correcting for
    #atmospheric disturbances:
    TB_surf_19v = (TB19v - t_atm_surf * (2.0 * tau19v - tau19v**2.0 + 0.01)) / (1.0 - 2.0 * tau19v + tau19v**2.0 - 0.01)
    TB_surf_37v = (TB37v - t_atm_surf * (2.0 * tau37v - tau37v**2.0 + 0.01)) / (1.0 - 2.0 * tau37v + tau37v**2.0 - 0.01)

    #Find new atmospheric surface brightness temperature and mean surface
    #emissions by solving for first year and multi-year ice concentrations.
    c1 = TB_surf_19v - TB_w_19v
    c2 = TB_surf_37v - TB_w_37v
    Cmy = (a11 * c2 - a21 * c1) / d_coef
    Cfy = (a22 * c1 - a12 * c2) / d_coef
    CT = Cfy + Cmy
    
    t_atm_surf = To[idx] + (SAT - To[idx]) * CT
    
    #interpolate opacity between arctic and subarctic values:
    tau19v = tau_a19v[idx] + (t_atm_surf - T_a[idx]) * (tau_sa19v[idx] - tau_a19v[idx]) / (T_sa[idx] - T_a[idx])
    tau37v = tau_a37v[idx] + (t_atm_surf - T_a[idx]) * (tau_sa37v[idx] - tau_a37v[idx]) / (T_sa[idx] - T_a[idx])
    
    #find emitted brightness temperature at the surface by correcting for
    #atmospheric disturbances:
    TB_surf_19v = (TB19v - t_atm_surf * (2.0 * tau19v - tau19v**2.0 + 0.01)) / (1.0 - 2.0 * tau19v + tau19v**2.0 - 0.01)
    TB_surf_37v = (TB37v - t_atm_surf * (2.0 * tau37v - tau37v**2.0 + 0.01)) / (1.0 - 2.0 * tau37v + tau37v**2.0 - 0.01)

    #Find new atmospheric surface brightness temperature and mean surface
    #emissions by solving for first year and multi-year ice concentrations.
    c1 = TB_surf_19v - TB_w_19v
    c2 = TB_surf_37v - TB_w_37v
    Cmy = (a11 * c2 - a21 * c1) / d_coef
    Cfy = (a22 * c1 - a12 * c2) / d_coef
    CT = Cfy + Cmy
    
    return CT
    
def calval(tb37v,tb37h):
    #the calval/NRL alg. described in Lo, 1983
    #A0=1.3, A1=0.019, in Natalias code and she is calling it NRL
    A0 = 1.2
    A1 = -0.025
    C = A0 - A1 * (tb37v - tb37h)
    return C

def onechannel(tb6h):
    """Simple 1 channel algorithm
    """
    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    fy6h = tiepts[17]
    my6h = tiepts[16]

    ow6h = 82.3
    i6h = (fy6h+my6h)/2

    ct = (tb6h - ow6h)/(i6h - ow6h)
    return ct

def P90(tb85v, tb85h):
    X=(tb85v-tb85h)
    P=(X-2.63)/0.752

    d3=1.64/100000.0
    d2=-0.0016
    d1=0.0192
    d0=0.971

    c1 = d3 * P**3.0 + d2 * P**2.0 + d1 * P + d0

    c = c1+(P-8)/700 #to adjust near SIC0
    if (P>48):
        c=-0.026 #to prevent large P85 giving ice
    if (P<8.5):
        c=1.03 #to prevent low P85 losing ice

    return c


def pr(tb18v, tb18h, tb37v, tb37h):
    """Simple polarization ratio algorithm
    """

    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]
    ow18v = tiepts[6]
    ow18h = tiepts[9]
    ow37v = tiepts[0]
    ow37h = tiepts[3]
    fy18v = tiepts[8]
    fy18h = tiepts[11]
    fy37v = tiepts[2]
    fy37h = tiepts[5]
    my18v = tiepts[7]
    my18h = tiepts[10]
    my37v = tiepts[1]
    my37h = tiepts[4]

    i18v = (fy18v+my18v)/2
    i18h = (fy18h+my18h)/2
    i37v = (fy37v+my37v)/2
    i37h = (fy37h+my37h)/2

    PR18 = (tb18v - tb18h)/(tb18v + tb18h)
    PR37 = (tb37v - tb37h)/(tb37v + tb37h)

    c18 =  (ow18v*(1 - PR18) - ow18h*(1 + PR18))/(PR18*(i18v + i18h - ow18v - ow18h) - (i18v - i18h - ow18v + ow18h))
    c37 =  (ow37v*(1 - PR37) - ow37h*(1 + PR37))/(PR37*(i37v + i37h - ow37v - ow37h) - (i37v - i37h - ow37v + ow37h))
    c_old = (c18 + c37)/2
    c = c_old/(2-c_old)

    return c

def twochannel10(tb10v, tb10h):
    """Simple 2 channel algorithm 10 GHz
    """
    ct=1.33313-0.01686*(tb10v-tb10h)

    return ct

def twochannel18(tb18v, tb18h):
    """Simple 2 channel algorithm 18 GHz
    """

    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    ow18v = tiepts[6]
    ow18h = tiepts[9]
    fy18v = tiepts[8]
    fy18h = tiepts[11]
    my18v = tiepts[7]
    my18h = tiepts[10]


    cf = ((tb18h - ow18h)*(my18v - ow18v) - (tb18v - ow18v)*(my18h - ow18h))/((fy18h - ow18h)*(my18v - ow18v) - (fy18v - ow18v)*(my18h - ow18h))
    cm = ((tb18h - ow18h) - cf*(fy18h - ow18h))/(my18h - ow18h)

    ct = cf + cm
    return ct

def twochannel22(tb22v, tb22h):
    """Simple 2 channel algorithm 22 GHz
    """
    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    ow22v = tiepts[24]
    ow22h = tiepts[27]
    fy22v = tiepts[26]
    fy22h = tiepts[29]
    my22v = tiepts[25]
    my22h = tiepts[28]

    cf = ((tb22h - ow22h)*(my22v - ow22v) - (tb22v - ow22v)*(my22h - ow22h))/((fy22h - ow22h)*(my22v - ow22v) - (fy22v - ow22v)*(my22h - ow22h))
    cm = ((tb22h - ow22h) - cf*(fy22h - ow22h))/(my22h - ow22h)

    ct = cf + cm
    return ct

def twochannel37(tb37v, tb37h):
    """Simple 2 channel algorithm 37 GHz
    """
    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    ow37v = tiepts[0]
    ow37h = tiepts[3]
    fy37v = tiepts[2]
    fy37h = tiepts[5]
    my37v = tiepts[1]
    my37h = tiepts[4]


    cf = ((tb37h - ow37h)*(my37v - ow37v) - (tb37v - ow37v)*(my37h - ow37h))/((fy37h - ow37h)*(my37v - ow37v) - (fy37v - ow37v)*(my37h - ow37h))
    cm = ((tb37h - ow37h) - cf*(fy37h - ow37h))/(my37h - ow37h)

    ct = cf + cm
    return ct

def twochannel37_linear(tb37v,tb37h):
    ct=1.21233-0.01879*(tb37v-tb37h)
    return ct

def n5esmr(tb18h):
    """Simple 1 channel algorithm at 18h measured by esmr at nimbus 5 
    """
    tiepts = [209.81, 187.18, 246.29, 145.29, 175.72, 235.15, 183.72,\
              219.66, 251.56, 108.46, 201.66, 237.16, 161.35, 242.91,\
              251.59, 82.13, 218.74, 232.14, 167.34, 235.26, 250.89, \
              88.26, 212.47, 234.16, 196.41, 208.82, 250.18, 128.23, \
              192.09, 236.42, 178.01, 229.52, 169.52, 220.94, 243.20, 196.94]

    ow18h = tiepts[9]
    fy18h = tiepts[11]
    my18h = tiepts[10]

    i18h = (fy18h+my18h)/2

    ct = (tb18h - ow18h)/(i18h - ow18h)
    return ct
