from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Positions of the L_i points
def L1(mu):  # Returns the x for L1 given mu
    flg = 0
    x = (mu / (3 * (1 - mu))) ** (1 / 3)
    max_iter = 1000
    n_it = 0
    while flg == 0 and n_it < max_iter:
        x1 = (mu * (1 - x) ** 2 / (3 - 2 * mu - x * (3 - mu - x))) ** (1 / 3)
        if np.abs(x - x1) < 1e-15:
            flg = 1
        x = x1
        n_it += 1
    return mu - 1 + x

def L2(mu):  # Returns the x for L2 given mu
    flg = 0
    x = (mu / (3 * (1 - mu))) ** (1 / 3)
    max_iter = 1000
    n_it = 0
    while flg == 0 and n_it < max_iter:
        x1 = (mu * (1 + x) ** 2 / (3 - 2 * mu + x * (3 - mu + x))) ** (1 / 3)
        if np.abs(x - x1) < 1e-15:
            flg = 1
        x = x1
        n_it += 1
    return mu - 1 - x

def L3(mu):  # Returns the x for L3 given mu
    flg = 0
    x = 1 - 6 * mu / 12
    max_iter = 1000
    n_it = 0
    while flg == 0 and n_it < max_iter:
        x1 = ((1 - mu) * (1 + x) ** 2 / (1 + 2 * mu + x * (2 + mu + x))) ** (1 / 3)
        if np.abs(x - x1) < 1e-15:
            flg = 1
        x = x1
        n_it += 1
    return mu + x

# Jacobi constant and derivatives :
def Omega(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return 1 / 2 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2 + 1 / 2 * mu * (1 - mu)

def Omega_x(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return x - (1 - mu) * (x - mu) / (r1 ** 3) - mu * (x - mu + 1) / (r2 ** 3)

def Omega_y(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return y - (1 - mu) * y / (r1 ** 3) - mu * y / (r2 ** 3)

def Omega_xx(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return 1 - (1 - mu) / r1 ** 3 + (3 * (1 - mu) * (x - mu) ** 2) / (r1 ** 5) - mu / r2 ** 3 + (
                3 * mu * (x - mu + 1) ** 2) / r2 ** 5

def Omega_xy(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return y * ((3 * (1 - mu) * (x - mu)) / r1 ** 5 + (3 * mu * (x - mu + 1)) / r2 ** 5)

def Omega_yy(x, y, mu):
    r1 = np.sqrt((x - mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - mu + 1) ** 2 + y ** 2)
    return 1 - (1 - mu) / r1 ** 3 - mu / r2 ** 3 + y ** 2 * ((3 * (1 - mu)) / r1 ** 5 + (3 * mu) / r2 ** 5)

# Function to compute the RTBP
def RTBP(t, x, mu):
    r1 = np.sqrt((x[0] - mu) ** 2 + x[1] ** 2)
    r2 = np.sqrt((x[0] - mu + 1) ** 2 + x[1] ** 2)
    Omega_x = x[0] - (1 - mu) * (x[0] - mu) / (r1 ** 3) - mu * (x[0] - mu + 1) / (r2 ** 3)
    Omega_y = x[1] - (1 - mu) * x[1] / (r1 ** 3) - mu * x[1] / (r2 ** 3)
    return [x[2], x[3], 2 * x[3] + Omega_x, -2 * x[2] + Omega_y]

# Jacobian of the RTBP
def Jacobian(x, y, mu):
    return np.array([[0, 0, 1, 0], [0, 0, 0, 1], [Omega_xx(x, y, mu), Omega_xy(x, y, mu), 0, 2],
                     [Omega_xy(x, y, mu), Omega_yy(x, y, mu), -2, 0]])

# Eigenvalues and eigenvectors of the L3
def vaps_L3(mu):
    return np.linalg.eig(Jacobian(L3(mu), 0, mu))[0]

def veps_L3(mu):
    return np.linalg.eig(Jacobian(L3(mu), 0, mu))[1]

def vaps_L2(mu):
    return np.linalg.eig(Jacobian(L2(mu), 0, mu))[0]

def veps_L2(mu):
    return np.linalg.eig(Jacobian(L2(mu), 0, mu))[1]

# Functions for the P o i n c a r Map ( details on code Assigment 3)
def g_sigma(x):
    return x[1]

def grad_g_sigma(x):
    return np.array([0, 1, 0, 0])

def Poincar_Map(x, ncrossings, mu,idir):
    if mu < 0.001:
        print('Error')
    elif mu < 0.002:
        val = 500
    elif mu < 0.003:
        val = 300
    elif mu < 0.005:
        val = 210
    elif mu < 0.01:
        val = 150
    elif mu < 0.1:
        val = 100
    elif mu < 0.2:
        val = 40
    else:
        val = 30
    #idir = 1
    if idir != 1 and idir != -1:
        return print("Error!")
    t_glob = 0
    y = x [1]
    xp = x [2]
    yp = x [3]
    x = x [0]
    for j in range ( ncrossings ) :
 #We first integrate until we cross y=0 ( backwards or forwards )
        t = 0
        t_span =[0 ,val ]
        if j != 0:
            sol = solve_ivp ( fun = lambda t , y : RTBP (t ,y , mu ) , t_span = t_span , y0 =[ x ,y , xp , yp
] , t_eval = np . array ([ idir *10**-10]) , rtol =3*10**-14 , atol =10**-14)
            t_glob += idir *10**-10
            x = sol .y [0][0]
            y = sol .y [1][0]
            xp = sol . y [2][0]
            yp = sol . y [3][0]
        teval = np . linspace (0 , val ,1000000)
        sol = solve_ivp ( fun = lambda t , y : RTBP (t ,y , mu ) , t_span = t_span , y0 =[ x ,y , xp , yp ] ,
t_eval = teval , rtol =3*10**-14 , atol =10**-14)
        t_ini = 0
        flg2 = 0
        if y >= 0:
            for i in range (1000000) :
                if sol . y [1][ i ] <0:
                    if idir == 1:
                        x = sol .y [0][ i -1]
                        y = sol .y [1][ i -1]
                        xp = sol . y [2][ i -1]
                        yp = sol . y [3][ i -1]
                        t_ini = sol . t [i -1]
                        flg2 =1
                        break
                    if idir == -1:
                        x = sol .y [0][ i ]
                        y = sol .y [1][ i ]
                        xp = sol . y [2][ i ]
                        yp = sol . y [3][ i ]
                        t_ini = sol . t [ i ]
                        flg2 =1
                        break
        if y < 0 and flg2 == 0:
            for i in range (1000000) :
                if sol . y [1][ i ] >0:
                    if idir == 1:
                        x = sol .y [0][ i -1]
                        y = sol .y [1][ i -1]
                        xp = sol . y [2][ i -1]
                        yp = sol . y [3][ i -1]
                        t_ini = sol . t [i -1]
                        break
                    if idir == -1:
                        x = sol .y [0][ i ]
                        y = sol .y [1][ i ]
                        xp = sol . y [2][ i ]
                        yp = sol . y [3][ i ]
                        t_ini = sol . t [ i ]
                        break

 #We do Newton as many times as needed
        t_span = [0, val]
        flg = 0
        flg4 = 0
        iter_pm = 0
        while (flg == 0) and (iter_pm < 1000):
            if t==0:
                t=np.array([t])
            sol = solve_ivp(fun=lambda t, y: RTBP(t, y, mu), t_span=t_span, y0=[x, y, xp, yp],
                    t_eval=np.array(t), rtol=3e-14, atol=1e-14)
            t1 = t - (g_sigma(sol.y) / (np.dot(grad_g_sigma(sol.y), RTBP(0, sol.y, mu))))
            t1 = np.real(t1) % (2 * np.pi)
            if np.abs(t - t1) > 1.5 and flg4 == 0:
                print('Oscillations')
                flg4 = 1
            if np.abs(t - t1) < 1e-14:
                flg = 1
            t = t1
            iter_pm += 1
            if iter_pm > 999:
                print('Error! Massa llarg')
        print(t)
        sol = solve_ivp(fun=lambda t, y: RTBP(t, y, mu), t_span=t_span, y0=[x, y, xp, yp],
                        t_eval=np.array(t), rtol=3e-14, atol=1e-14)
        x = sol.y[0][0]
        y = sol.y[1][0]
        xp = sol.y[2][0]
        yp = sol.y[3][0]
        t_glob += t + t_ini
    return [sol.y.real, t_glob]
