import numpy as np
from assignement8 import *
from tqdm import tqdm
import concurrent.futures
##Bisection method
def function(x):
    # Define your function here, for example:
    return (x-1)*(x-101)



# Exemple d'utilisation :
# Définir la fonction f et les intervalles sur lesquels vous souhaitez exécuter la bissection
# results = parallel_bisection(f, intervals)


def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        print("Bisection method may not converge.")
        return None

    iteration = 0
    while (b - a) / 2 > tol and iteration < max_iter:
        midpoint = (a + b) / 2
        if func(midpoint) == 0:
            return midpoint  # Found exact root
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iteration += 1

    return (a + b) / 2

def bisection(f, a, b):
    tol = 1e-6

    fa = f(a)
    fb = f(b)

    condition = True
    while condition:
        if np.abs(fa) < tol:
            return a
        if np.abs(fb) < tol:
            return b

        if np.sign(fa) == np.sign(fb):
            raise Exception("The scalars a and b do not bound a root")

        # get midpoint
        m = (a + b) / 2
        fm = f(m)

        if np.abs(fm) < tol:
            return m
        elif np.sign(fa) == np.sign(fm):
            a = m
        elif np.sign(fb) == np.sign(fm):
            b = m

def compute_bisection(f, interval):
    a, b = interval
    return bisection(f, a, b)

# Fonction principale pour la parallélisation
def parallel_bisection(f, intervals):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(compute_bisection, [f]*len(intervals), intervals)
    return list(results)
# # Define the interval [a, b] where the root lies
# a = -100
# b = 100
#
# root = bisection_method(function, a, b)
# if root is not None:
#     print(f"Approximate root: {root}")

# ##Computation of L1
# mu=0.1
# def f1(x):
#     return((mu*(1+x)**2/(3-2*mu+x*(3-mu+x)))**(1/3))
#
# xi_0=(mu/3*(1-mu))**(1/3)
#
# while np.abs(f1(xi_0)-xi_0)>1e-14:
#     xi_0=f1(xi_0)
#
# xi_1=mu-1-xi_0
# ##computation of L2
# def f2(x):
#     return((mu*(1-x)**2/(3-2*mu-x*(3-mu-x)))**(1/3))
#
# xi_0=(mu/(3*(1-mu)))**(1/3)
# xi_2=f2(xi_0)
#
# while np.abs(xi_2-xi_0)>1e-14:
#     xi_0=xi_2
#     xi_2=f2(xi_2)
#
# xi_2=mu-1+xi_0
# ##computation of L3
# def f3(x):
#     return(((1-mu)*(1+x)**2/(1+2*mu+x*(2+mu+x)))**(1/3))
#
#
# xi_0=1-7/12*mu
# xi_3=f3(xi_0)
#
# while np.abs(xi_3-xi_0)>1e-14:
#     xi_0=xi_3
#     xi_3=f3(xi_3)
#
#
# xi_3=mu+xi_0
# ##computation of C(l1), C(l2) and C(l3):
#
# def r1(x,y):
#     return(np.sqrt((x-mu)**2+y**2))
#
# def r2(x,y):
#     return(np.sqrt((x-mu+1)**2+y**2))
#
# def omega(x,y):
#     return((1/2)*(x**2+y**2)+1/2*mu*(1-mu)+(1-mu)/r1(x,y)+mu/r2(x,y))
#
# def C(X):
#     x=X[0]
#     y=X[1]
#     xp=X[2]
#     yp=X[3]
#     return(2*omega(x,y)-(xp**2+yp**2))
#
# L3=np.array([xi_3,0,0,0])
# L2=np.array([xi_1,0,0,0])
# L1=np.array([xi_2,0,0,0])
#
# print(f"xL1={xi_2}, C(L1)={C(L1)}")
# print(f"xL2={xi_1}, C(L2)={C(L2)}")
# print(f"xL3={xi_3}, C(L3)={C(L3)}")



###First routine
mu=0.1

print(L3(mu))

def rout_1(C,x,sign):
    if sign:
        return(np.array([x,0,0,-np.sqrt(-C+2*Omega(x,0,mu))]))
    else:
        return(np.array([x,0,0,-np.sqrt(-C+2*Omega(x,0,mu))]))


def F(x,C):
    x0=rout_1(C,x,False)
    a=Poincar_Map(x0,2,mu,1)

    if a[0][1]>1*10**-14:
        print("error")

    return(a[0][2][0])


# c_val = 3.189
# xincC = L3(mu) + 1e-5
# xincini = 1e-5
#
# x1 = xincC
# x2 = x1
# a = F(x1, c_val)
# b = a
# print(x1, x2, a, b)
#
# while a * b > 0:
#     x2 = x1
#     b = a
#     x1 += 1e-3
#     a = F(x1, c_val)
#     print(x1, x2, a, b)

x1=1.061618908571058
x2=1.060618908571058
# x=bisection_method(lambda x: F(x, 3.189), x1, x2)
# print(x)
# x=bisection(lambda x: F(x, 3.189), x1, x2)
# print(x)

#### COMPUTATION OF THE OTHER POINTS
# C_results = np.linspace(3.189, 2.1, round((3.189 - 2.1) / 0.001) + 1)
# x_results = np.zeros(len(C_results))
# x_results[0] = 1.0610140258055256  # We put the first point
#
# pbar = tqdm(total=len(C_results) - 1)
# for i in range(len(C_results) - 1):
#     i = i + 1
#     Ct = C_results[i]
#     x1 = x_results[i - 1]
#     x2 = x1
#     a = F(x1, Ct)
#     b = a
#     while a * b > 0:
#         x2 = x1
#         b = a
#         x1 += 1e-3
#         a = F(x1, Ct)
#     print("bisection step")
#     x_results[i] = bisection(lambda x: F(x, Ct), x1, x2)
#     pbar.update(1)
# pbar.close()
#
# # PLOTS
# plt.plot(x_results, C_results)
# plt.ylabel(r"$C$")
# plt.xlabel(r"$x_0$")
# plt.show()

# Some orbits plot
ind_plot = np.array([1089, 689, 39])
for i in ind_plot:
    mu = 0.1
    x = x_results[i]
    Ct = C_results[i]
    x0 = rout_1(x, Ct, mu)
    val = PoincareMap(x0, 2, 0.1)[1] * 2
    sol = solve_ivp(fun=lambda t, y: RTBP(t, y, mu), t_span=[0, val], y0=x0, t_eval=np.linspace(0, val, 1000),
                    rtol=3e-14, atol=1e-14)
    nom = r"$C=$" + str(Ct)
    plt.plot(sol.y[0], sol.y[1], label=nom,)
    plt.scatter(L3(mu), 0, label=r"$L_3$")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$x$")
    plt.legend()
    plt.show()




