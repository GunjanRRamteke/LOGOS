import random
from sklearn.linear_model import LinearRegression as Lreg
from sklearn.metrics import mean_absolute_error as mae_c
import matplotlib.pyplot as plt
import numpy as np

xmn, xmx = -2.0, 2.0
for i in range(6, 12, 1):
    print(i, np.round(np.linspace(xmn, xmx,i),3))


def error_plot(errorP_OC, X_cont, lb, xcc, ycc):

    x_mn, x_mx = xcc[0], xcc[1]
    y_mn, y_mx = ycc[0], ycc[1]

    plt.figure(figsize=(6,6))
    plt.bar(errorP_OC, X_cont, width=0.8, color='royalblue')

    xlab = np.linspace(x_mn, x_mx, 5)
    ylab = np.linspace(y_mn, y_mx, 5)

    plt.xlim([ x_mn, x_mx])
    plt.ylim([ y_mn, y_mx])

    plt.xticks(xlab)
    plt.yticks(ylab)
    plt.tick_params(axis='both', labelsize=20)
#    plt.set_facecolor('#EBEBEB')
    plt.grid(which='major', color='black', linewidth=1.0)
    plt.grid(which='minor', color='black', linewidth=1.0)

    plt.grid(linestyle='--', alpha=0.5, zorder=1)
    plt.savefig(f'{lb}_error_bar_.png',format='png', dpi= 600)
    plt.show()


def act_pred(x, y, lb, mn, mx):

    # Fit a linear regression model using np.polyfit
    degree = 1
    coefficients = np.polyfit(x, y, degree)  # coefficients: [slope, intercept]
    linear_fit = np.poly1d(coefficients)
    x_new = np.linspace(0, 2, 100)
    y_fit = linear_fit(x_new)

    # Calculate the correlation coefficient
    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]
    r_sq = correlation_coefficient ** 2
    slope, intercept = coefficients[0], coefficients[1]

    # MAE
    x = np.array(x)
    y = np.array(y)

    errorP = np.round([(x[i]-y[i])  for i in range(len(x))],2)
    color = np.round([abs(errorP[i])  for i in range(len(x))],2)
#    errorP = np.round([(x[i]-y[i])  for i in range(800)],2) 
#    color = np.round([abs(errorP[i])  for i in range(800)],2)

    mae_s = np.std(np.abs(x-y))
    print('mae_s', 'mae', 'rmse')
    print(mae_s)
    print(mae_c(x,y))
    print(np.sqrt(mae_c(x,y)))


    print(f'Correlation Coefficient: {correlation_coefficient:.8f}')
    print('r_sq', 'slope', 'intercept')

    print(r_sq)
    print(slope)
    print(intercept)

    '''------------------------------------------------------------------------------

    f1 = open('Cords','w')

    errorP.sort()
    errorP_OC = list(set(errorP))
    c1 = 0
    X_cont = []


    for j in errorP_OC:
        for i in (errorP):
            if ( j == i):
                c1 = c1 + 1

        X_cont.append(c1)
        print(j, c1, sep = '   ', file = f1)
        c1 = 0
    plt.bar(errorP_OC, X_cont)
    plt.show()


       ------------------------------------------------------------------------------'''

    X, Y = x, y
    b, a = np.polyfit(X, Y, deg=1)
    Zr = np.array([0.0,0.0] )
    dist_list = [np.linalg.norm(Zr - np.array([X[i], Y[i]])) for i in range(len(Y))]

    mn_I = np.argmin(dist_list)
    mx_I = np.argmax(dist_list)
    xseq = np.linspace(X[mn_I], X[mx_I], num=100)

    #_____________________________________________________________________________________

    cmap  = 'plasma'
    reg_l = 'yellow'
    reg_l = 'darkorange'
    reg_l = 'red'

    plt.figure(figsize=(6,6))
    plt.scatter(X,Y,s=20, c=color, cmap = cmap, alpha=0.5, zorder=2)
    plt.plot(xseq, a + b * xseq, color=reg_l, lw=2.0)
    
    x_mn, y_mn = mn[0], mn[1]
    x_mx, y_mx = mx[0], mx[1]

    lab = np.linspace(x_mn, y_mn, 5)
    plt.xlim([ x_mn, y_mn])
    plt.ylim([ x_mn, y_mn])

    plt.xticks(lab)
    plt.yticks(lab)
    plt.tick_params(axis='both', labelsize=20)
    plt.text(8, 13, 'RMSE = {}')
    plt.text(8, 11, 'RMSE = {}')

    plt.grid(which='major', color='black', linewidth=1.0)
    plt.grid(which='minor', color='black', linewidth=1.0)

    plt.grid(linestyle='--', alpha=0.5, zorder=1)
    plt.savefig(f'{lb}_act_vs_pred.png',format='png', dpi= 600)
    plt.show()



