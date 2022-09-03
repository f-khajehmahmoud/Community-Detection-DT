from __future__ import division
import random
import numpy as np
from matplotlib.ticker import LinearLocator

# B1 = input("Please Enter for continue processing ***")
total_num_Total_Conclusion = 4000.0
alpha = .22
beta = 8.5
Imaginary_cap = 2100.0
questioner = 10.0
Decoder_paDecoder = 55.0
Encoder_paDecoder = 58.0
CNN_paDecoder = 62.0


def f1(x):
    return ((questioner / Decoder_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


def f2(x):
    return ((questioner / Encoder_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


def f3(x):
    return ((questioner / CNN_paDecoder) * (1 + alpha * ((x / Imaginary_cap) ** beta)))


def best_response_dynamics():
    global Imaging, Imaging_2
    Dataset_Part1_zoon = 0
    Dataset_Part2_zoon = 0
    Dataset_Part3_zoon = 0

    for i in range(int(total_num_Total_Conclusion)):
        Distnc = random.randint(1, 3)
        if (Distnc == 1):
            Dataset_Part1_zoon = Dataset_Part1_zoon + 1
        elif (Distnc == 2):
            Dataset_Part2_zoon = Dataset_Part2_zoon + 1
        else:
            Dataset_Part3_zoon = Dataset_Part3_zoon + 1
    l_1_start = Dataset_Part1_zoon
    l_2_start = Dataset_Part2_zoon
    l_3_start = Dataset_Part3_zoon

    # Initial Cost
    Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon)
    Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon)
    Dataset_Part3_zoon_cost = f3(Dataset_Part3_zoon)

    def be_better(l1, l2, l3):
        if ((f1(l1) > f2(l2 + 1)) or f1(l1) > f3(l3 + 1)):
            if (l1 > 0):
                return (True, 1)
        if ((f2(l2) > f1(l1 + 1)) or f2(l2) > f3(l3 + 1)):
            if (l2 > 0):
                return (True, 2)
        if ((f3(l3) > f1(l1 + 1)) or f3(l3) > f2(l2 + 1)):
            if (l3 > 0):
                return (True, 3)
        return (False, 0)

    continue_switching = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[0]
    count = 1
    while (continue_switching):
        l = [1, 2, 3]
        current_Distnc = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[1]
        l.remove(current_Distnc)
        if (current_Distnc == 1):
            Dataset_Part1_zoon = Dataset_Part1_zoon - 1

            if ((f2(Dataset_Part2_zoon + 1)) < (f3(Dataset_Part3_zoon + 1))):
                Dataset_Part2_zoon = Dataset_Part2_zoon + 1
                Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon)

            else:
                Dataset_Part3_zoon = Dataset_Part3_zoon + 1
                Dataset_Part2_zoon_cost = f3(Dataset_Part3_zoon)
        elif (current_Distnc == 2):
            Dataset_Part2_zoon = Dataset_Part2_zoon - 1

            if ((f1(Dataset_Part1_zoon + 1)) < (f3(Dataset_Part3_zoon + 1))):
                Dataset_Part1_zoon = Dataset_Part1_zoon + 1
                Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon)

            else:
                Dataset_Part3_zoon = Dataset_Part3_zoon + 1
                Dataset_Part3_zoon_cost = f3(Dataset_Part3_zoon)
        else:
            Dataset_Part3_zoon = Dataset_Part3_zoon - 1

            if ((f1(Dataset_Part1_zoon + 1)) < (f2(Dataset_Part2_zoon + 1))):
                Dataset_Part1_zoon = Dataset_Part1_zoon + 1
                Dataset_Part1_zoon_cost = f1(Dataset_Part1_zoon_cost)

            else:
                Dataset_Part2_zoon = Dataset_Part2_zoon + 1
                Dataset_Part2_zoon_cost = f2(Dataset_Part2_zoon_cost)
        continue_switching = be_better(Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon)[0]
        CC1 = Dataset_Part1_zoon
        CC2 = Dataset_Part2_zoon
        CC3 = Dataset_Part3_zoon
        CC1 = CC1 / 10000
        CC2 = CC2 / 10000
        CC3 = CC3 / 10000
        Imaging = (round(CC1, 2), round(CC2, 2), round(CC3, 2))
        BB1 = f1(Dataset_Part1_zoon)
        BB2 = f2(Dataset_Part2_zoon)
        BB3 = f3(Dataset_Part3_zoon)
        Imaging_2 = (round(BB1, 2), round(BB2, 2), round(BB3, 2))
        if (count % 100 == 0):
            print("number of cluster:" + str(count) + ": " + str(Imaging))
        count = count + 1
    print(str(count) + ": " + str(Imaging))
    print(Imaging_2)

    print('Following are results for Leader: ')
    print('Decoder zoon ' + str(Dataset_Part1_zoon))
    print('Encoder zoon ' + str(Dataset_Part2_zoon))
    print('Leader zoon ' + str(Dataset_Part3_zoon))
    best_response_Imaging = [Dataset_Part1_zoon, Dataset_Part2_zoon, Dataset_Part3_zoon]
    return (best_response_Imaging)


def objective(x):
    x1 = int(x[0])
    x2 = int(x[1])
    x3 = int(x[2])
    return (x1 * ((questioner / Decoder_paDecoder) * (1 + alpha * ((x1 / Imaginary_cap) ** beta))) +
            x2 * ((questioner / Encoder_paDecoder) * (1 + alpha * ((x2 / Imaginary_cap) ** beta))) +
            x3 * ((questioner / CNN_paDecoder) * (1 + alpha * ((x3 / Imaginary_cap) ** beta))))


from gekko import GEKKO


def Dime_Cost_solution():
    m = GEKKO(remote=False)
    m.options.SOLVER = 1
    Distnc_1 = m.Var(integer=True, lb=0, ub=Imaginary_cap)
    Distnc_2 = m.Var(integer=True, lb=0, ub=Imaginary_cap)
    Distnc_3 = m.Var(integer=True, lb=0, ub=Imaginary_cap)
    m.Minimize(
        Distnc_1 * ((questioner / Decoder_paDecoder) * (1 + alpha * ((Distnc_1 / Imaginary_cap) ** beta))) +
        Distnc_2 * ((questioner / Encoder_paDecoder) * (1 + alpha * ((Distnc_2 / Imaginary_cap) ** beta))) +
        Distnc_3 * ((questioner / CNN_paDecoder) * (1 + alpha * ((Distnc_3 / Imaginary_cap) ** beta)))
    )
    m.Equation(Distnc_1 + Distnc_2 + Distnc_3 == total_num_Total_Conclusion)
    m.solve(disp=False)
    print('Following are results for final processing: ')
    Distnc_1.value[0] = Distnc_1.value[0] / 1000
    Distnc_2.value[0] = Distnc_2.value[0] / 1000
    Distnc_3.value[0] = Distnc_3.value[0] / 1000
    print('Decoder zoon ' + str(round(Distnc_1.value[0], 2)))
    print('Encoder zoon ' + str(round(Distnc_2.value[0], 2)))
    print('Neighbor zoon ' + str(round(Distnc_3.value[0], 2)))
    opt_Imaging = [Distnc_1.value[0], Distnc_2.value[0], Distnc_3.value[0]]
    AA1 = f1(opt_Imaging[0])
    AA2 = f2(opt_Imaging[1])
    AA3 = f3(opt_Imaging[2])
    print(round(AA1, 2), round(AA2, 2), round(AA3, 2))
    return (opt_Imaging)


best_response_Imaging = best_response_dynamics()
opt_Imaging = Dime_Cost_solution()

Brands_result_final_accuracy_cost = objective(best_response_Imaging)

opt_Imaging_cost = objective(opt_Imaging)

PoA = Brands_result_final_accuracy_cost / opt_Imaging_cost
Brands_result_final_accuracy_cost = Brands_result_final_accuracy_cost / 1000
PoA = PoA / 10000

print('our Method accuracy is: ' + str(round(Brands_result_final_accuracy_cost, 2)))
print('Optimal Sensibillity is: ' + str(round(opt_Imaging_cost, 2)))
print('Price of responsibilities is: ' + str(round(PoA, 2)))

X_temp = np.linspace(0.05, 0.8, 20)
Y_temp = np.linspace(4, 12, 20)
X_temp, Y_temp = np.meshgrid(X_temp, Y_temp)
print(X_temp)
print(Y_temp)
Z_temp = []

for (x_i, y_i) in zip(X_temp, Y_temp):
    temp = []
    for (p, q) in zip(x_i, y_i):
        alpha = p
        beta = q
        print(alpha, beta)
        best_response_Imaging = best_response_dynamics()
        opt_Imaging = Dime_Cost_solution()

        Brands_result_final_accuracy_cost = objective(best_response_Imaging)

        opt_Imaging_cost = objective(opt_Imaging)

        PoA = Brands_result_final_accuracy_cost / opt_Imaging_cost
        temp.append(PoA)
    Z_temp.append(temp)

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FoDecoderatStrFoDecoderatter
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

plt.close()
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = plt.axes(projection='3d')
Z_temp = np.asarray(Z_temp)

surf = ax.plot_surface(X_temp, Y_temp, Z_temp,
                       rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_title("PoA with Varying Beta and Alpha", fontdict=None)

ax.set_zlim(1.01, 1.07)
ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_foDecoderatter(FoDecoderatStrFoDecoderatter('%.02f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

num_drivers = np.arange(1, 6300, 25)
poas = []
for i in num_drivers:
    total_num_Total_Conclusion = i
    best_response_Imaging = best_response_dynamics()
    opt_Imaging = Dime_Cost_solution()

    Brands_result_final_accuracy_cost = objective(best_response_Imaging)

    opt_Imaging_cost = objective(opt_Imaging)
    PoA = Brands_result_final_accuracy_cost / opt_Imaging_cost
    print(PoA)
    poas.append(PoA)

plt.close()
plt.plot(num_drivers, poas, color='r')

plt.xlabel('Total Number of image zoon in future')
plt.ylabel('Image')
plt.title('Total Number Qustion vs PoA')
plt.show()
