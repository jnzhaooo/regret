import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
import numpy as np
import re
# style name
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', 
# '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 
# 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 
# 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 
# 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 
# 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 
# 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 
# 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 
# 'tableau-colorblind10']
# plt.style.use('seaborn-whitegrid')
#plt.figure()
fig,ax = plt.subplots(4,3)
fig.tight_layout()



def plot_result(file_name:str, subplot_id: int):
    '''
    Visualize the result based on data.
    
    file_name: String
        The name of file containing the result.(e.g.: 20_tree)
    subplot_id: Int
        The ID of subwindow in 6.
    '''
    plt.subplot(4,3,subplot_id)
    x = np.array([0,0.2,0.5,0.8,1])
    # y1 = []
    # y2 = []
    # y3 = []

    # for i in range(5):
    #     y1_data = np.loadtxt('./result/{}/cost_regret_{}.txt'.format(file_name, int(x[i]*10)))
    #     y2_data = np.loadtxt('./result/{}/cost_minmax_{}.txt'.format(file_name, int(x[i]*10)))
    #     y3_data = np.loadtxt('./result/{}/cost_best_case_{}.txt'.format(file_name, int(x[i]*10)))

    #     y1.append(y1_data.mean())
    #     y2.append(y2_data.mean())
    #     y3.append(y3_data.mean())
    y1 = np.loadtxt('./result/{}/mean_regret_{}.txt'.format(file_name, int(10)))
    y2 = np.loadtxt('./result/{}/mean_minmax_{}.txt'.format(file_name, int(10)))
    y3  =np.loadtxt('./result/{}/mean_best_case_{}.txt'.format(file_name, int(10)))


    
    
    # s = UnivariateSpline(x,y,s=1)
    x_new = np.linspace(x.min(),x.max(),300)
    y_smooth1 = make_interp_spline(x,y1,k=1)(x_new)
    y_smooth2 = make_interp_spline(x,y2,k=1)(x_new)
    y_smooth3 = make_interp_spline(x,y3,k=1)(x_new)
    plt.scatter(x,y1,s=20,c='#1f77b4')
    plt.scatter(x,y2,s=20,c='#ff7f0e')
    plt.scatter(x,y3,s=20,c='#2ca02c')
    # plt.scatter(x,y1,s=20,c='lightblue')
    # plt.scatter(x,y2,s=20,c='lightsalmon')
    # plt.scatter(x,y3,s=20,c='lightgreen')

    
    # plot std
    # plt.errorbar(x, y1, yerr=std_1, fmt='o', color='red', ecolor='#1f77b4', elinewidth=3, capsize=0, label='regret Deviation')
    # plt.errorbar(x, y2, yerr=std_2, fmt='o', color='red', ecolor='#ff7f0e', elinewidth=3, capsize=0, label='minmax Deviation')
    # plt.errorbar(x, y3, yerr=std_3, fmt='o', color='red', ecolor='#2ca02c', elinewidth=3, capsize=0, label='best Deviation')

    # show the mean() on plt
    # for i in range(len(x)):
    #     plt.annotate('{:.2f}'.format(y1[i]), xy = (x[i], y1[i]))
    #     plt.annotate('{:.2f}'.format(y2[i]), xy = (x[i], y2[i]))
    #     plt.annotate('{:.2f}'.format(y3[i]), xy = (x[i], y3[i]))
    # plt.plot(x_new,y_smooth1,label="regret strategy", c='lightblue')
    # plt.plot(x_new,y_smooth2,label="worst-case strategy", c='lightsalmon')
    # plt.plot(x_new,y_smooth3,label="best-case strategy", c='lightgreen')
    plt.plot(x_new,y_smooth1,label="regret strategy", c='#1f77b4')
    plt.plot(x_new,y_smooth2,label="worst-case strategy", c='#ff7f0e')
    plt.plot(x_new,y_smooth3,label="best-case strategy", c='#2ca02c')

    plt.xlabel("probability of obstacles", fontsize=13)
    plt.ylabel("cost",fontsize=13)
    n_nodes = int(re.findall("\d+\.?\d*", file_name)[0])
    if 's' in file_name:
        plt.title('{} nodes ($\phi_1$)'.format(n_nodes), fontsize=13)
    else:
        plt.title('{} nodes ($\phi_2$)'.format(n_nodes), fontsize=13)
    if subplot_id==12:
        plt.legend()
    #plt.show()



plot_result('15_single',1)
plot_result('20_single',2)
plot_result('30_single',3)
plot_result('50_single',7)
plot_result('80_single',8)
plot_result('100_single',9)

plot_result('15',4)
plot_result('20_tree',5)
plot_result('30',6)
plot_result('50',10)
plot_result('80_tree',11)
plot_result('100_tree',12)

plt.savefig('./result/result.png')
plt.show()