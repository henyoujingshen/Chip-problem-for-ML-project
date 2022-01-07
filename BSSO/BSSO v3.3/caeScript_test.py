import os

import numpy as np

design_para = np.array([0.850, 0.240, 0.220, 0.032, 11.200])
design_csv = np.savetxt('design_para.csv', design_para)

cmd_cae = os.system('abaqus cae noGUI=D:\MLproject\BSSO_Chippackaging\\5_variables_result\chip_simulation.py')

with open('D:\MLproject\BSSO_Chippackaging\\5_variables_result\\result_miao\chip_warpage.csv') as f:
    warpageData = np.loadtxt(f)

warpage = warpageData[0]
print(warpage)