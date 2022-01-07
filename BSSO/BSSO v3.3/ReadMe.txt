v3.3版本：
使用真实仿真脚本进行选点评估，每次脚本运行大约一分钟，但是整体代码跑100轮的时间远远不止100分钟
仿真脚本文件为“chip_simulation”，调用方式如下：
design_para = np.array([0.850, 0.240, 0.220, 0.032, 11.200])
design_csv = np.savetxt('design_para.csv', design_para)
cmd_cae = os.system('abaqus cae noGUI=D:\MLproject\BSSO_Chippackaging\\5_variables_result\chip_simulation.py')
with open('D:\MLproject\BSSO_Chippackaging\\5_variables_result\\result_miao\chip_warpage.csv') as f:
    warpageData = np.loadtxt(f)
warpage = warpageData[0]
但是在嵌入之前shishan代码的过程中，进行了比较多的修改。
