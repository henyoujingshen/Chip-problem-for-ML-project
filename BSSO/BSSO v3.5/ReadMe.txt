v3.5版本：
更新了画图功能，samplesContour，可以绘制2D测试函数的全局或局部等高线图，可以绘制局部区域收缩图，可以绘制选点分布图，global cluster分布图。
调整所有超参数位置在代码开头。
修正：将全局聚类由X+y聚类变为X聚类
添加了EI-LP代码，可以使用sequential-EI或batch-EI来进行优化(LCB等同理)
