# Machine-learning-project-chip-problem
ML-final project for chip problem
1.The randon forest and XGboost alogorithm are recently added.

EA_XGB_optimal_cluster 是聚多个类，选最好类的前几名

EA_XGB_cluster_cluster_resample 聚多个类，选最好类的前几名；对于重复的采样点，选择种群中次优点进行替换

  更新了方法： (line 294)
  
    sampleSelect(candidate_population, sample_num): 
    
    -> sampleSelect(candidate_population, sample_Train):
    
  更改方法调用：
  
    sample = sampleSelect(pop, len(Sample_Train)) 
    
    -> sample = sampleSelect(pop, Sample_Train) 


EA_XGB_cluster_optimal_point 是聚多个类，选择每个类的最优点 

EA_XGB_cluster_optimal_point_resample 聚多个类，选择每个类的最优点；对于重复的采样点，选择种群中次优点进行替换 

  更新了方法： (line 291)
  
    sampleSelect(candidate_population, sample_num): 
    
    -> sampleSelect(candidate_population, sample_Train):
    
  更改方法调用：
  
    sample = sampleSelect(pop, len(Sample_Train)) 
    
    -> sample = sampleSelect(pop, Sample_Train) 

EA_XGB_cluster_OptPoint_v3

  更新了方法：
  
  mutation(individual, pb)： 进行矩阵化操作提速；同时加入无效突变更正机制（提速20%）
  
  iterate(Sample_Train, Sample_Points, Data, num_boost_round): 减少了一次种群clone（提速1%）
  
  新增了方法：
  
  monte_carlo_sampling(num_samples): 随机采样，用于初始种群的生成
