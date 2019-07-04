Introduction<br>
Amyotrophic lateral sclerosis (ALS), also known as motor neuron disease (MND) or Lou Gehrig's disease, is a specific disease which causes the death of neurons controlling voluntary muscles. 
We are trying to create a series of models that predict ALSFRS-R scores that evaluates the progression of the disease over time with PRO-ACT database.
In PRO-ACT, these records are all patients getting ALS disease. Some also use the term motor neuron disease for a group of conditions of which ALS is the most common. 
It may begin with weakness in the arms or legs, which is limb onset. It may begin with difficulty speaking or swallowing, which is bulbar onset. 
About half of people develop at least mild difficulties with thinking and behavior and most people experience pain. 
Most eventually lose the ability to walk, use their hands, speak, swallow, and breathe.
The cause is not known in 90% to 95% of cases, but is believed to involve both genetic and environmental factors. 
The remaining 5–10% of cases are inherited from a person's parents. About half of these genetic cases are due to one of two specific genes. 
The underlying mechanism involves damage to both upper and lower motor neurons. The diagnosis is based on a person's signs and symptoms, with testing done to rule out other potential causes.

Mean response variable of each month
![image](https://github.com/Heanlily/Capstone/blob/master/%EF%BC%81%EF%BC%81%EF%BC%81%EF%BC%81%EF%BC%81%EF%BC%81%EF%BC%81.png)

Comparison of results from different models 
  | XGB | LGB | DNN | Weighted Ensemble 
 ------------- | ------------- | ------------- | ------------- | ------------- 
 R_square | 0.702 | 0.704 | 0.606 | 0.706 
 RMSE | 4.896 | 4.882 | 5.631 | 4.864 
 Slope | 0.988 | 1.001 | 0.946 | 1.009 
 Intercept | 0.460 | 0.225 | 2.246 | -0.194 
 Skewness | -0.537 | -0.522 | 2.194 | -0.648 

名称 | 地址 | 备注
-- | -- | --
小明 | 幸福路 | 打断点
