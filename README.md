# Particle-Filter-for-IBHP

Particle Filter for IBHP(Python Realization). Paper: ***The Indian Buffet Hawkes Process to Model Evolving Latent Influences***

## Update Particle Weight

Each particle weight updated by:

<img src="/Users/huangjinnan/Library/Application Support/typora-user-images/image-20211207094956358.png" alt="image-20211207094956358" style="zoom:12%;" />

In which,

<img src="/Users/huangjinnan/Library/Application Support/typora-user-images/image-20211207095224295.png" alt="image-20211207095224295" style="zoom:12%;" />

## Update Model Parameters

使用MH算法更新超参数的后验分布，可以得到几个超参数的后验分布是正比于先验乘以似然。

<img src="/Users/huangjinnan/Library/Application Support/typora-user-images/image-20211207095335987.png" alt="image-20211207095335987" style="zoom:12%;" />

t=n时，MH算法中的似然（Hawkes Likelihood）计算方法如下：

<img src="/Users/huangjinnan/Library/Application Support/typora-user-images/image-20211207095527745.png" alt="image-20211207095527745" style="zoom:12%;" />

对应的对数似然：

<img src="/Users/huangjinnan/Library/Application Support/typora-user-images/image-20211207095544725.png" alt="image-20211207095544725" style="zoom:12%;" />
