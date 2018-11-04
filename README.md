# MLEM_ct_figure
基于MLEM的CT图像降噪
ML（Maximum Likehood）最大似然估计
EM（Expection Maxiumization）最大化期望值
算法流量比较复杂，具体请自行查阅相关资料。

分python、Matlab两个版本实现MLEM

## matlab_script
  demo_MLEM_Simulation.m 为主函数，其余与辅助函数。其中主函数用到了Maltab的内置函数phantom(im_size, im_size)，这里是生成了大小为im_size, im_size的头部ct图片，为了在python实现，把图片矩阵元素保存为im.csv,在python上读取即可。
  另外，Matlab上还用到了自带的加噪函数imnoise，这函数我在python也自己写了一次，函数为：PoissonNoise(img, noise_scale)
  
## python版本
   MLEM.py ：所有辅助函数和自函数都放一起，运行时间较长，矩阵运算建议开启GPU，最后输出10个迭代图，可以看到每一次迭代，图像都会比上一次迭代清晰。

## images
  输出的图，只放了部分。
