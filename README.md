## QA_Bert

这个小项目是利用bert预训练模型来做一个QA问题，旨在练习怎么使用pre-train model来完成自己下游任务。

### 1.    数据集

数据集比较小，已经放在项目的data文件了

### 2.    相关库

```python
pip pip install transformers==4.5.0 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
```

### 3.    运行

```python
# 终端运行
python -m main
```



**备注**：

* 运行模型的时候不要挂梯子
* 模型比较大，没有放在项目里，预测结果放在test_result