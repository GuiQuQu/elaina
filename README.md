

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

pytorch==2.1.2 cuda:12.1 cudnn:8.2.1

TODO List
1. [ ] (p0)利用分类的方式来做mp-docvqa,因此需要训练分类器，目前有两个可选的分类器，选用的分类器是internvl2-2b
2. [ ] (p0)利用分类的方式来做mp-docvqa,因此需要训练分类器，目前有两个可选的分类器，选用的分类器是clip,这个相当于复现了之前的结果
3. [ ] (p1)验证internvl2-2b在docvqa的结果
4. [ ] (p1)验证internvl2-2b + handle_ocr 在docvqa上的结果
5. [ ] (p1)尝试使用attention的机制来压缩image token,使得模型一次可以输入的image token数量变多，建立在高分辨率需要切图的基础上


ddp需要注意的问题
1. 开启gradient_checkpointing时，find_unused_parameters=False，否则会报错
问题

进度
## 9.18
1. mp-docvqa，使用internvl2-2b作分类的code写完了，正式开始训练

## 9.19
1. 完成了训练过程，发现代码没有补充保存最后的ckpt,需要补充一下这部分，已经做的训练不需要了，直接用中间权重即可。
2. trainer的模型权重保存方式是safetensors,可能会导致safetensors不好读取的问题
3. 写评测部分的代码，要求支持 1.多ckpt评测,通过传入ckpt_list的方式 2.评测结果保存到文件中 3. 自定义评测函数 4. 自定义评测指标
4. 支持TODO，优先级没有3高, 多卡，数据并行评测，从简单好实现的角度考虑，不要在最后wait各个进程都结束在收集结果，直接在进程内把结果保存到文件中，最后合并文件

## 9.24
1. TODO 加上了保存最后ckpt的代码，需要进行debug测试, ok
2. 目前不处理保存为safetensors的问题，在评测load model的时候会处理 ok
3. 写做作测试的代码,写完了等待debug,支持自定义模型forward 
4. 写metrics
5. 目前没有测试过

## 9.29
1. [x] debug save the last model
2. [x] debug tester code
3. [ ] 添加转换所有输入到符合model的device的逻辑(没懂)
4. [ ] write metrics

## 10.09
1. 在dataset中添加在test的时候保存的内容,然后将这些内容保存到对应的文件中
2. metrics应该怎么写(读保存的文件,载入json, 然后计算指标)


分类指标
1. 计算分类的auc
2. 针对统计结果，找到相同qid的的预测结果，将模型分进行排序，然后


```python
import numpy as np
import pandas as pd

def calc_auc(y_true, y_pred):
    pair = list(zip(y_true, y_pred))
    pair = sorted(pair, key=lambda x: x[1])
    df = pd.DataFrame([[x[0],x[1],i+1]for i, x in enumerate(pair)]  columns=['y_true', 'y_pred'])

    # 将预测值相同的样本的rank取平均值
    for k,v in df.y_pred.value_counts().items():
        if v > 1:
            rank_mean = df[df.y_pred == k]["rank"].mean()
            df.loc[df.y_pred == k, "rank"] = rank_mean

    pos_df = df[df.y_true == 1]
    m = pos_df.shape[0] # 正样本数
    n = df.shape[0] - m # 负样本数
    return (pos_df["rank"].sum() - m * (m + 1) / 2) / (m * n)
```