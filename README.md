

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
1. TODO 加上了保存最后ckpt的代码，需要进行测试
2. 目前不处理保存为safetensors的问题，在评测load model的时候会处理
