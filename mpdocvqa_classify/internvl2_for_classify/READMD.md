
```bash
transformers==4.37.2
```

need flash-attn


模型结构
1. internvl2 输入一张图像和一个问题, 构造prompt来让模型回答A或者B,得到internvl2的输出之后,拿最后一层的最后一个token的输出来当作emb
emb后面接mlp来做二分类

模型的损失计算也使用二分类来完成