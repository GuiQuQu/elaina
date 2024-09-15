

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

