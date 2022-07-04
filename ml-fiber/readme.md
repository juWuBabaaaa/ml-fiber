导入网络
```
from building import PoseNet
```

初始化网络
```
m = PoseNet(nstack=2, inp_dim=64, oup_dim=1)
```

导入训练好的模型参数
```
m.load_state_dict(torch.load("model_state_dict.pt"))
```

关键点预测
```
y = m(input)

# input 尺寸: 1 x 1 x 4的倍数 x 32 （4的倍数可能与nstack有关）
    * 1 - batchsize
    * 1 - rgb channels
    * 长
    * 宽
#
```output尺寸：1 x nstack x 1 x 4的倍数 x 32
    取出: output[:, -1, :, :, :][0, 0, :, :]  得到长x宽heatmap

