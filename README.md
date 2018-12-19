### 剪枝步骤
1. 运行 train_origin_net.py 生成原始的模型，并保存其参数
2. 运行 pure_net.ipynb 对每一层进行剪枝操作，并保存下剪枝后的维度和权重， 保存为 pb 文件
3. 运行 test_origin_net.py 和 test_pure_net.py 查看剪枝后的模型与原模型精度对比
4. 运行 train_pure_net.py rebirth剪枝后的模型 

