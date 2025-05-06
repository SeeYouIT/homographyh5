环境


S1 预训练HomographyNet模型
S1.1 生成预训练数据集
python b1.py
S1.2 预训练模型
python train.py

S2 训练我们的模型
S2.1 生成训练数据集
python b2.py
S2.2 标签转换
更改mark2h.py中的参数num = '000003'，后执行
python mark2h.py
将生成的xxxxxx.txt文件移动到/b/training和/b/validation
S2.2 训练模型
python train4paper.py

S3使用模型
python test4paper.py