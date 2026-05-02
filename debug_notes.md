现象： 运行代码时报错 FileNotFoundError: [WinError 3] 系统找不到指定的路径。: './data\\MNIST'

原因分析： Windows 系统下，由于相对路径解析或权限问题，torchvision 无法自动创建不存在的多级 data 文件夹。

修改点： 在本地手动创建了 data 文件夹（或在代码中将 root 改为了基于 os.path.abspath 的绝对路径）。

现象： 发现原代码中 epochs = 3，与公众号文章中展示的 5 个训练周期 不符，导致初始测试准确率与文章有细微差距。

原因分析： 默认设置的训练轮数较少，模型尚未完全收敛。

修改点： 将 simple_cnn.py 和 train_lenet.py 中的 epochs 统一修改为 5，使模型充分训练，并与文章保持一致。