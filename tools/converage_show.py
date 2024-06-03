import re
import matplotlib.pyplot as plt

# 定义从日志文件中提取iteration和loss的函数
def extract_iterations_and_losses(log_file):
    iterations = []
    losses = []
    with open(log_file, 'r') as file:
        for line in file:
            # 使用正则表达式匹配iteration和loss
            iter_match = re.search(r"Iter \[(\d+)/80000\]", line)
            loss_match = re.search(r"loss: ([\d\.]+)", line)
            if iter_match and loss_match:
                iterations.append(int(iter_match.group(1)))
                loss = float(loss_match.group(1))
                # 如果损失值大于2，则将其置为2
                # if loss > 1:
                #     loss = 1
                losses.append(loss)
    return iterations, losses

# 列出所有日志文件的路径
# log_files = ['/home/sunzc/mmsegmentation/work_dirs/segformer_pipeline5/20230407_173201.log', '/home/sunzc/mmsegmentation/work_dirs/segformer_org/20230427_014536.log', '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_pipeline_new/20230425_112557.log', '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_org/20230510_220500.log']

# log_files = ['/home/sunzc/mmsegmentation/work_dirs/segformer_pipeline5/20230407_173201.log', '/home/sunzc/mmsegmentation-master/20230311_112127.log', '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_pipeline_new/20230425_112557.log', '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_org/20230510_220500.log', "/home/sunzc/mmsegmentation-master/20230317_084020.log"]

# labels = ['RNightSeg (Segformer)', 'Segformer', 'RNightSeg (DeepLabV3+) ', 'DeepLabV3+', 'UperSwin']


log_files = [ '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_pipeline_new/20230425_112557.log', '/home/sunzc/mmsegmentation/work_dirs/deeplabv3_org/20230510_220500.log']
labels = ['RNightSeg (DeepLabV3+) ', 'DeepLabV3+']


# 设置图形参数
plt.figure(figsize=(12, 8))
# plt.title('Training Loss over Iterations', fontsize=16)
plt.xlabel('Iterations', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.ylim(0, 1)
plt.xlim(0, 80000)

# 遍历每个日志文件，提取数据并绘制曲线
for log_file, label in zip(log_files, labels):
    iterations, losses = extract_iterations_and_losses(log_file)
    plt.plot(iterations, losses, label=label, linewidth=2)

# 美化曲线
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图形
plt.show()
