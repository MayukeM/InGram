import argparse  # argparse 是一个命令行参数解析包，可以用来方便地读取命令行参数
import json


def parse(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./data/", type=str)  # default: 默认值
    parser.add_argument('--data_name', default='NL-100', type=str)  # 数据集名称，
    parser.add_argument('--exp', default='exp', type=str)  # 实验名称，用于保存模型
    parser.add_argument('-m', '--margin', default=2, type=float)  # 超参数，损失函数中的margin
    parser.add_argument('-lr', '--learning_rate', default=5e-4, type=float)  # 超参数， 学习率
    parser.add_argument('-nle', '--num_layer_ent', default=2, type=int)  # 实体编码器的层数
    parser.add_argument('-nlr', '--num_layer_rel', default=2, type=int)  # 关系编码器的层数
    parser.add_argument('-d_e', '--dimension_entity', default=32, type=int)  # 实体编码器的维度
    parser.add_argument('-d_r', '--dimension_relation', default=32, type=int)  # 关系编码器的维度
    parser.add_argument('-hdr_e', '--hidden_dimension_ratio_entity', default=8, type=int)  # 实体编码器的隐层维度
    parser.add_argument('-hdr_r', '--hidden_dimension_ratio_relation', default=4, type=int)  # 关系编码器的隐层维度
    parser.add_argument('-b', '--num_bin', default=10, type=int)  # 实体编码器的隐层维度
    parser.add_argument('-e', '--num_epoch', default=10000, type=int)  # 训练的轮数
    if test:  # 测试时加载的模型轮数
        parser.add_argument('--target_epoch', default=6600, type=int)  # 测试时加载的模型轮数
    parser.add_argument('-v', '--validation_epoch', default=200, type=int)  # 每隔多少轮进行一次验证
    parser.add_argument('--num_head', default=8, type=int)  # 多头注意力机制的头数
    parser.add_argument('--num_neg', default=10, type=int)  # 负样本的数量
    parser.add_argument('--best', action='store_true')  # 是否加载最优模型，这里的store_true是指，如果命令行中有这个参数，则args.best为True，否则为False
    if not test:  # 训练时是否加载模型
        parser.add_argument('--no_write', action='store_true')  # 是否保存模型，no_write为True时不保存模型

    args = parser.parse_args()  # 解析命令行参数

    if test and args.best:  # 测试时加载最优模型
        remaining_args = []
        with open(f"./ckpt/best/{args.data_name}/config.json") as f:
            configs = json.load(f)
        for key in vars(args).keys():
            if key in configs:
                vars(args)[key] = configs[key]
            else:
                remaining_args.append(key)

    return args
