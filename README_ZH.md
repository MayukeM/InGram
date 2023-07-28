# InGram: 基于关系图的归纳知识图嵌入
这个代码库是以下论文的官方实现： 

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, To appear in the 40th International Conference on Machine Learning (ICML), 2023.

所有代码都由Jaejun Lee (jjlee98@kaist.ac.kr)编写。如果你使用了这个代码或数据，请引用我们的论文。

```bibtex
@article{ingram,
  author={Jaejun Lee and Chanyoung Chung and Joyce Jiyoung Whang},
  title={{I}n{G}ram: Inductive Knowledge Graph Embedding via Relation Graphs},
  year={2023},
  journal={arXiv preprint arXiv:2305.19987},
  doi = {10.48550/arXiv.2305.19987}
}
```

## 环境要求

我们使用Python 3.8和PyTorch 1.12.1与cudatoolkit 11.3。

你可以使用以下命令安装所有的依赖：

```shell
pip install -r requirements.txt
```

## 复现论文中的结果

我们在 NVIDIA RTX A6000，NVIDIA GeForce RTX 2080 Ti和NVIDIA GeForce RTX 3090上进行了所有实验。我们提供了我们用于在14个数据集上生成归纳链接预测结果的检查点。如果你想使用这些检查点，请将解压后的ckpt文件夹放在代码的相同目录下。

你可以从[https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing下载这些检查点 ↗](https://drive.google.com/file/d/1aZrx2dYNPT7j4TGVBOGqHMdHRwFUBqx5/view?usp=sharing下载这些检查点)。

运行以下命令以重新生成我们论文中的结果：

```python
python3 test.py --best --data_name [dataset_name]
```

## 从头开始训练

要从头开始训练InGram，请使用参数运行`train.py`。请参考`my_parser.py`以获得参数的例子。请使用附录C中提供的范围调整我们模型的超参数，因为由于随机性，最佳超参数可能不同。

`train.py`的参数列表：
- `--data_name`：数据集的名称
- `--exp`：实验名称
- `-m, --margin`：$\gamma$
- `-lr, --learning_rate`：学习率
- `-nle, --num_layer_ent`：$\widehat{L}$
- `-nlr, --num_layer_rel`：$L$
- `-d_e, --dimension_entity`：$\widehat{d}$
- `-d_r, --dimension_relation`：$d$
- `-hdr_e, --hidden_dimension_ratio_entity`：$\widehat{d'}/\widehat{d}$
- `-hdr_r, --hidden_dimension_ratio_relation`：$d'/d$
- `-b, --num_bin`：$B$
- `-e, --num_epoch`：要运行的epoch数
- `--target_epoch`：要运行测试的epoch（仅用于test.py）
- `-v, --validation_epoch`：验证的时长
- `--num_head`：$\widehat{K}=K$
- `--num_neg`：每个三元组的负三元组数
- `--best`：使用提供的检查点（仅用于test.py）
- `--no_write`：不保存检查点（仅用于train.py）