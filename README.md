# OpenSourceChineseLLM

收集了部分开源中文大模型（10b以内，可能也不算大模型），以及他们的资料。欢迎分享合适的模型~

**注意：以下排名只按作者接触到的顺序，不包含任何其他含义！**

|                  | 基座模型         | 额外的预训练数据集                                                                                                                                                                  | 指令微调数据集                                                                                                                                                                                                                                                                          | 额外的训练过程 | 量化支持 | 项目特点                  |
|------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------:|------|-----------------------|
| ChatGLM-6b       | GLM          | NA                                                                                                                                                                         | NA                                                                                                                                                                                                                                                                               |   NA    | ✔      | 较早的开源模型，有很多项目集成       |
| Chinese-Vicuna   | LLaMA-7B/13B | ❌                                                                                                                                                                          | belle+guanaco                                                                                                                                                                                                                                                                    |    ❌    | ✔    | 易上手；以Lora方式训练，显存需求小   |
| Chinese-ChatLLaMA | LLaMA-7B/13B | [中英平行语料](https://statmt.org/wmt18/translation-task.html#download)、[中文维基、社区互动、新闻数据](https://github.com/CLUEbenchmark/CLUECorpus2020)、[科学文献](https://github.com/ydli-ai/CSL) | [BELLE](https://github.com/LianjiaTech/BELLE), [pCLUE](https://github.com/CLUEbenchmark/pCLUE), [CSL](https://github.com/ydli-ai/CSL), [CLUECorpus](https://github.com/brightmart/nlp_chinese_corpus#5%E7%BF%BB%E8%AF%91%E8%AF%AD%E6%96%99translation2019zh), [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset) |    ❌    | ✔     | 数据量大； TencentPretrain |
| BELLE            | TODO         |                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                               |         |       |                       |
| Open-Llama                 | TODO         |                                                                                                                                                                            |                                                                                                                                                                                                                                                                                                                                               |         |       |                       |

## ChatGLM-6b

### 开发者

清华大学，智谱AI

### 论文

[1] A. Zeng et al. [GLM-130B: An Open Bilingual Pre-Trained Model](https://openreview.net/pdf?id=-Aw0rrrPUF). ICLR 2023.

[2] Z. Du et
al. [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360). ACL 2022.

### 项目地址

[https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

### 模型下载地址

[1] 原始模型

* HuggingFace:  "THUDM/chatglm-6b"
* [清华网盘](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/)

[2] int4量化模型

* HuggingFace: "THUDM/chatglm-6b-int4"

[3] int4量化+embedding量化模型

* HuggingFace: "THUDM/chatglm-6b-int4-qe"

### 模型简述

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需
6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 **1T 标识符的中英双语训练**，辅以**监督微调**、**反馈自助**、**人类反馈强化学习**等技术的加持，62
亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考[博客](https://chatglm.cn/blog)。

### 资源消耗

NA表示原始项目未提供

| 名称        | 内存需求  | 最低 GPU 显存（推理） | 最低 GPU 显存（高效参数微调） |
|-----------|-------|---------------|-------------------|
| FP16（无量化） | 13GB  | 13 GB         | 14 GB             |
| INT8      | NA    | 8 GB          | 9 GB              |
| INT4      | 5.2GB | 6 GB          | 7 GB              |
| INT4+QE   | NA    | 4.3 GB        | NA                |
| CPU       | 32GB  | -             | -                 |

### 提供DEMO的场景

* 自我认知
* 提纲写作
* 文案写作
* 邮件写作助手
* 信息抽取
* 角色扮演
* 评论比较
* 旅游向导

### licence：

仓库的代码依照 [Apache-2.0](https://github.com/THUDM/ChatGLM-6B/blob/main/LICENSE) 协议开源，ChatGLM-6B
模型的权重的使用则需要遵循 [Model License](https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE)。

## Chinese-Vicuna

### 开发者

个人开发者：Chenghao Fan, Zhenyi Lu and Jie Tian

### 项目地址

[https://github.com/Facico/Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)

### 模型下载地址

[1] belle+guanaco语料上训练的7b模型

* HuggingFace:  "Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco"

[2] belle+guanaco语料上训练的13b模型

* HuggingFace: "Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"

[3] 中文医学问答垂直语料上训练的

* HuggingFace: "Chinese-Vicuna/Chinese-Vicuna-continue-finetune-3.5epoch-cMedQA2"

### 模型简述

基于LLAMA 7b/13b， 利用lora模型进行指令微调。项目的主要目的在于语言模型的普及工作。

### 提供DEMO的场景

* 多轮交互
    * 制作披萨
    * 学生购买手机
    * 介绍北京
* 多轮翻译
* 知识问答
* 开放式、观点类回答
* 数值计算、推理
* 写信、写文章
* 写代码
* 伦理、拒答类（alignment相关）

### licence：

仓库的代码依照 Apache-2.0 协议开源，ChatGLM-6B 模型的权重的使用则需要遵循 gpl-3.0。

## Chinese-ChatLLaMA

### 开发者

Authors: Yudong Li, Zhe Zhao, Yuhao Feng, Cheng Hou, Shuang Liu, Hao Li, Xianxu Hou

Corresponding Authors: Linlin Shen, Kimmo Yan

### 项目地址

[https://github.com/ydli-ai/Chinese-ChatLLaMA](https://github.com/ydli-ai/Chinese-ChatLLaMA)

### 模型下载地址

[1] 7b模型

* HuggingFace:  "P01son/ChatLLaMA-zh-7B"

[2] 7b-int4

* HuggingFace: "P01son/ChatLLaMA-zh-7B-int4"

### 模型简述

本项目向社区提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。 模型基于 TencentPretrain 多模态预训练框架构建， 将陆续开放 7B、13B、30B、65B 规模的中文基础模型
LLaMA-zh 权重。

ChatLLaMA 支持简繁体中文、英文、日文等多语言。 LLaMA 在预训练阶段主要使用英文，为了将其语言能力迁移到中文上，首先进行中文增量预训练， 使用的语料包括中英平行语料、中文维基、社区互动、新闻数据、科学文献等。再通过
Alpaca 指令微调得到 Chinese-ChatLLaMA。

### 提供DEMO的场景

* 推荐/问答
* 机器翻译
* 数学/代码
* 机器写作


### licence：

仓库的代码依照 Apache-2.0 协议开源，ChatGLM-6B 模型的权重的使用则需要遵循 gpl-3.0。