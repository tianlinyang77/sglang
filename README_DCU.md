# <div align="center"><strong>SGLang</strong></div>

## sglang_dcu简介
SGLang是一个用于大型语言模型和多模态模型的高性能服务框架，旨在在从单个GPU到大型分布式集群的各种设置中提供低延迟和高吞吐量的推理，我们基于开源社区做了DCU平台的适配和针对性的优化。
其核心功能包括：快速运行时：通过RadixAttention提供高效的服务，用于前缀缓存、零开销CPU调度器、预填充解码分解、推测解码、连续批处理、分页注意力、张量/流水线/专家/数据并行性、结构化输出、分块预填充、量化（FP4/FP8/INT4/AWQ/GPTQ）和多LoRA批处理。
广泛的模型支持：支持各种语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral等）、嵌入模型（e5-Mistral、gte、mcdse）、奖励模型（Skywork）和扩散模型（WAN、Qwen-Image），易于扩展以添加新模型。与大多数Hugging Face模型和OpenAI API兼容。
强化学习和训练后主干：SGLang是一个经过验证的全球推广后端，具有原生强化学习集成，并被AReaL、Miles、slime、Tunix、verl等知名训练后框架采用。

## 使用源码编译方式安装
提供2种环境准备方式:

1. 基于光源pytorch2.5.1基础镜像环境:根据pytorch2.5.1、python、dtk及系统下载对应的镜像版本。

2. 基于现有python环境:安装pytorch2.5.1,pytorch whl包下载目录:[https://cancon.hpccube.com:65024/4/main/pytorch](https://cancon.hpccube.com:65024/4/main/pytorch),根据python、dtk版本,下载对应pytorch2.5.1的whl包。安装命令如下:
```shell
pip install torch* (下载的torch的whl包)
pip install setuptools wheel
```

### 源码编译安装
```shell
git clone  https://developer.sourcefind.cn/codes/OpenDAS/sglang.git #根据需要的分支进行切换
```
安装依赖:
```shell
pip install -r requirements_dcu.txt
```

- 提供2种源码编译方式(进入sglang目录):
```
编译安装sgl_kernel
cd sgl-kernel
python setup_hip.py install

1. 编译whl包并安装
python setup.py bdist_wheel 
cd dist
pip install sglang*

2. 源码编译sglang
pip install -e "python[all_hip]" --no-deps --no-build-isolation --no-index
```
### 运行基础环境准备
1、使用上面基于光源pytorch2.5.1基础镜像环境

2、根据pytorch2.5.1、python、dtk及系统下载对应的依赖包:
- flash_attn: [https://cancon.hpccube.com:65024/4/main/flash_attn](https://cancon.hpccube.com:65024/4/main/flash_attn)
- flash_mla: [https://download.sourcefind.cn:65024/4/main/flash_mla](https://download.sourcefind.cn:65024/4/main/flash_mla)
- lightop: [https://download.sourcefind.cn:65024/4/main/lightop](https://download.sourcefind.cn:65024/4/main/lightop)
- lmslim: [https://cancon.hpccube.com:65024/4/main/lmslim](https://cancon.hpccube.com:65024/4/main/lmslim)
- triton: [https://cancon.hpccube.com:65024/4/main/triton](https://cancon.hpccube.com:65024/4/main/triton)
- vllm: [https://download.sourcefind.cn:65024/4/main/vllm](https://download.sourcefind.cn:65024/4/main/vllm)

### 注意事项
+ 若使用 pip install 下载安装过慢,可添加源:-i  https://mirrors.huaweicloud.com/artifactory/pypi-public/simple

## 验证
- python -c "import sglang; print(sglang.\_\_version__)",

## Known Issue
- 无

## 参考资料
- [README_ORIGIN](README_ORIGIN.md)
- [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)


