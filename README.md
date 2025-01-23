
# PMC-CLIP

使用到相关模型

a. microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract

b. axiong/pmc_oa

## 1. Debian环境部署


```bash
> cd /opt/Data/PythonVenv
> python3 -m venv PMC-CLIP
> source /opt/Data/PythonVenv/PMC-CLIP/bin/activate
```

## 2. 依赖安装

```bash
>
> pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
> python setup.py develop  # install pmc_clip with dev mode
>
```

## 3. 运行服务

```bash
>
> python example/RunCommond.py
>
```
