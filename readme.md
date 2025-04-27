# N/A

``` bash
# use venv, if conda,use python -m before pip and uv
pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple/ https://pypi.tuna.tsinghua.edu.cn/simple"
pip install uv
python -m uv pip install -r requirements.txt
```
