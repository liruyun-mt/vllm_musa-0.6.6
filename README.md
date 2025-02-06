# vllm_musa-0.6.6
此仓库用于 musify vllm v0.6.6 to MUSA device

# 镜像
以 S4000 为例：
```
docker run \
-it \
--restart always \
--network host \
--name vllm_v0.6.6 \
--shm-size 80G \
--privileged \
--env MTHREADS_VISIBLE_DEVICES=all \
-v /home/mccxadmin/ruyun.li:/home/mt/ruyun.li \
-v /jfs/:/jfs/ \--hostname=work \
--security-opt seccomp=unconfined \
registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py310 \
/bin/bash
```
# 使用说明

```
git clone https://github.com/liruyun-mt/vllm_musa-0.6.6.git
cd vllm_musa-0.6.6
# vllm 编译 
pip3 install ray[default]==2.10.0
pip3 install transformers==4.44.0
bash build_musa.sh
```

目前还有报错


# 此仓库在vllm-v0.6.6上做了什么
1. 使用musa_porting 自动将.cu .cuh文件转换为 .mu .muh文件，同时替换所有文件中 cuda 相关的字符串:

> musify 脚本来源于 vllm_musa-0.4.2 : https://github.com/MooreThreads/vllm_musa/blob/main/musa_porting.py

```
cd vllm-v0.6.6
# 在vllm v0.6.6 的基础上做 musify
python3 musa_porting.py
```


2. 编译：

> 编译脚本来源于 musa_vllm-0.4.2: https://github.com/MooreThreads/vllm_musa/blob/main/build_musa.sh

```
bash build_musa.sh
```

# 报错记录

* 【20250206】attention_kernels.mu找不到
```
Compiling objects...
Using envvar MAX_JOBS (128) as the number of workers...
ninja: error: '/home/mt/ruyun.li/code/vllm_musa.latest/csrc_musa/attention/attention_kernels.mu', needed by '/home/mt/ruyun.li/code/vllm_musa.latest/build/temp.linux-x86_64-cpython-310/csrc_musa/attention/attention_kernels.o', missing and no known rule to make it
Traceback (most recent call last):
  File "/opt/conda/envs/py310/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2096, in _run_ninja_build
    subprocess.run(
  File "/opt/conda/envs/py310/lib/python3.10/subprocess.py", line 526, in run
    raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '128']' returned non-zero exit status 1.
```

归因：set_up.py 找不到 attention_kernels.mu，是因为 v0.4.2 升级到 v0.6.6 后，attention_kernels.cu 变成了 attention_kernels.cuh,  musify 后也只有attention_kernels.muh
```
v0.4.2: vllm_musa.latest/csrc_musa/attention/attention_kernels.mu
v0.6.6: vllm_musa.latest/csrc_musa/attention/attention_kernels.muh
```
正在解决：

* 【20250126】找不到vllm版本号

解决方法：在 vllm/__init__.py 中增加一行
```
__version__ = "0.6.6"
```

* 【20250126】Vllm requirements 不满足

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
vllm 0.6.6 requires nvidia-ml-py>=12.560.30, which is not installed.
vllm 0.6.6 requires xformers==0.0.28.post3; platform_system == "Linux" and platform_machine == "x86_64", which is not installed.
vllm 0.6.6 requires torch==2.5.1, but you have torch 2.2.0a0+git8ac9b20 which is incompatible.
vllm 0.6.6 requires torchvision==0.20.1, but you have torchvision 0.17.2+c1d70fe which is incompatible.
```
解决方法：编译前环境不需要vllm
