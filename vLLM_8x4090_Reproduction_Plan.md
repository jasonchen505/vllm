# vLLM 8卡4090完整复现Plan

> 基于vLLM源码深度分析，针对8卡4090算力资源制定的全流程复现计划
> 目标：系统性掌握vLLM框架核心功能，积累LLM推理优化实战经验

---

## 一、硬件资源评估

### 1.1 8卡4090规格

| 参数 | 数值 |
|------|------|
| GPU数量 | 8张 |
| 单卡显存 | 24GB GDDR6X |
| 总显存 | 192GB |
| CUDA算力 | 8.9 |
| FP16算力 | 165 TFLOPS |
| 内存带宽 | 1008 GB/s |

### 1.2 可复现的模型规模

| 模型规模 | 参数量 | 显存需求(FP16) | 显存需求(INT8) | 可行性 |
|----------|--------|---------------|---------------|--------|
| 小模型 | 1-3B | 2-6GB | 1-3GB | 单卡即可 |
| 中模型 | 7-8B | 14-16GB | 7-8GB | 单卡可行 |
| 大模型 | 13B | 26GB | 13GB | 需TP=2 |
| 超大模型 | 30-40B | 60-80GB | 30-40GB | 需TP=4 |
| 巨型模型 | 70B | 140GB | 70GB | 需TP=8 |

### 1.3 推荐复现模型

**单卡测试**：
- `facebook/opt-125m` (125M) - 快速验证
- `Qwen/Qwen3-0.6B` (600M) - 轻量测试
- `meta-llama/Llama-3.2-1B` (1B) - 单卡完整测试

**多卡测试**：
- `meta-llama/Llama-3.1-8B` (8B) - TP=1/2
- `Qwen/Qwen3-8B` (8B) - TP=1/2
- `meta-llama/Llama-3.1-70B` (70B) - TP=8（需INT8量化）

---

## 二、复现阶段规划

### 阶段一：环境搭建与基础验证（Day 1-2）

#### 目标
- 搭建vLLM开发环境
- 验证基础推理功能
- 熟悉项目结构

#### 任务清单

**1.1 环境搭建**
```bash
# 创建虚拟环境
conda create -n vllm python=3.12 -y
conda activate vllm

# 安装vLLM（从源码）
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

**1.2 基础推理测试**
```python
# examples/basic/offline_inference/basic.py
from vllm import LLM, SamplingParams

# 测试小模型
llm = LLM(model="facebook/opt-125m")
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Output: {output.outputs[0].text!r}")
```

**1.3 在线服务测试**
```bash
# 启动API服务
vllm serve facebook/opt-125m --port 8000

# 测试API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Hello, my name is",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

**1.4 学习产出**
- [ ] 环境搭建文档
- [ ] 基础推理代码注释
- [ ] 项目结构思维导图

---

### 阶段二：核心功能复现（Day 3-7）

#### 目标
- 复现PagedAttention
- 复现连续批处理
- 复现Prefix Caching
- 复现Chunked Prefill

#### 任务清单

**2.1 PagedAttention深入理解**
```bash
# 阅读源码
vim vllm/v1/core/block_pool.py
vim vllm/v1/core/kv_cache_manager.py
vim vllm/v1/attention/backend.py

# 运行相关测试
pytest tests/v1/core/test_kv_cache_utils.py -v
pytest tests/v1/core/test_single_type_kv_cache_manager.py -v
```

**实验设计**：
```python
# 测试不同block_size的影响
import time
from vllm import LLM, SamplingParams

def test_block_size(block_size, num_requests=100):
    llm = LLM(
        model="facebook/opt-125m",
        block_size=block_size,
        gpu_memory_utilization=0.9
    )
    
    prompts = [f"Test prompt {i}" for i in range(num_requests)]
    sampling_params = SamplingParams(max_tokens=10)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    return elapsed, len(outputs)

# 对比不同block_size
for bs in [8, 16, 32]:
    elapsed, count = test_block_size(bs)
    print(f"Block size {bs}: {elapsed:.2f}s, {count} requests")
```

**2.2 连续批处理复现**
```bash
# 阅读调度器源码
vim vllm/v1/core/sched/scheduler.py

# 运行调度器测试
pytest tests/v1/core/test_scheduler.py -v
pytest tests/v1/core/test_scheduler_e2e.py -v
```

**实验设计**：
```python
# 测试连续批处理 vs 静态批处理
from vllm import LLM, SamplingParams
import time

def test_continuous_batching(max_num_seqs):
    llm = LLM(
        model="facebook/opt-125m",
        max_num_seqs=max_num_seqs,
        enable_chunked_prefill=True
    )
    
    # 模拟不同长度的请求
    prompts = [
        "Short prompt",
        "Medium length prompt " * 10,
        "Long prompt " * 100,
    ] * 20
    
    sampling_params = SamplingParams(max_tokens=50)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    return elapsed, len(outputs)

# 测试不同并发数
for max_seqs in [8, 16, 32, 64]:
    elapsed, count = test_continuous_batching(max_seqs)
    print(f"max_num_seqs={max_seqs}: {elapsed:.2f}s, {count} requests")
```

**2.3 Prefix Caching复现**
```bash
# 阅读前缀缓存源码
vim vllm/v1/core/kv_cache_manager.py
vim tests/v1/core/test_prefix_caching.py

# 运行前缀缓存测试
pytest tests/v1/core/test_prefix_caching.py -v
```

**实验设计**：
```python
# 测试Prefix Caching效果
from vllm import LLM, SamplingParams
import time

def test_prefix_caching(enable_caching, num_rounds=5):
    llm = LLM(
        model="facebook/opt-125m",
        enable_prefix_caching=enable_caching,
        gpu_memory_utilization=0.9
    )
    
    # 模拟多轮对话
    system_prompt = "You are a helpful assistant."
    history = ""
    
    total_time = 0
    for round in range(num_rounds):
        user_input = f"Question {round}: What is AI?"
        prompt = f"{system_prompt}\n{history}\nUser: {user_input}\nAssistant:"
        
        sampling_params = SamplingParams(max_tokens=100)
        
        start = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start
        total_time += elapsed
        
        history += f"\nUser: {user_input}\nAssistant: {outputs[0].outputs[0].text}"
    
    return total_time, num_rounds

# 对比有无Prefix Caching
for enable in [False, True]:
    total_time, rounds = test_prefix_caching(enable)
    print(f"Prefix Caching={enable}: {total_time:.2f}s for {rounds} rounds")
```

**2.4 Chunked Prefill复现**
```bash
# 阅读Chunked Prefill相关代码
vim vllm/v1/core/sched/scheduler.py  # 搜索 long_prefill_token_threshold

# 运行相关测试
pytest tests/v1/core/test_scheduler.py -k "chunked" -v
```

**实验设计**：
```python
# 测试Chunked Prefill效果
from vllm import LLM, SamplingParams
import time

def test_chunked_prefill(enable_chunked, threshold=2048):
    llm = LLM(
        model="facebook/opt-125m",
        enable_chunked_prefill=enable_chunked,
        long_prefill_token_threshold=threshold,
        max_num_batched_tokens=4096
    )
    
    # 长prompt测试
    long_prompt = "This is a very long prompt. " * 500  # ~3000 tokens
    short_prompt = "Short prompt"
    
    prompts = [long_prompt, short_prompt] * 10
    sampling_params = SamplingParams(max_tokens=50)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    return elapsed, len(outputs)

# 对比有无Chunked Prefill
for enable in [False, True]:
    elapsed, count = test_chunked_prefill(enable)
    print(f"Chunked Prefill={enable}: {elapsed:.2f}s, {count} requests")
```

**2.5 学习产出**
- [ ] PagedAttention原理图解
- [ ] 连续批处理流程图
- [ ] Prefix Caching实验报告
- [ ] Chunked Prefill实验报告

---

### 阶段三：高级功能复现（Day 8-14）

#### 目标
- 复现投机解码
- 复现量化功能
- 复现分布式并行
- 复现结构化输出

#### 任务清单

**3.1 投机解码复现**
```bash
# 阅读投机解码源码
vim vllm/v1/spec_decode/eagle.py
vim vllm/v1/spec_decode/ngram_proposer.py
vim tests/v1/spec_decode/test_eagle.py
vim tests/v1/spec_decode/test_ngram.py

# 运行投机解码测试
pytest tests/v1/spec_decode/test_ngram.py -v
```

**实验设计**：
```python
# 测试N-gram投机解码
from vllm import LLM, SamplingParams
import time

def test_speculative_decoding(speculative_model, num_spec_tokens):
    llm = LLM(
        model="facebook/opt-125m",
        speculative_model=speculative_model,
        num_speculative_tokens=num_spec_tokens,
        use_v2_block_manager=True
    )
    
    prompts = [f"Test prompt {i}" for i in range(50)]
    sampling_params = SamplingParams(max_tokens=100)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    return elapsed, len(outputs)

# 测试不同投机策略
configs = [
    (None, 0),  # 无投机解码
    ("ngram", 3),  # N-gram投机
    ("ngram", 5),  # N-gram投机更多token
]

for model, tokens in configs:
    elapsed, count = test_speculative_decoding(model, tokens)
    print(f"Speculative={model}, tokens={tokens}: {elapsed:.2f}s")
```

**3.2 量化功能复现**
```bash
# 阅读量化相关代码
vim vllm/model_executor/layers/quantization/
vim tests/quantization/

# 运行量化测试
pytest tests/quantization/ -v -k "fp8"
```

**实验设计**：
```python
# 测试KV Cache量化
from vllm import LLM, SamplingParams

def test_kv_cache_quant(kv_cache_dtype):
    llm = LLM(
        model="facebook/opt-125m",
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=0.9
    )
    
    prompts = [f"Test prompt {i}" for i in range(100)]
    sampling_params = SamplingParams(max_tokens=50)
    
    outputs = llm.generate(prompts, sampling_params)
    return len(outputs)

# 对比不同KV Cache精度
for dtype in ["auto", "fp8"]:
    count = test_kv_cache_quant(dtype)
    print(f"KV Cache dtype={dtype}: {count} requests completed")
```

**3.3 分布式并行复现**
```bash
# 测试Tensor Parallelism
# 单机多卡测试
CUDA_VISIBLE_DEVICES=0,1 vllm serve facebook/opt-125m \
  --tensor-parallel-size 2 \
  --port 8000

# 测试Pipeline Parallelism
CUDA_VISIBLE_DEVICES=0,1 vllm serve facebook/opt-125m \
  --pipeline-parallel-size 2 \
  --port 8001
```

**实验设计**：
```python
# 测试不同并行策略
import subprocess
import time

def test_parallelism(tp_size, pp_size, num_gpus):
    cmd = f"""
    CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(num_gpus))} \
    vllm serve facebook/opt-125m \
    --tensor-parallel-size {tp_size} \
    --pipeline-parallel-size {pp_size} \
    --port 8000
    """
    
    # 启动服务
    proc = subprocess.Popen(cmd, shell=True)
    time.sleep(30)  # 等待启动
    
    # 测试API
    import requests
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "facebook/opt-125m",
            "prompt": "Hello",
            "max_tokens": 10
        }
    )
    
    proc.terminate()
    return response.status_code

# 测试不同并行配置
configs = [
    (1, 1, 1),  # 单卡
    (2, 1, 2),  # TP=2
    (1, 2, 2),  # PP=2
    (2, 2, 4),  # TP=2, PP=2
]

for tp, pp, gpus in configs:
    status = test_parallelism(tp, pp, gpus)
    print(f"TP={tp}, PP={pp}: Status {status}")
```

**3.4 结构化输出复现**
```bash
# 阅读结构化输出代码
vim vllm/v1/structured_output/
vim tests/v1/structured_output/

# 运行结构化输出测试
pytest tests/v1/structured_output/ -v
```

**实验设计**：
```python
# 测试JSON Schema约束输出
from vllm import LLM, SamplingParams
import json

def test_structured_output():
    llm = LLM(model="facebook/opt-125m")
    
    # 定义JSON Schema
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"}
        },
        "required": ["name", "age", "city"]
    }
    
    prompt = "Generate a person profile:"
    sampling_params = SamplingParams(
        max_tokens=100,
        guided_decoding_backend="xgrammar",
        guided_json=json_schema
    )
    
    outputs = llm.generate([prompt], sampling_params)
    
    # 验证输出是否符合Schema
    output_text = outputs[0].outputs[0].text
    try:
        parsed = json.loads(output_text)
        print(f"Valid JSON: {parsed}")
        return True
    except json.JSONDecodeError:
        print(f"Invalid JSON: {output_text}")
        return False

test_structured_output()
```

**3.5 学习产出**
- [ ] 投机解码实验报告
- [ ] 量化效果对比报告
- [ ] 分布式并行性能测试报告
- [ ] 结构化输出使用指南

---

### 阶段四：RL/后训练集成复现（Day 15-21）

#### 目标
- 复现RLHF集成
- 复现权重热更新
- 复现批量推理优化

#### 任务清单

**4.1 RLHF集成复现**
```bash
# 阅读RLHF示例代码
vim examples/rl/rlhf_ipc.py
vim examples/rl/rlhf_nccl.py

# 运行RLHF示例
python examples/rl/rlhf_ipc.py
```

**实验设计**：
```python
# 简化版RLHF流程
from vllm import LLM, SamplingParams
import torch

def test_rlhf_workflow():
    # 1. 初始化推理引擎
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.7,
        enforce_eager=True
    )
    
    # 2. 生成样本
    prompts = ["Hello, my name is", "The capital of France is"]
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    outputs = llm.generate(prompts, sampling_params)
    
    # 3. 模拟奖励计算
    rewards = [0.8, 0.9]  # 模拟奖励
    
    # 4. 模拟策略更新（实际中会用PPO/GRPO）
    # 这里只是演示流程
    
    # 5. 权重热更新（演示）
    # llm.wake_up()  # 唤醒引擎
    
    return outputs, rewards

outputs, rewards = test_rlhf_workflow()
print(f"Generated {len(outputs)} outputs with rewards {rewards}")
```

**4.2 权重热更新复现**
```bash
# 阅读权重更新相关代码
vim vllm/distributed/weight_transfer/
vim examples/rl/skip_loading_weights_in_engine_init.py
```

**实验设计**：
```python
# 测试权重热更新
from vllm import LLM, SamplingParams
import torch

def test_weight_update():
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.7,
        enforce_eager=True
    )
    
    # 1. 初始推理
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(temperature=0, max_tokens=20)
    outputs1 = llm.generate(prompts, sampling_params)
    print(f"Before update: {outputs1[0].outputs[0].text}")
    
    # 2. 模拟权重更新（实际中会从训练端接收新权重）
    # 这里只是演示sleep/wake流程
    
    # 3. 再次推理验证
    outputs2 = llm.generate(prompts, sampling_params)
    print(f"After update: {outputs2[0].outputs[0].text}")

test_weight_update()
```

**4.3 批量推理优化复现**
```bash
# 阅读benchmark代码
vim benchmarks/benchmark_serving.py
vim benchmarks/benchmark_throughput.py

# 运行benchmark
vllm bench throughput --model facebook/opt-125m --num-prompts 100
```

**实验设计**：
```python
# 测试批量推理性能
from vllm import LLM, SamplingParams
import time

def test_batch_inference(batch_sizes):
    results = {}
    
    for batch_size in batch_sizes:
        llm = LLM(
            model="facebook/opt-125m",
            max_num_seqs=batch_size,
            gpu_memory_utilization=0.9
        )
        
        prompts = [f"Test prompt {i}" for i in range(batch_size)]
        sampling_params = SamplingParams(max_tokens=50)
        
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start
        
        throughput = len(outputs) / elapsed
        results[batch_size] = {
            "time": elapsed,
            "throughput": throughput,
            "requests": len(outputs)
        }
        
        del llm  # 释放显存
    
    return results

# 测试不同batch size
batch_sizes = [8, 16, 32, 64, 128]
results = test_batch_inference(batch_sizes)

for bs, stats in results.items():
    print(f"Batch size {bs}: {stats['time']:.2f}s, "
          f"{stats['throughput']:.2f} req/s")
```

**4.4 学习产出**
- [ ] RLHF集成流程文档
- [ ] 权重热更新使用指南
- [ ] 批量推理性能测试报告
- [ ] 与SGLang对比分析

---

### 阶段五：Benchmark与性能调优（Day 22-28）

#### 目标
- 运行完整benchmark
- 性能调优实践
- 问题定位能力训练

#### 任务清单

**5.1 完整Benchmark运行**
```bash
# 启动服务
vllm serve meta-llama/Llama-3.2-1B \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --port 8000

# 运行serving benchmark
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3.2-1B \
  --dataset-name sharegpt \
  --num-prompts 100

# 运行throughput benchmark
vllm bench throughput \
  --model meta-llama/Llama-3.2-1B \
  --num-prompts 100 \
  --input-len 256 \
  --output-len 256
```

**5.2 性能调优实践**
```python
# 测试不同配置的性能
from vllm import LLM, SamplingParams
import time

def benchmark_config(config_name, **kwargs):
    llm = LLM(model="facebook/opt-125m", **kwargs)
    
    prompts = [f"Test prompt {i}" for i in range(100)]
    sampling_params = SamplingParams(max_tokens=50)
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    
    throughput = len(outputs) / elapsed
    print(f"{config_name}: {elapsed:.2f}s, {throughput:.2f} req/s")
    
    del llm

# 测试不同配置
configs = [
    ("Baseline", {}),
    ("Prefix Caching", {"enable_prefix_caching": True}),
    ("Chunked Prefill", {"enable_chunked_prefill": True}),
    ("Combined", {
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True
    }),
]

for name, kwargs in configs:
    benchmark_config(name, **kwargs)
```

**5.3 问题定位训练**
```bash
# 模拟常见问题场景

# 1. OOM问题
vllm serve meta-llama/Llama-3.2-1B \
  --gpu-memory-utilization 0.99 \
  --max-num-seqs 1000

# 2. 延迟问题
vllm serve meta-llama/Llama-3.2-1B \
  --max-num-batched-tokens 32768

# 3. 缓存命中率低
vllm serve meta-llama/Llama-3.2-1B \
  --enable-prefix-caching \
  --block-size 128
```

**5.4 学习产出**
- [ ] 完整benchmark报告
- [ ] 性能调优指南
- [ ] 问题定位手册
- [ ] 最佳实践总结

---

## 三、关键实验指标

### 3.1 性能指标

| 指标 | 说明 | 测量方法 |
|------|------|---------|
| TTFT | 首token延迟 | `time_to_first_token` |
| TPOT | 每token延迟 | `time_per_output_token` |
| Throughput | 吞吐量(req/s) | `num_requests / elapsed_time` |
| GPU Util | GPU利用率 | `nvidia-smi` |
| Cache Hit | 缓存命中率 | `prefix_cache_hits` |

### 3.2 实验对比表

| 实验 | 配置 | TTFT | TPOT | Throughput | 备注 |
|------|------|------|------|------------|------|
| 基线 | 默认配置 | - | - | - | 对照组 |
| +Prefix Cache | enable_prefix_caching=True | - | - | - | 预期降低TTFT |
| +Chunked Prefill | enable_chunked_prefill=True | - | - | - | 预期提高吞吐 |
| +投机解码 | speculative_model=ngram | - | - | - | 预期提高吞吐 |
| 综合优化 | 全部启用 | - | - | - | 最优配置 |

---

## 四、学习资源

### 4.1 必读文档

1. **架构文档**
   - `docs/design/arch_overview.md`
   - `docs/design/paged_attention.md`
   - `docs/design/prefix_caching.md`

2. **配置文档**
   - `docs/configuration/`
   - `docs/features/`

3. **API文档**
   - `docs/api/`

### 4.2 关键源码文件

```
vllm/v1/engine/core.py              # 引擎核心
vllm/v1/core/sched/scheduler.py     # 调度器
vllm/v1/core/kv_cache_manager.py    # KV缓存管理
vllm/v1/core/block_pool.py          # 块池管理
vllm/v1/attention/backend.py        # 注意力后端
vllm/v1/worker/gpu_model_runner.py  # GPU模型运行器
vllm/v1/spec_decode/                # 投机解码
```

### 4.3 测试用例

```
tests/v1/core/test_prefix_caching.py  # 前缀缓存测试
tests/v1/core/test_scheduler.py       # 调度器测试
tests/v1/spec_decode/                 # 投机解码测试
tests/quantization/                   # 量化测试
```

---

## 五、时间规划

| 阶段 | 时间 | 主要任务 | 产出 |
|------|------|---------|------|
| 阶段一 | Day 1-2 | 环境搭建、基础验证 | 环境文档、基础代码 |
| 阶段二 | Day 3-7 | 核心功能复现 | 实验报告、原理图解 |
| 阶段三 | Day 8-14 | 高级功能复现 | 性能测试报告 |
| 阶段四 | Day 15-21 | RL/后训练集成 | 集成文档、使用指南 |
| 阶段五 | Day 22-28 | Benchmark与调优 | 完整benchmark报告 |

---

## 六、风险与应对

### 6.1 潜在风险

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 显存不足 | 无法运行大模型 | 使用量化、减小batch size |
| 编译失败 | 环境搭建失败 | 使用预编译wheel、Docker |
| 测试超时 | 实验进度延迟 | 优先核心测试、并行执行 |
| 网络问题 | 模型下载失败 | 使用镜像、离线模式 |

### 6.2 备选方案

- **显存不足**：使用INT8/INT4量化
- **编译失败**：使用Docker镜像
- **时间不足**：优先核心功能，跳过边缘场景

---

## 七、验收标准

### 7.1 功能验收

- [ ] 能独立运行vLLM基础推理
- [ ] 能配置并启动在线服务
- [ ] 能运行benchmark并分析结果
- [ ] 能定位并解决常见问题

### 7.2 知识验收

- [ ] 能解释PagedAttention原理
- [ ] 能解释连续批处理流程
- [ ] 能解释Prefix Caching机制
- [ ] 能解释Chunked Prefill优势

### 7.3 实战验收

- [ ] 能独立完成性能调优
- [ ] 能独立定位OOM问题
- [ ] 能独立定位延迟问题
- [ ] 能独立完成RLHF集成

---

**最后更新**：2026-06-27

**适用场景**：8卡4090算力资源，vLLM框架全流程复现
