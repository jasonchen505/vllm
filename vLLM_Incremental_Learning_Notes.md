# vLLM 增量学习点记录

> 基于深入探索vLLM源码后，对比之前两轮分析增量新学习到的关键点
> 记录在制定8卡4090复现Plan过程中发现的新细节

---

## 一、vLLM V1架构新特性

### 1.1 EngineCore异步调度机制

**新发现**：vLLM V1支持真正的异步调度，通过`AsyncScheduler`实现。

```python
# vllm/config/scheduler.py
def get_scheduler_cls(self) -> type["SchedulerInterface"]:
    if self.scheduler_cls is None:
        if self.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler
            return AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        return Scheduler
```

**关键设计**：
- 异步调度允许CPU调度与GPU计算重叠
- 通过`async_scheduling`配置启用
- 支持Pipeline Parallelism的多步in-flight调度

### 1.2 Batch Queue机制

**新发现**：vLLM支持批量队列，可以预调度多个batch。

```python
# vllm/v1/engine/core.py
def step_with_batch_queue(self):
    """Schedule and execute batches with the batch queue.
    
    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    2. If there is no new scheduled batch, block until the first batch
       in the job queue is finished.
    3. Update the scheduler from the output.
    """
    batch_queue = self.batch_queue
    assert batch_queue is not None
    
    # Try to schedule a new batch if the batch queue is not full
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
        
        # Add this step's future to the queue
        batch_queue.appendleft((future, scheduler_output, exec_future))
        if len(batch_queue) < self.batch_queue_size:
            return None, model_executed  # Don't block
```

**关键设计**：
- `batch_queue_size`控制预调度的batch数量
- 优先填充队列，再获取结果
- 减少GPU空闲时间

### 1.3 Deferred Block Free

**新发现**：vLLM支持延迟释放块，用于异步KV缓存传输。

```python
# vllm/v1/engine/core.py
# Every GPU write enqueued by this and earlier steps has completed, so it is
# safe to return deferred-free blocks to the pool.
if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
    self.processed_step_seq += 1
    self._drain_deferred_frees()
```

**关键设计**：
- 在异步KV缓存传输场景下，块释放需要延迟
- 确保GPU写操作完成后再释放
- 避免数据竞争

### 1.4 KV Connector用于P/D分离

**新发现**：vLLM通过KV Connector实现Prefill-Decode分离架构。

```python
# vllm/v1/core/sched/scheduler.py
# KV Connector pushes/pull of remote KVs for P/D and offloading.
self.connector = None
kv_transfer_config = self.vllm_config.kv_transfer_config
if kv_transfer_config is not None:
    self.connector = KVConnectorFactory.create_connector(
        config=self.vllm_config,
        role=KVConnectorRole.SCHEDULER,
        kv_cache_config=self.kv_cache_config,
    )
```

**支持的传输后端**：
- NIXL：高性能RDMA传输
- Mooncake：分布式存储
- gRPC：跨节点传输

---

## 二、调度器深度细节

### 2.1 Token级调度的精确实现

**新发现**：调度器以token粒度工作，而不是请求粒度。

```python
# vllm/v1/core/sched/scheduler.py
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

**关键设计**：
- 每个请求维护`num_computed_tokens`和`num_tokens_with_spec`
- 调度器尝试分配token使两者对齐
- 自然支持chunked prefill、prefix caching、speculative decoding

### 2.2 两阶段调度的详细流程

**新发现**：调度器分为两个阶段，先调度running请求，再调度waiting请求。

```python
# vllm/v1/core/sched/scheduler.py
# First, schedule the RUNNING requests.
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)
    
    # 分配KV缓存块
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is None:
        # 内存不足，抢占低优先级请求
        preempted_req = self.running.pop()
        self._preempt_request(preempted_req)
        continue
    
    token_budget -= num_new_tokens

# Next, schedule the WAITING requests.
while self.waiting and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
    
    request = self.waiting.peek_request()
    # 计算前缀缓存命中
    computed_blocks, num_computed = self.kv_cache_manager.get_computed_blocks(request)
    # 分配新块
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is None:
        break
    
    # 从 waiting 移到 running
    self.running.append(request)
```

### 2.3 抢占机制的实现细节

**新发现**：vLLM支持两种抢占策略：FCFS和Priority。

```python
# vllm/v1/core/sched/scheduler.py
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),
    )
    self.running.remove(preempted_req)
else:
    preempted_req = self.running.pop()
```

**抢占流程**：
1. 选择最低优先级的请求
2. 释放其KV缓存块
3. 将请求移回waiting队列
4. 更新请求状态为PREEMPTED

### 2.4 Watermark机制

**新发现**：vLLM通过Watermark机制避免频繁抢占。

```python
# vllm/config/scheduler.py
watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
"""Fraction of total KV cache blocks to keep free (the watermark) when
admitting waiting or preempted requests into the running queue."""

# vllm/v1/core/kv_cache_manager.py
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)

# 在分配时检查
available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
required_blocks = num_blocks_to_allocate + watermark_blocks
if required_blocks > available_blocks:
    return None  # 无法分配
```

**关键设计**：
- Watermark是KV缓存块的预留比例
- 在调度waiting/preempted请求时检查
- 避免频繁抢占导致的性能抖动

---

## 三、KV Cache管理深度细节

### 3.1 BlockPool的LRU驱逐实现

**新发现**：BlockPool使用双向链表实现O(1)驱逐。

```python
# vllm/v1/core/block_pool.py
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        # 所有kv-cache块
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # 空闲块队列（双向链表）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 前缀缓存：block_hash -> block
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

**关键设计**：
- `KVCacheBlock`内嵌双向链表指针
- `FreeKVCacheBlockQueue`维护LRU顺序
- 驱逐时从队头弹出最久未使用的块

### 3.2 前缀缓存的块哈希实现

**新发现**：vLLM使用层级哈希实现前缀缓存。

```python
# vllm/v1/core/kv_cache_utils.py
@dataclass
class BlockHash:
    token_ids: tuple[int, ...]   # 块中的token IDs
    extra_keys: tuple            # 额外键（如LoRA ID、mm hash）

def get_block_hash(
    block_tokens: tuple[int, ...],
    parent_block_hash: int | None,
    extra_keys: tuple = (),
) -> BlockHash:
    """计算块哈希，包含父块哈希"""
    return BlockHash(
        token_ids=block_tokens,
        extra_keys=extra_keys,
    )
```

**关键设计**：
- 每个块的哈希包含父块的哈希
- 确保前缀的唯一性
- 支持LoRA、多模态等额外键

### 3.3 HybridKVCacheCoordinator

**新发现**：vLLM使用协调器管理不同类型的KV缓存。

```python
# vllm/v1/core/kv_cache_coordinator.py
class HybridKVCacheCoordinator:
    """协调不同类型注意力层的KV缓存"""
    
    def __init__(self, kv_cache_config, ...):
        # 不同类型的KV缓存组
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        
    def get_num_blocks_to_allocate(self, request_id, num_tokens, ...):
        """计算需要分配的块数"""
        # 考虑不同组的block_size
        pass
```

**支持的KV缓存类型**：
- FullAttentionSpec：全注意力
- SlidingWindowSpec：滑动窗口注意力
- MambaSpec：Mamba SSM层
- MLAAttentionSpec：DeepSeek MLA

### 3.4 Watermark的精确控制

**新发现**：Watermark只在调度waiting/preempted请求时生效。

```python
# vllm/v1/core/kv_cache_manager.py
watermark_blocks = 0
# The watermark is applied to waiting/preempted requests only, and only
# when there's at least one request already scheduled.
if has_scheduled_reqs and request.status in (
    RequestStatus.WAITING,
    RequestStatus.PREEMPTED,
):
    watermark_blocks = self.watermark_blocks
```

**关键设计**：
- 只在有请求已调度时应用watermark
- 只对waiting/preempted请求生效
- 避免对running请求的干扰

---

## 四、RL/后训练集成细节

### 4.1 IPC权重传输机制

**新发现**：vLLM通过CUDA IPC实现高效的权重传输。

```python
# examples/rl/rlhf_ipc.py
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

class TrainModel:
    def broadcast_weights(self, llm_handle, packed=False):
        """Broadcast weights to the inference engine using IPC."""
        trainer_args = IPCTrainerSendWeightsArgs(
            send_mode="ray", llm_handle=llm_handle, packed=packed
        )
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=self.train_model.named_parameters(),
            trainer_args=trainer_args,
        )
```

**关键设计**：
- 使用CUDA IPC handle传输权重
- 支持Ray分布式框架
- 支持packed/chunked传输模式

### 4.2 Sleep/Wake机制

**新发现**：vLLM支持多级休眠以释放GPU内存。

```python
# vllm/v1/engine/core.py
def sleep(self, level: int = 1):
    """休眠引擎，释放GPU内存
    
    Level 0: 暂停调度，不改变GPU内存
    Level 1: 将模型权重卸载到CPU，丢弃KV缓存
    Level 2: 丢弃所有GPU内存
    """
    pass

def wake_up(self, tags=None):
    """唤醒引擎，恢复GPU内存"""
    pass
```

**使用场景**：
- RL训练时，推理引擎休眠释放内存
- 训练完成后，唤醒引擎继续推理
- 支持增量唤醒（只恢复部分功能）

### 4.3 Weight Transfer Engine

**新发现**：vLLM支持多种权重传输后端。

```python
# vllm/config.py
@dataclass
class WeightTransferConfig:
    backend: str = "ipc"  # ipc, nccl, http
    # IPC后端：使用CUDA IPC handle
    # NCCL后端：使用NCCL集合通信
    # HTTP后端：使用HTTP传输
```

**支持的后端**：
- IPC：单机多卡，低延迟
- NCCL：多机多卡，高吞吐
- HTTP：跨节点，灵活部署

---

## 五、与SGLang的对比新发现

### 5.1 缓存管理对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **数据结构** | 块表+块池 | Radix Tree |
| **哈希方式** | 层级哈希（父块+当前块） | 前缀匹配 |
| **驱逐策略** | LRU双向链表 | LRU/LFU/SLRU |
| **缓存粒度** | 块级别（16/32 tokens） | 前缀级别（任意长度） |

### 5.2 调度策略对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **调度粒度** | Token级 | Token级 |
| **调度策略** | FCFS/Priority | LPM/DFS-Weight/FCFS |
| **抢占机制** | 支持（FCFS/Priority） | 支持（Retraction） |
| **Watermark** | 支持 | 不支持 |

### 5.3 注意力后端对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **主力后端** | FlashAttention | FlashInfer |
| **后端数量** | 10+ | 40+ |
| **MLA支持** | 专用MLA后端 | 多种MLA后端 |
| **CUDA Graph** | 原生支持 | 原生支持 |

### 5.4 RL集成对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| **权重传输** | IPC/NCCL/HTTP | 内置API |
| **Sleep/Wake** | 支持（多级） | 不支持 |
| **RL框架集成** | 需要自定义 | 原生支持AReaL/Miles等 |
| **批量推理** | 支持 | 支持 |

---

## 六、8卡4090复现的关键发现

### 6.1 显存优化策略

**新发现**：4090的24GB显存需要精细优化。

```python
# 推荐配置
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    gpu_memory_utilization=0.9,  # 使用90%显存
    max_num_seqs=64,  # 控制并发数
    enable_chunked_prefill=True,  # 启用分块预填充
    long_prefill_token_threshold=2048,  # 长prompt阈值
    block_size=16,  # 使用16的块大小
)
```

### 6.2 并行策略选择

**新发现**：4090支持NVLink，但带宽有限。

```python
# 单卡测试
CUDA_VISIBLE_DEVICES=0 vllm serve facebook/opt-125m --port 8000

# 双卡TP测试
CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.2-1B \
  --tensor-parallel-size 2 --port 8000

# 8卡TP测试（70B模型需要INT8量化）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve meta-llama/Llama-3.1-70B \
  --tensor-parallel-size 8 \
  --quantization awq \
  --port 8000
```

### 6.3 Benchmark运行建议

**新发现**：4090的benchmark需要调整参数。

```bash
# 小模型benchmark
vllm bench throughput \
  --model facebook/opt-125m \
  --num-prompts 100 \
  --input-len 128 \
  --output-len 128

# 中模型benchmark
vllm bench throughput \
  --model meta-llama/Llama-3.2-1B \
  --num-prompts 50 \
  --input-len 256 \
  --output-len 256

# Serving benchmark
vllm bench serve \
  --backend openai \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3.2-1B \
  --dataset-name sharegpt \
  --num-prompts 50
```

---

## 七、关键源码路径速查（增量版）

### 7.1 核心引擎

```
vllm/v1/engine/core.py              # EngineCore核心引擎
  - step()                          # 主循环
  - step_with_batch_queue()         # 批量队列模式
  - sleep()/wake_up()               # 休眠/唤醒
  - _initialize_kv_caches()         # KV缓存初始化
```

### 7.2 调度器

```
vllm/v1/core/sched/scheduler.py     # 调度器实现
  - schedule()                      # 主调度函数
  - _preempt_request()              # 抢占请求
  - update_from_output()            # 更新状态
  - get_grammar_bitmask()           # 结构化输出
```

### 7.3 KV缓存管理

```
vllm/v1/core/kv_cache_manager.py    # KV缓存管理器
  - allocate_slots()                # 分配槽位
  - free()                          # 释放块
  - get_computed_blocks()           # 获取已计算块
  - cache_blocks()                  # 缓存块

vllm/v1/core/block_pool.py          # 块池管理
  - BlockPool                       # 块池
  - FreeKVCacheBlockQueue           # 空闲队列
  - BlockHashToBlockMap             # 哈希映射
```

### 7.4 注意力后端

```
vllm/v1/attention/backend.py        # 注意力后端抽象
  - AttentionBackend                # 后端基类
  - AttentionMetadataBuilder        # 元数据构建器
  - CommonAttentionMetadata         # 通用元数据

vllm/v1/attention/backends/         # 具体后端实现
  - flash_attn.py                   # FlashAttention
  - flashinfer.py                   # FlashInfer
  - triton_attn.py                  # Triton
```

### 7.5 投机解码

```
vllm/v1/spec_decode/                # 投机解码
  - eagle.py                        # EAGLE
  - ngram_proposer.py               # N-gram
  - draft_model.py                  # 独立草稿模型
  - dynamic/                        # 动态投机
```

### 7.6 分布式

```
vllm/distributed/                   # 分布式
  - parallel_state.py               # 并行状态
  - kv_transfer/                    # KV传输
  - weight_transfer/                # 权重传输
  - eplb/                           # 专家负载均衡
```

---

## 八、实验验证新方法

### 8.1 调度器单元测试

```bash
# 运行调度器测试
pytest tests/v1/core/test_scheduler.py -v

# 运行前缀缓存测试
pytest tests/v1/core/test_prefix_caching.py -v

# 运行KV缓存测试
pytest tests/v1/core/test_kv_cache_utils.py -v
```

### 8.2 性能分析工具

```python
# 使用vLLM内置的性能分析
from vllm.profiler import profile

with profile("my_profile"):
    outputs = llm.generate(prompts, sampling_params)
```

### 8.3 内存分析

```bash
# 监控GPU内存
nvidia-smi -l 1

# 使用vLLM的内存分析
VLLM_LOGGING_LEVEL=DEBUG vllm serve facebook/opt-125m
```

---

## 九、待深入探索的点

### 9.1 待探索

- [ ] AsyncScheduler的完整实现
- [ ] Batch Queue的性能影响
- [ ] KV Connector的NIXL后端
- [ ] Dynamic Speculative Decoding
- [ ] EAGLE3的实现细节

### 9.2 待实验

- [ ] 不同block_size的性能对比
- [ ] 不同watermark值的影响
- [ ] 不同max_num_seqs的吞吐量
- [ ] Prefix Caching在多轮对话中的效果
- [ ] 投机解码在不同任务上的接受率

### 9.3 待对比

- [ ] vLLM vs SGLang的吞吐量对比
- [ ] vLLM vs SGLang的延迟对比
- [ ] vLLM vs SGLang的显存使用对比
- [ ] vLLM vs SGLang的RL集成易用性

---

**最后更新**：2026-06-27

**记录场景**：制定8卡4090复现Plan过程中的增量学习
