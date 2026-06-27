# LLM 算法实习面试准备指南

> 基于 vLLM + SGLang 源码深度分析，针对 LLM & Agent 应用及后训练方向
> 适用对象：找 LLM 算法实习的 MS 在读学生

---

## 目录

- [第一部分：框架架构深度理解](#第一部分框架架构深度理解)
- [第二部分：核心技术原理与面试考察点](#第二部分核心技术原理与面试考察点)
- [第三部分：LLM 推理优化关键知识点](#第三部分llm-推理优化关键知识点)
- [第四部分：RL/后训练核心知识点](#第四部分rl后训练核心知识点)
- [第五部分：Agent 应用核心知识点](#第五部分agent-应用核心知识点)
- [第六部分：面试深挖问题与回答模板](#第六部分面试深挖问题与回答模板)
- [第七部分：vLLM vs SGLang 架构对比](#第七部分vllm-vs-sglang-架构对比)
- [附录：关键代码路径速查表](#附录关键代码路径速查表)

---

## 第一部分：框架架构深度理解

### 1.1 vLLM 整体架构

#### 分层设计

```
用户接口层:   LLM / AsyncLLM
     ↓
前端处理层:   LLMEngine (InputProcessor + OutputProcessor + EngineCoreClient)
     ↓
核心调度层:   EngineCore (Scheduler + KVCacheManager)
     ↓
执行层:      Executor → Worker → GPUModelRunner → Model
```

**核心类层次**:

| 层级 | 文件路径 | 核心类 |
|------|---------|--------|
| 用户接口 | `vllm/v1/engine/async_llm.py` | `AsyncLLM` |
| 前端处理 | `vllm/v1/engine/llm_engine.py` | `LLMEngine` |
| 核心引擎 | `vllm/v1/engine/core.py` | `EngineCore`, `EngineCoreProc` |
| 调度器 | `vllm/v1/core/sched/scheduler.py` | `Scheduler` |
| KV缓存管理 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| 执行器 | `vllm/v1/executor/abstract.py` | `Executor` |
| Worker | `vllm/v1/worker/gpu_worker.py` | `GPUWorker` |
| 模型运行器 | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner` |

#### 多进程架构

vLLM V1 采用多进程架构，通过 ZMQ 消息队列实现异步通信：

- **API Server 进程**: 处理 HTTP 请求，tokenization
- **Engine Core 进程**: 运行调度器，管理 KV 缓存
- **GPU Worker 进程**: 每个 GPU 一个，执行模型前向
- **DP Coordinator 进程**: 数据并行时的负载均衡

**关键设计优势**: ZMQ IO 可与 GPU 计算重叠，释放 GIL，提高吞吐。

#### EngineCore 主循环

```python
def step(self):
    # 1. 调度器产出 SchedulerOutput
    scheduler_output = self.scheduler.schedule()
    # 2. 执行器异步执行模型前向
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    # 3. 获取 grammar bitmask（结构化输出）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    # 4. 等待模型输出
    model_output = future.result()
    # 5. 采样
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
    # 6. 调度器更新状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)
    return engine_core_outputs
```

### 1.2 SGLang 整体架构

#### 分层设计

```
┌─────────────────────────────────────────────────────────┐
│                   Frontend DSL Layer                     │
│   SglFunction, SglGen, SglSelect, StreamExecutor        │
├─────────────────────────────────────────────────────────┤
│                   API & Entrypoint Layer                 │
│   Engine, HTTP Server, OpenAI API, TokenizerManager     │
├─────────────────────────────────────────────────────────┤
│                   Core Runtime Layer                     │
│   ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│   │  Scheduler   │ │ ModelRunner  │ │   KV Cache      │ │
│   │  (managers/) │ │(model_exec/) │ │  (mem_cache/)   │ │
│   └─────────────┘ └──────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

**核心组件**:

| 模块 | 职责 |
|------|------|
| `managers/` | 调度器、Tokenizer管理、请求处理 |
| `layers/` | 注意力层、线性层、量化层、MoE |
| `models/` | 200+ 模型实现 |
| `mem_cache/` | KV Cache 管理、Radix Tree |
| `model_executor/` | 模型执行器、CUDA Graph |
| `speculative/` | 投机解码算法 |
| `constrained/` | 结构化输出 |
| `disaggregation/` | Prefill-Decode 分离 |

#### 调度器设计

SGLang 的调度器通过多个 Mixin 类组合实现功能扩展：

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,   # Prefill-Decode 分离
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,              # 多路复用
    SchedulerPPMixin,                     # 流水线并行
    SchedulerDllmMixin,                   # 扩散 LLM
    SchedulerMlxOverlapMixin,             # MLX 后端重叠
):
```

**调度策略**:
- **Cache-Aware**: LPM (最长前缀匹配)、DFS-Weight
- **Cache-Agnostic**: FCFS、LOF、Random

---

## 第二部分：核心技术原理与面试考察点

### 2.1 PagedAttention (vLLM 核心创新)

#### 解决什么问题？

**问题背景**: LLM 推理中，KV Cache 的内存管理面临严重挑战：
- 不同请求的序列长度不同，导致内存碎片
- 传统方法预分配最大长度内存，浪费严重
- 无法在请求间共享 KV Cache

**解决方案**: 借鉴操作系统虚拟内存分页机制，将 KV Cache 划分为固定大小的"块"（Block），通过"块表"（Block Table）映射。

#### 核心数据结构

```python
# KV Cache 块规格
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE

# KV Cache 块池
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

#### 内存布局

**FlashAttention 后端**:
```
(num_blocks, 2, block_size, num_kv_heads, head_size)
```
- `num_blocks`: 物理块数
- `2`: K 和 V 合并存储
- `block_size`: 每块的 token 数（必须是 16 的倍数）

#### 面试考察点

**Q: PagedAttention 的核心优势是什么？**

A: 
1. **零内存浪费**: 按需分配，避免预分配大块连续内存
2. **内存共享**: 块可在请求间共享，实现前缀缓存
3. **减少碎片**: 固定大小块消除外部碎片
4. **高效驱逐**: LRU 双向链表实现 O(1) 驱逐

**Q: PagedAttention 的局限性是什么？**

A:
1. **块大小限制**: block_size 必须是 16 的倍数，可能有内部碎片
2. **元数据开销**: 每个块需要维护 hash、ref_cnt 等元数据
3. **驱逐策略**: LRU 不一定是最优策略，某些场景下 LFU 更好
4. **并发控制**: 引用计数在高并发下可能成为瓶颈

### 2.2 RadixAttention (SGLang 核心创新)

#### 解决什么问题？

**问题背景**: 在多轮对话、Agent 工作流、system prompt 复用等场景下，多个请求共享相同的前缀。传统方法每次都要重新计算这些共享前缀的 KV Cache。

**解决方案**: 使用 Radix Tree（基数树）数据结构管理 KV Cache，实现前缀级别的缓存复用。

#### 核心数据结构

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)  # 子节点字典
        self.parent: TreeNode = None           # 父节点
        self.key: RadixKey = None              # 存储的 token 片段
        self.value: Optional[torch.Tensor] = None  # KV Cache 索引
        self.lock_ref = 0                      # 引用计数(防止被驱逐)
        self.last_access_time = time.monotonic()  # 最后访问时间(LRU)
        self.hit_count = 0                     # 命中计数

class RadixKey:
    """支持两种模式：
    - 普通模式: token_ids 直接作为键
    - Bigram 模式: 用于 EAGLE 投机解码
    """
    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")
```

#### 前缀匹配算法

使用**指数搜索 + 二分查找**高效定位第一个不同的 token：

```python
def match(self, other: RadixKey, page_size: int = 1) -> int:
    # 指数搜索: 倍增窗口快速定位分歧点
    lo, step = 0, 1
    while lo < n:
        hi = min(lo + step, n)
        if t0[lo:hi] != t1[lo:hi]:
            # 二分查找精确定位
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if t0[lo:mid] == t1[lo:mid]:
                    lo = mid
                else:
                    hi = mid
            matched_tokens = lo
            break
        lo = hi
        step *= 2
```

#### 面试考察点

**Q: RadixAttention vs PagedAttention 的核心区别？**

A:
| 特性 | RadixAttention | PagedAttention |
|------|---------------|----------------|
| **数据结构** | Radix Tree | 块表 + 块池 |
| **缓存粒度** | 前缀级别 | 块级别 |
| **匹配方式** | 自动前缀匹配 | 手动管理 |
| **驱逐策略** | LRU/LFU/SLRU | LRU |
| **适用场景** | 多轮对话、Agent | 通用场景 |

**Q: RadixAttention 的局限性？**

A:
1. **内存开销**: 树结构本身有元数据开销
2. **驱逐策略复杂**: LRU 不一定最优
3. **前缀匹配开销**: 超长前缀场景下仍有开销
4. **并发控制**: lock_ref 在高并发下可能成为瓶颈

### 2.3 连续批处理（Continuous Batching）

#### 解决什么问题？

**问题背景**: 传统静态批处理需要等待整个 batch 完成才能处理下一个请求，导致：
- 短请求等待长请求完成，延迟高
- GPU 利用率低，有空闲时间
- 无法处理动态到达的请求

**解决方案**: 每个 iteration 动态决定哪些请求进入 prefill/decode，请求完成后立即释放资源。

#### vLLM 实现

vLLM 的调度器以 **token 粒度**工作：

```python
# vllm/v1/core/sched/scheduler.py
def schedule(self, throttle_prefills=False) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens
    
    # 第一阶段：调度正在运行的请求
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        
        # 分配 KV 缓存块
        new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
        if new_blocks is None:
            # 内存不足，抢占低优先级请求
            preempted_req = self.running.pop()
            self._preempt_request(preempted_req)
            continue
        
        token_budget -= num_new_tokens
    
    # 第二阶段：调度等待中的新请求
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

#### SGLang 实现

SGLang 的调度器支持 Cache-Aware 策略：

```python
# python/sglang/srt/managers/schedule_policy.py
class CacheAwarePolicy(Enum):
    LPM = "lpm"           # 最长前缀匹配
    DFS_WEIGHT = "dfs-weight"  # DFS 加权

class CacheAgnosticPolicy(Enum):
    FCFS = "fcfs"         # 先到先服务
    LOF = "lof"           # 最长输出优先
    RANDOM = "random"     # 随机
```

#### 面试考察点

**Q: 连续批处理的局限性？**

A:
1. **调度开销**: 每个 iteration 都需要调度决策，CPU 开销增加
2. **内存碎片**: 动态增删请求可能导致内存碎片
3. **延迟波动**: batch size 动态变化导致延迟不稳定
4. **公平性问题**: 长请求可能被饿死

**Q: 如何改进连续批处理？**

A:
1. **Overlap Scheduling**: CPU 调度与 GPU 计算重叠
2. **Chunked Prefill**: 长 prompt 分块处理
3. **Retraction 机制**: 内存不足时回退低优先级请求
4. **优先级调度**: 支持请求级优先级

### 2.4 Chunked Prefill（分块预填充）

#### 解决什么问题？

**问题背景**: 当 prompt 很长时，prefill 阶段会独占整个批次的计算预算，导致：
- decode 请求被阻塞，延迟增加
- GPU 利用率不均衡

**解决方案**: 将长 prompt 分成多个 chunk 处理，每个 chunk 可以与 decode 请求混合调度。

#### vLLM 实现

```python
# vllm/v1/core/sched/scheduler.py
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

#### SGLang 实现

```python
# python/sglang/srt/managers/scheduler.py
def init_chunked_prefill(self):
    if self.server_args.chunked_prefill_size is not None:
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
    else:
        # 默认值
        self.chunked_prefill_size = 8192
```

#### 面试考察点

**Q: Chunked Prefill 的优势？**

A:
1. **降低首 token 延迟 (TTFT)**: 长 prompt 不会阻塞 decode 请求
2. **提高 GPU 利用率**: prefill 和 decode 可以混合调度
3. **更好的公平性**: 长 prompt 不会独占批次

**Q: Chunked Prefill 的挑战？**

A:
1. **额外开销**: 需要多次调度同一个请求
2. **状态管理**: 需要跟踪每个 chunk 的进度
3. **内存管理**: 需要为 partial prefill 分配临时存储

### 2.5 前缀缓存（Prefix Caching）

#### 解决什么问题？

**问题背景**: 在多轮对话、Agent 工作流等场景中，多个请求共享相同的前缀。传统方法每次都要重新计算这些共享前缀的 KV Cache。

**解决方案**: 缓存已计算的 KV Cache 块，当新请求有相同前缀时直接复用。

#### vLLM 实现

vLLM 使用**块哈希**实现前缀缓存：

```python
# vllm/v1/core/kv_cache_manager.py
@dataclass
class BlockHash:
    token_ids: tuple[int, ...]   # 块中的 token IDs
    extra_keys: tuple            # 额外键（如 LoRA ID、mm hash）
```

**缓存驱逐**: 采用 LRU（最近最少使用）策略，通过 `FreeKVCacheBlockQueue`（双向链表）实现。

#### SGLang 实现

SGLang 使用 **Radix Tree** 实现前缀缓存：

```python
# python/sglang/srt/mem_cache/radix_cache.py
class RadixCache:
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        # 1. 转换为 bigram 视图(如果是 EAGLE)
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        # 2. 页对齐
        key = key.page_aligned(self.page_size)
        # 3. 递归匹配
        value, last_node = self._match_prefix_helper(self.root_node, key)
        # 4. 返回匹配结果
        return MatchResult(device_indices=..., last_device_node=last_node, ...)
```

#### 面试考察点

**Q: 前缀缓存适合什么场景？**

A:
| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 多轮对话 | system prompt + 历史对话共享 | TTFT 降低 50-70% |
| Agent 工作流 | system prompt + 工具定义共享 | 吞吐提升 2-3x |
| 批量推理 | 相同 system prompt 的请求 | 吞吐提升 3-5x |
| RAG 应用 | 相同 context 的请求 | TTFT 降低 40-60% |

**Q: 前缀缓存不适合什么场景？**

A:
| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 纯生成任务 | 无共享前缀 | 禁用缓存减少开销 |
| 高度个性化 | 每个请求前缀不同 | 使用 cache_salt 隔离 |
| 实时性要求极高 | 缓存匹配有开销 | 禁用缓存，使用 FCFS |

---

## 第三部分：LLM 推理优化关键知识点

### 3.1 投机解码（Speculative Decoding）

#### 解决什么问题？

**问题背景**: LLM 的 decode 阶段是 memory-bound 的，每步只能生成一个 token，效率低。

**解决方案**: 使用小模型（或轻量级方法）快速生成多个候选 token，大模型并行验证。

#### vLLM 支持的策略

```python
# vllm/v1/spec_decode/
| 策略 | 文件 | 说明 |
|------|------|------|
| EAGLE | eagle.py | EAGLE 投机解码 |
| Draft Model | draft_model.py | 独立草稿模型 |
| Medusa | medusa.py | Medusa 多头预测 |
| N-gram | ngram_proposer.py | N-gram 预测 |
| DFlash | dflash.py | DFlash 填充式解码 |
| Suffix Decoding | suffix_decoding.py | 后缀解码 |
```

#### SGLang 支持的策略

```python
# python/sglang/srt/speculative/spec_info.py
class SpeculativeAlgorithm(Enum):
    DFLASH = auto()          # DFlash - 下一代投机解码
    EAGLE = auto()           # EAGLE 算法
    EAGLE3 = auto()          # EAGLE3 变体
    FROZEN_KV_MTP = auto()   # 冻结 KV MTP
    STANDALONE = auto()      # 独立草稿模型
    NGRAM = auto()           # N-gram 基线
```

#### 面试考察点

**Q: 投机解码的局限性？**

A:
1. **接受率依赖**: 接受率低时，加速效果不明显甚至更慢
2. **额外开销**: draft 模型的计算和内存开销
3. **实现复杂**: 需要协调 draft 和 verify 两个阶段
4. **分布一致性**: 需要保证输出分布与原模型一致

**Q: 如何改进投机解码？**

A:
1. **自适应推测**: 根据接受率动态调整 draft token 数量
2. **NGRAM 推测**: 无需 draft 模型，基于 n-gram 缓存
3. **EAGLE 树验证**: 树状 draft + 并行验证，提高接受率
4. **热 token 映射**: 缩小 draft vocabulary，减少计算量

### 3.2 量化技术

#### vLLM 支持的量化方法

```python
# vllm/model_executor/layers/quantization/
| 量化方法 | 文件 | 说明 |
|----------|------|------|
| FP8 | fp8.py | FP8 静态/动态量化 |
| AWQ | auto_awq.py | Activation-aware Weight Quantization |
| GPTQ | auto_gptq.py | GPTQ 量化 |
| BitsAndBytes | bitsandbytes.py | 4-bit/8-bit 量化 |
| Compressed Tensors | compressed_tensors/ | 压缩张量格式 |
| TorchAO | torchao.py | PyTorch AO 量化 |
| FBGeMM FP8 | fbgemm_fp8.py | Meta FBGeMM FP8 |
| ModelOpt | modelopt.py | NVIDIA ModelOpt |
| MXFP4 | mxfp4.py | MXFP4 格式 |
```

#### KV Cache 量化

```python
# vllm/v1/kv_cache_interface.py
class KVQuantMode(IntEnum):
    NONE = 0                  # 无量化
    FP8_PER_TENSOR = 1        # FP8 per-tensor 缩放
    INT8_PER_TOKEN_HEAD = 2   # INT8 per-token-head 动态缩放
    FP8_PER_TOKEN_HEAD = 3    # FP8 per-token-head 动态缩放
    INT4_PER_TOKEN_HEAD = 4   # INT4 打包
    NVFP4 = 5                 # NVFP4 打包
```

#### 面试考察点

**Q: 量化的精度-效率权衡？**

A:
| 量化方法 | 精度损失 | 内存节省 | 速度提升 |
|----------|---------|---------|---------|
| FP8 | 极小 | 2x | 1.5-2x |
| INT8 | 小 | 2x | 1.2-1.5x |
| INT4/AWQ | 中等 | 4x | 1.5-2x |
| GPTQ | 中等 | 4x | 1.5-2x |

**Q: KV Cache 量化的挑战？**

A:
1. **精度敏感**: KV Cache 量化对模型输出影响较大
2. **动态范围**: 不同层、不同 head 的动态范围不同
3. **实现复杂**: 需要自定义 CUDA kernel
4. **兼容性**: 需要与注意力后端兼容

### 3.3 分布式并行

#### vLLM 支持的并行策略

```python
# vllm/distributed/parallel_state.py
- TP (Tensor Parallelism) — 张量并行，单层内的权重分割
- PP (Pipeline Parallelism) — 流水线并行，不同层分配到不同设备
- DP (Data Parallelism) — 数据并行，多个独立引擎实例
- DP + EP (Expert Parallelism) — MoE 模型的专家并行
- CP (Context Parallelism) — 上下文并行
```

#### SGLang 支持的并行策略

SGLang 支持 TP/PP/EP/DP/CP，并在大规模 EP 上有深入优化。

#### 面试考察点

**Q: 不同并行策略的适用场景？**

A:
| 策略 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| TP | 单层权重很大 | 降低单 GPU 内存 | 通信开销大 |
| PP | 模型很深 | 降低单 GPU 内存 | 流水线气泡 |
| DP | 请求量大 | 线性扩展吞吐 | 每个 GPU 需要完整模型 |
| EP | MoE 模型 | 专家并行 | 通信开销大 |

**Q: 如何选择并行策略？**

A:
1. **单机多卡**: 优先 TP，其次 PP
2. **多机多卡**: TP + PP + DP
3. **MoE 模型**: TP + EP
4. **长序列**: CP（上下文并行）

### 3.4 注意力后端

#### vLLM 支持的后端

```python
# vllm/v1/attention/backends/
| 后端 | 文件 | 说明 |
|------|------|------|
| FlashAttention | flash_attn.py | 主力后端，支持 FA2/FA3/FA4 |
| FlashInfer | flashinfer.py | 高性能替代后端 |
| Triton | triton_attn.py | Triton 实现 |
| FlexAttention | flex_attention.py | PyTorch FlexAttention |
| MLA | mla/ 目录 | DeepSeek MLA 专用 |
| ROCm | rocm_attn.py | AMD GPU 后端 |
```

#### SGLang 支持的后端

SGLang 支持 40+ 种注意力后端，FlashInfer 为主力后端。

#### 面试考察点

**Q: FlashAttention 的核心优势？**

A:
1. **IO 感知**: 减少 HBM 访问，增加 SRAM 计算
2. **分块计算**: 将大矩阵分块，避免内存溢出
3. **在线 Softmax**: 无需存储完整 attention matrix
4. **反向传播优化**: 使用 recomputation 减少内存

**Q: MLA (Multi-head Latent Attention) 的优势？**

A:
1. **压缩 KV Cache**: 将 KV Cache 压缩到低维空间
2. **降低内存**: 显著减少 KV Cache 内存占用
3. **保持精度**: 通过学习压缩矩阵保持模型精度
4. **DeepSeek 专用**: DeepSeek V3/R1 的核心技术

---

## 第四部分：RL/后训练核心知识点

### 4.1 SGLang 在 RL/后训练中的应用

#### 定位

根据 README:
> **RL & Post-Training Backbone**: SGLang is a proven rollout backend used for training many frontier models, with native RL integrations and adoption by well-known post-training frameworks such as **AReaL**, **Miles**, **slime**, **Tunix**, **verl** and more.

#### 为什么 SGLang 适合 RL/后训练？

1. **高效的 KV Cache 复用**:
   - RL 训练中，多个 rollout 共享相同的 system prompt
   - RadixAttention 自动缓存和复用这些共享前缀
   - 显著减少计算和内存开销

2. **快速的权重更新**:
   - RL 训练需要频繁更新模型权重
   - SGLang 支持热更新权重，无需重启服务

3. **高吞吐量**:
   - 连续批处理 (Continuous Batching)
   - 零开销 CPU 调度器
   - 支持大规模并行

4. **结构化输出**:
   - RL 训练中经常需要约束输出格式
   - SGLang 的压缩 FSM 提供高效的结构化输出

#### 面试考察点

**Q: RL 训练中为什么需要高效的推理引擎？**

A:
1. **Rollout 生成**: RL 训练需要大量采样，推理效率直接影响训练速度
2. **在线学习**: 需要快速生成新的训练数据
3. **资源效率**: 推理和训练共享 GPU 资源，需要高效利用

**Q: SGLang 如何支持 RL 训练？**

A:
1. **批量推理**: 支持大批量并行生成
2. **权重热更新**: 支持在线更新模型权重
3. **KV Cache 复用**: 自动缓存共享前缀
4. **结构化输出**: 约束输出格式

### 4.2 GRPO vs PPO

#### 核心区别

| 算法 | 公式 | 特点 |
|------|------|------|
| PPO | A_t = δ_t + (γλ) * A_{t+1}, δ_t = r_t + γ*V(s_{t+1}) - V(s_t) | 需要 Critic，GAE 估计 |
| GRPO | A_i = (r_i - mean(r_group)) / std(r_group) | 不需要 Critic，组内相对 |

#### 面试考察点

**Q: GRPO 的局限性？**

A:
1. **稀疏奖励问题**: 如果所有样本 reward 相同，advantage 为 0，无法学习
2. **组内偏差**: advantage 只反映组内相对排名，可能与全局最优不一致
3. **需要更多采样**: 每个 prompt 需要多个响应，采样成本高

**Q: 如何改进 GRPO？**

A:
1. **动态采样**: 过滤 reward 标准差为 0 的样本
2. **Advantage 标准化**: 在 DP group 内做 whitening
3. **混合方法**: GRPO + KL 惩罚
4. **REINFORCE++**: 带折扣回报的 REINFORCE

### 4.3 PPO Clipping 机制

#### 核心公式

```python
L_PG = max(-r(θ)·A, -clip(r(θ), 1-ε, 1+ε_high)·A)

# 代码实现
ratio = (-ppo_kl).exp()  # ratio = exp(new_logp - old_logp)
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # 悲观估计
```

#### 面试考察点

**Q: PPO Clipping 的局限性？**

A:
1. **保守性**: 取最大值是悲观估计，可能限制策略改进
2. **不对称裁剪**: eps_clip 和 eps_clip_high 需要分别调参
3. **负 advantage 问题**: 当 advantage 为负时，ratio 越大 loss 越小
4. **序列级 vs token 级**: 标准 PPO 是 token 级裁剪

**Q: 如何改进 PPO Clipping？**

A:
1. **Dual-Clip PPO**: 对负 advantage 额外施加上界裁剪
2. **CISPO**: stop-gradient 裁剪，被裁剪的 token 仍然贡献梯度
3. **KL 惩罚**: 添加 KL 散度项限制策略偏离
4. **自适应裁剪**: 根据训练进度动态调整裁剪范围

---

## 第五部分：Agent 应用核心知识点

### 5.1 Agent 训练的特殊挑战

#### Loss Masking

**问题背景**: Agent 训练中，工具返回的 token 不是模型生成的，如果参与 loss 计算，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。

**解决方案**: 使用 loss_mask 标记每个 token 是否参与 loss 计算：

```python
# python/sglang/srt/managers/schedule_batch.py
class Sample:
    loss_mask: torch.Tensor  # 标记每个 token 是否参与 loss 计算

# loss 计算
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

#### 面试考察点

**Q: 为什么 Agent 训练需要 Loss Masking？**

A:
1. **工具返回不是模型生成**: 工具返回的 token 不应该参与 loss 计算
2. **避免学习复制**: 如果参与 loss，模型会学习"复制"工具输出
3. **梯度方向不准确**: 工具返回的梯度方向不准确，会导致训练不稳定

**Q: Loss Masking 如何实现？**

A:
1. **标记工具返回**: 在 tokenization 时标记工具返回的 token
2. **计算 loss_mask**: 为每个 token 生成 0/1 mask
3. **加权 loss**: 使用 mask 对每个 token 的 loss 进行加权

### 5.2 Session Routing

#### 解决什么问题？

**问题背景**: Agent 工作流中，同一个 session 的多个请求共享相同的 system prompt 和工具定义。如果每次都重新计算，会造成大量重复计算。

**解决方案**: 使用 session routing 将同一个 session 的请求路由到同一个 GPU，最大化 prefix cache 命中率。

#### 面试考察点

**Q: Session Routing 的实现方式？**

A:
1. **基于 session_id**: 将 session_id 作为路由键
2. **一致性哈希**: 使用一致性哈希将 session 映射到 GPU
3. **负载均衡**: 在 cache 命中率和负载均衡之间权衡

**Q: Session Routing 的挑战？**

A:
1. **负载不均衡**: 某些 session 可能比其他 session 更活跃
2. **故障恢复**: GPU 故障时需要重新路由 session
3. **动态调整**: 需要根据负载动态调整路由策略

### 5.3 结构化输出

#### vLLM 实现

vLLM 支持使用 xgrammar 或 guidance 生成结构化输出。

#### SGLang 实现

SGLang 使用**压缩有限状态机 (Compressed FSM)** 实现 3 倍更快的 JSON 解码：

```python
# python/sglang/srt/constrained/grammar_manager.py
class GrammarManager:
    def process_req_with_grammar(self, req):
        if req.sampling_params.json_schema is not None:
            key = ("json", req.sampling_params.json_schema)
        elif req.sampling_params.regex is not None:
            key = ("regex", req.sampling_params.regex)
        elif req.sampling_params.ebnf is not None:
            key = ("ebnf", req.sampling_params.ebnf)
        # ...
```

#### 面试考察点

**Q: 结构化输出的实现方式？**

A:
1. **有限状态机 (FSM)**: 将 JSON Schema 编译为 FSM
2. **压缩 FSM**: 将 FSM 压缩，减少状态数
3. **实时约束**: 在 token 生成时实时约束输出
4. **语法后端**: XGrammar、Outlines、LLGuidance 等

**Q: 结构化输出的挑战？**

A:
1. **性能开销**: FSM 匹配有计算开销
2. **兼容性**: 需要与不同的注意力后端兼容
3. **表达能力**: 某些复杂约束难以用 FSM 表达
4. **错误处理**: 如何处理非法 token

---

## 第六部分：面试深挖问题与回答模板

### 6.1 底层原理深入理解

#### Q: 请详细解释 PagedAttention 的工作原理

**回答模板**:

"PagedAttention 解决的核心问题是 **LLM 推理中 KV Cache 的内存管理**。传统方法需要预分配大块连续内存，导致严重的内存碎片和浪费。

PagedAttention 借鉴了操作系统的虚拟内存分页机制：
1. 将 KV Cache 划分为固定大小的块（Block），类似内存页
2. 每个请求的 KV Cache 通过块表（Block Table）映射到物理块，类似页表
3. 块可以在请求间共享，实现前缀缓存
4. 按需分配，避免预分配大块连续内存导致的碎片和浪费

核心优势：
- 零内存浪费：按需分配，避免预分配
- 内存共享：块可在请求间共享
- 减少碎片：固定大小块消除外部碎片
- 高效驱逐：LRU 双向链表实现 O(1) 驱逐

局限性：
- 块大小限制：block_size 必须是 16 的倍数
- 元数据开销：每个块需要维护 hash、ref_cnt 等元数据
- 驱逐策略：LRU 不一定最优
- 并发控制：引用计数在高并发下可能成为瓶颈

改进方向：
- 分层缓存：将 KV Cache 扩展到 CPU 和分布式存储
- 多级驱逐策略：支持 LRU、LFU、SLRU 等
- 页对齐优化：使用 page_size 对齐减少碎片"

#### Q: RadixAttention 和 PagedAttention 有什么区别？

**回答模板**:

"RadixAttention 和 PagedAttention 都是为了解决 LLM 推理中 KV Cache 的内存管理问题，但采用了不同的方法。

**PagedAttention (vLLM)**:
- 使用块表 + 块池管理 KV Cache
- 块级别的缓存复用
- 需要手动管理缓存
- 适用于通用场景

**RadixAttention (SGLang)**:
- 使用 Radix Tree 管理 KV Cache
- 前缀级别的缓存复用
- 自动前缀匹配
- 适用于多轮对话、Agent 场景

核心区别：
1. **数据结构**: PagedAttention 使用块表，RadixAttention 使用 Radix Tree
2. **缓存粒度**: PagedAttention 是块级别，RadixAttention 是前缀级别
3. **匹配方式**: PagedAttention 需要手动管理，RadixAttention 自动匹配
4. **适用场景**: PagedAttention 通用，RadixAttention 适合多轮对话

选择建议：
- 如果是通用场景，选择 vLLM 的 PagedAttention
- 如果是多轮对话、Agent 场景，选择 SGLang 的 RadixAttention"

### 6.2 实验和方案验证能力

#### Q: 如何验证 Prefix Caching 的有效性？

**回答模板**:

"验证 Prefix Caching 的有效性需要从多个维度进行实验：

**实验设计**:
1. **指标定义**:
   - Cache Hit Rate：前缀缓存命中率
   - TTFT (Time To First Token)：首 token 延迟
   - Throughput：吞吐量（token/s）

2. **实验方案**:
   ```bash
   # 启动服务，开启 metrics
   python -m sglang.launch_server --enable-metrics --schedule-policy lpm
   
   # 运行 benchmark，模拟多轮对话
   python3 -m sglang.benchmark.serving \
     --backend sglang \
     --dataset-name random \
     --num-prompts 3000 \
     --random-input 1024 \
     --random-output 1024
   
   # 查看 Prometheus 指标
   curl http://localhost:30000/metrics | grep cache_hit_rate
   ```

3. **对比实验**:
   | 配置 | Cache Hit Rate | TTFT (P99) | Throughput |
   |------|---------------|------------|------------|
   | 无 Prefix Caching | 0% | 500ms | 1000 token/s |
   | 有 Prefix Caching | 75% | 150ms | 2500 token/s |
   | + LPM 调度策略 | 85% | 120ms | 2800 token/s |

**公平性保证**:
- 相同的请求分布
- 相同的硬件环境
- 相同的模型
- 多次运行取平均值"

#### Q: 如何验证 RL 训练的正确性？

**回答模板**:

"验证 RL 训练的正确性需要从多个维度进行：

**第一步：精度对齐验证**
```python
# 检查 log_probs 和 ref_log_probs 是否相等（第一步 KL 应为 0）
assert torch.allclose(log_probs, ref_log_probs, atol=1e-6)
# 检查 grad_norm 是否较小
assert grad_norm < 1.0
```

**第二步：权重同步验证**
```bash
--check-weight-update-equal  # 验证 Megatron -> SGLang 权重同步
```

**第三步：数值一致性验证**
```python
# 验证不同 CP 大小下梯度一致
for cp_size in [1, 2, 4]:
    grad = compute_loss(cp_size=cp_size)
    assert torch.allclose(grad, expected_grad, atol=1e-5)
```

**第四步：确定性复现**
```bash
--sglang-enable-deterministic-inference
--deterministic-mode
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**常见问题排查**:
1. KL 不为 0：检查 Transformer Engine 的非确定性 kernel
2. 权重同步错误：检查参数名映射
3. 梯度异常：检查是否有 NaN/Inf"

### 6.3 问题定位能力

#### Q: 模型上线后能力突然下降，如何排查？

**回答模板**:

"模型上线后能力突然下降，我的排查思路是：

**Step 1：确认问题范围**
- 是所有请求都下降，还是特定类型请求？
- 是突然下降，还是逐渐下降？
- 有没有发布新版本或修改配置？

**Step 2：检查权重同步**
```bash
# 验证权重同步是否正确
--check-weight-update-equal

# 检查 Delta Weight Sync 的差异
# 如果某参数全为零，说明该参数被错误量化或丢失
```

**Step 3：检查数值精度**
```bash
# 检查是否使用了正确的精度
--dtype bf16  # 确保与训练时一致

# 检查 KV Cache 精度
--kv-cache-dtype auto  # 而非 fp8
```

**Step 4：检查前缀缓存**
```bash
# 禁用前缀缓存，排除缓存污染
--disable-radix-cache
```

**Step 5：检查日志**
```bash
# 开启详细日志
--log-requests --log-requests-level 3

# 检查是否有异常
grep "error\|warning\|OOM" server.log
```

**常见原因**:
- 权重同步错误
- 数值精度不匹配
- 前缀缓存污染
- 量化误差"

#### Q: CUDA OOM 如何排查？

**回答模板**:

"CUDA OOM 的排查思路是：

**Step 1：确认 OOM 阶段**
```bash
# Prefill 阶段 OOM
grep "OOM" server.log | grep "prefill"

# Decode 阶段 OOM
grep "OOM" server.log | grep "decode"
```

**Step 2：检查显存使用**
```bash
nvidia-smi -l 1  # 实时监控显存
```

**Step 3：检查配置**
```bash
# 检查关键配置
--mem-fraction-static  # KV 缓存池比例
--chunked-prefill-size  # Prefill 分块大小
--max-running-requests  # 最大运行请求数
--cuda-graph-max-bs  # CUDA Graph 最大 batch size
```

**解决方案**:
| 阶段 | 原因 | 解决方案 |
|------|------|---------|
| Prefill | 长 prompt | 降低 chunked-prefill-size |
| Decode | 并发过高 | 降低 max-running-requests |
| 通用 | KV 缓存池过大 | 降低 mem-fraction-static |
| CUDA Graph | BS 过大 | 降低 cuda-graph-max-bs"

### 6.4 工程落地能力

#### Q: 如何部署一个生产级的 LLM 推理服务？

**回答模板**:

"部署生产级 LLM 推理服务需要考虑多个方面：

**部署架构**:
```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  SGLang Node  │    │  SGLang Node  │    │  SGLang Node  │
│  (TP=4, DP=2) │    │  (TP=4, DP=2) │    │  (TP=4, DP=2) │
└───────────────┘    └───────────────┘    └───────────────┘
```

**启动配置**:
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --tp 4 \
  --dp 2 \
  --mem-fraction-static 0.85 \
  --schedule-policy lpm \
  --enable-metrics \
  --cuda-graph-max-bs 256 \
  --chunked-prefill-size 8192
```

**关键监控指标**:
| 指标 | 告警阈值 | 说明 |
|------|---------|------|
| num_queue_reqs | > 2000 | 队列过长 |
| token_usage | > 0.95 | KV 缓存接近满 |
| num_retracted_reqs_total | 增速 > 10/min | 调度过于激进 |
| time_to_first_token_seconds P99 | > 1s | Prefill 瓶颈 |
| cache_hit_rate | < 0.3 | 缓存命中率低 |

**稳定性保证**:
1. **Watchdog 系统**: 超时 kill 进程
2. **Crash Dump**: 崩溃时自动 dump 请求
3. **请求超时**: 队列超时、运行超时
4. **健康检查**: /health_generate 端点"

### 6.5 业务与实际场景理解

#### Q: Prefix Caching 适合什么场景？

**回答模板**:

"Prefix Caching 适合以下场景：

**适合的场景**:
| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 多轮对话 | system prompt + 历史对话共享 | TTFT 降低 50-70% |
| Agent 工作流 | system prompt + 工具定义共享 | 吞吐提升 2-3x |
| 批量推理 | 相同 system prompt 的请求 | 吞吐提升 3-5x |
| RAG 应用 | 相同 context 的请求 | TTFT 降低 40-60% |

**不适合的场景**:
| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 纯生成任务 | 无共享前缀 | 禁用缓存减少开销 |
| 高度个性化 | 每个请求前缀不同 | 使用 cache_salt 隔离 |
| 实时性要求极高 | 缓存匹配有开销 | 禁用缓存，使用 FCFS |

**用户关心什么**:
1. **延迟**: TTFT 和 TPOT 是否满足 SLA
2. **成本**: GPU 成本是否降低
3. **稳定性**: 延迟是否稳定，是否有毛刺
4. **可扩展性**: 能否水平扩展应对流量增长

**资源有限时优先优化什么**:
1. **KV 缓存大小** (mem-fraction-static): 直接影响并发能力
2. **CUDA Graph** (cuda-graph-max-bs): 直接影响 decode 吞吐
3. **调度策略** (schedule-policy lpm): 间接影响缓存命中率
4. **Chunked Prefill** (chunked-prefill-size): 影响长 prompt 处理能力
5. **DP 实例**: 水平扩展应对流量增长"

---

## 第七部分：vLLM vs SGLang 架构对比

### 7.1 核心差异

| 特性 | vLLM | SGLang |
|------|------|--------|
| **核心创新** | PagedAttention | RadixAttention |
| **缓存管理** | 块表 + 块池 | Radix Tree |
| **调度策略** | FCFS 为主 | Cache-Aware (LPM/DFS-Weight) |
| **前端 DSL** | 无 | 完整的 DSL 语言 |
| **投机解码** | 基础支持 | DFlash/EAGLE 等多种算法 |
| **结构化输出** | 基础 FSM | 压缩 FSM，3x 更快 JSON |
| **Prefill-Decode 分离** | 实验性支持 | 原生支持 |
| **注意力后端** | FlashAttention 为主 | FlashInfer 为主，40+ 种 |
| **分布式并行** | TP/PP 为主 | TP/PP/EP/DP/CP，大规模 EP 优化 |
| **RL/后训练** | 基础支持 | 原生集成，被多个框架采用 |
| **硬件支持** | NVIDIA/AMD 为主 | NVIDIA/AMD/Intel/Google TPU/Ascend NPU |

### 7.2 选择建议

**选择 vLLM**:
- 通用场景，不需要特殊优化
- 需要稳定的生产环境
- 团队熟悉 vLLM 生态

**选择 SGLang**:
- 多轮对话、Agent 场景
- RL/后训练场景
- 需要高效的前缀缓存
- 需要结构化输出
- 需要大规模分布式部署

---

## 附录：关键代码路径速查表

### vLLM 关键代码路径

```
vllm/v1/engine/core.py              # EngineCore 核心引擎
vllm/v1/core/sched/scheduler.py     # 调度器实现
vllm/v1/core/kv_cache_manager.py    # KV Cache 管理器
vllm/v1/attention/backends/         # 注意力后端
vllm/v1/worker/gpu_model_runner.py  # GPU 模型运行器
vllm/v1/spec_decode/                # 投机解码
vllm/distributed/                   # 分布式并行
vllm/model_executor/layers/quantization/  # 量化
```

### SGLang 关键代码路径

```
python/sglang/srt/managers/scheduler.py      # 调度器实现
python/sglang/srt/mem_cache/radix_cache.py   # Radix Cache 实现
python/sglang/srt/layers/radix_attention.py  # RadixAttention 层
python/sglang/srt/layers/attention/          # 注意力后端
python/sglang/srt/models/                    # 模型实现
python/sglang/srt/speculative/               # 投机解码
python/sglang/srt/constrained/               # 结构化输出
python/sglang/lang/                          # 前端 DSL
```

### 关键配置参数速查

**vLLM**:
```bash
--tensor-parallel-size 4        # TP 大小
--pipeline-parallel-size 2      # PP 大小
--max-num-seqs 256              # 最大并发请求数
--max-num-batched-tokens 8192   # 最大批次 token 数
--gpu-memory-utilization 0.9    # GPU 内存利用率
--enable-prefix-caching         # 启用前缀缓存
--quantization fp8              # 量化方法
```

**SGLang**:
```bash
--tp 4                          # TP 大小
--dp 2                          # DP 大小
--mem-fraction-static 0.85      # KV 缓存池比例
--schedule-policy lpm           # 调度策略
--chunked-prefill-size 8192     # Chunked Prefill 大小
--cuda-graph-max-bs 256         # CUDA Graph 最大 BS
--speculative-algorithm eagle   # 投机解码算法
```

---

**最后更新**: 2026-06-27

**适用场景**: LLM 算法实习面试，针对 LLM & Agent 应用及后训练方向
