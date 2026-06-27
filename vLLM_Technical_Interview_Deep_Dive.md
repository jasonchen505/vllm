# vLLM 技术面试五类问题深度应对指南

> 基于 vLLM 源码深度分析，针对 LLM 算法实习面试
> 核心理念：不仅讲清楚概念，更要讲清楚**解决什么问题、局限性、改进方法、实验验证、问题定位、工程落地、业务价值**

---

## 目录

- [第一类：底层原理深入理解](#第一类底层原理深入理解)
- [第二类：实验和方案验证能力](#第二类实验和方案验证能力)
- [第三类：问题定位能力](#第三类问题定位能力)
- [第四类：工程落地能力](#第四类工程落地能力)
- [第五类：业务与实际场景理解](#第五类业务与实际场景理解)

---

## 第一类：底层原理深入理解

> **核心要求**：不是回答清楚概念，而是讲清楚**这个方法解决什么问题，存在哪些局限性，有哪些改进方法**

### 1.1 PagedAttention

#### 解决什么问题？

**问题背景**：LLM 推理中，KV Cache 的内存管理面临严重挑战：
- 不同请求的序列长度不同，导致内存碎片
- 传统方法预分配最大长度内存，浪费严重（一个 4096 token 的请求如果预分配，短请求会浪费大量内存）
- 无法在请求间共享 KV Cache

**具体痛点**：
- 内存碎片化：随着请求的到达和完成，GPU 内存中会出现大量不连续的小块空闲区域
- 内存浪费：预分配策略导致短序列请求占用过多内存
- 无法共享：相同前缀的请求（如多轮对话中的 system prompt）无法共享 KV Cache

**解决方案**：借鉴操作系统的虚拟内存分页机制，将 KV Cache 划分为固定大小的"块"（Block），通过"块表"（Block Table）映射。

#### 核心设计原理

**数据结构设计**（`vllm/v1/core/block_pool.py`）：

```python
class KVCacheBlock:
    # 块 ID（不可变）
    block_id: int
    # 块哈希（块满时分配，驱逐时重置）
    block_hash: BlockHash
    # 当前使用此块的请求数
    ref_cnt: int
    # 双向链表指针，用于空闲队列
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
```

**设计要点**：
1. **预分配所有块**：初始化时创建所有 KVCacheBlock，避免 Python 对象创建开销
2. **双向链表内嵌**：直接在块中嵌入链表指针，实现 O(1) 的队列操作
3. **引用计数**：支持块在请求间共享

**内存布局**（FlashAttention 后端）：
```
(num_blocks, 2, block_size, num_kv_heads, head_size)
```
- `num_blocks`: 物理块数
- `2`: K 和 V 合并存储
- `block_size`: 每块的 token 数（必须是 16 的倍数，这是 CUDA kernel 的要求）

#### 局限性

1. **块大小限制**：block_size 必须是 16 的倍数（CUDA kernel 对齐要求），可能导致内部碎片
   - 例如：如果请求有 17 个 token，需要 2 个块（32 token 空间），浪费 15 个 token 的空间

2. **元数据开销**：每个块需要维护 hash、ref_cnt、双向链表指针等元数据
   - 对于大量小块，元数据开销比例较高

3. **驱逐策略单一**：默认使用 LRU（最近最少使用），不一定是最优策略
   - 某些场景下 LFU（最不频繁使用）或 SLRU（分段 LRU）更好

4. **并发控制**：引用计数在高并发下可能成为瓶颈
   - 多个线程同时修改 ref_cnt 需要原子操作

5. **哈希冲突风险**：虽然 v0.11 使用 sha256 减少冲突，但理论上仍可能发生

#### 改进方法

1. **分层缓存（Hierarchical Cache）**：
   ```python
   # GPU (L1) -> CPU (L2) -> 分布式存储 (L3)
   class HierarchicalCache:
       def prefetch(self, node):  # 从 L2/L3 预取到 L1
       def write_back(self, node):  # 从 L1 写回到 L2/L3
   ```
   - 突破 GPU 显存限制，支持更长的上下文

2. **多级驱逐策略**：
   ```python
   # SLRU (Segmented LRU)：保护高频访问的缓存
   if hit_count < protect_threshold:
       return segment  # probationary segment
   else:
       return segment  # protected segment
   ```

3. **页对齐优化**：使用 `page_size` 对齐减少碎片
   - vLLM 支持 `hash_block_size` 与实际 block_size 分离

4. **内容寻址**：通过 `BlockHash` 实现跨节点的缓存共享
   ```python
   @dataclass
   class BlockHash:
       token_ids: tuple[int, ...]   # 块中的 token IDs
       extra_keys: tuple            # 额外键（如 LoRA ID、mm hash）
   ```

5. **KV Cache 量化**：
   ```python
   class KVQuantMode(IntEnum):
       NONE = 0                  # 无量化
       FP8_PER_TENSOR = 1        # FP8 per-tensor 缩放
       INT8_PER_TOKEN_HEAD = 2   # INT8 per-token-head 动态缩放
       FP8_PER_TOKEN_HEAD = 3    # FP8 per-token-head 动态缩放
       INT4_PER_TOKEN_HEAD = 4   # INT4 打包
       NVFP4 = 5                 # NVFP4 打包
   ```

#### 面试回答模板

"PagedAttention 解决的核心问题是 **LLM 推理中 KV Cache 的内存管理**。传统方法需要预分配大块连续内存，导致严重的内存碎片和浪费。

PagedAttention 借鉴了操作系统的虚拟内存分页机制：
1. 将 KV Cache 划分为固定大小的块（Block），类似内存页
2. 每个请求的 KV Cache 通过块表（Block Table）映射到物理块，类似页表
3. 块可以在请求间共享，实现前缀缓存
4. 按需分配，避免预分配大块连续内存导致的碎片和浪费

它的局限性在于：
1. 块大小必须是 16 的倍数，可能导致内部碎片
2. 每个块有元数据开销（hash、ref_cnt、链表指针）
3. 默认 LRU 驱逐策略不一定最优
4. 引用计数在高并发下可能成为瓶颈

改进方向包括：
1. 分层缓存，将 KV Cache 扩展到 CPU 和分布式存储
2. 多级驱逐策略（如 SLRU）
3. KV Cache 量化（FP8/INT8/INT4）减少内存占用
4. 内容寻址实现跨节点缓存共享"

### 1.2 连续批处理（Continuous Batching）

#### 解决什么问题？

**问题背景**：传统静态批处理需要等待整个 batch 完成才能处理下一个请求，导致：
- 短请求等待长请求完成，延迟高
- GPU 利用率低，有空闲时间
- 无法处理动态到达的请求

**具体痛点**：
- 如果一个 batch 中有一个 1000 token 的请求和一个 10 token 的请求，10 token 的请求需要等待 1000 token 的请求完成
- GPU 在等待期间处于空闲状态
- 新到达的请求无法加入正在运行的 batch

**解决方案**：每个 iteration 动态决定哪些请求进入 prefill/decode，请求完成后立即释放资源。

#### 核心设计原理

vLLM 的调度器以 **token 粒度**工作（`vllm/v1/core/sched/scheduler.py:388`）：

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    # 关键设计：没有"解码阶段"或"预填充阶段"的概念
    # 每个请求只有 num_computed_tokens 和 num_tokens_with_spec
    # 调度器尝试分配 token，使 num_computed_tokens 追赶 num_tokens_with_spec
    
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

**关键设计点**：
1. **Token 级调度**：不是请求级，而是 token 级，更细粒度
2. **两阶段调度**：先调度 running 请求，再调度 waiting 请求
3. **抢占机制**：内存不足时抢占低优先级请求
4. **前缀缓存集成**：自动查找并复用已缓存的前缀

#### 局限性

1. **调度开销**：每个 iteration 都需要调度决策，CPU 开销增加
   - 需要遍历所有 running 请求，计算新 token 数
   - 需要查找前缀缓存，分配新块

2. **内存碎片**：动态增删请求可能导致内存碎片
   - 虽然 PagedAttention 减少了碎片，但频繁的分配/释放仍会产生碎片

3. **延迟波动**：batch size 动态变化导致延迟不稳定
   - 当 batch size 较大时，每个请求的延迟会增加

4. **公平性问题**：长请求可能被饿死
   - 如果持续有新请求到达，长请求可能一直被推迟

5. **抢占开销**：被抢占的请求需要重新计算 KV Cache
   - 虽然有前缀缓存，但仍需要重新计算部分 token

#### 改进方法

1. **Overlap Scheduling**：CPU 调度与 GPU 计算重叠
   ```python
   def event_loop_overlap(self):
       result_queue = deque()
       while True:
           batch = self.get_next_batch_to_run()
           if batch:
               batch_result = self.run_batch(batch)  # GPU 计算
               result_queue.append((batch, batch_result))
           if self.last_batch:
               tmp_batch, tmp_result = result_queue.popleft()
               self.process_batch_result(tmp_batch, tmp_result)  # CPU 处理
   ```
   - vLLM V1 支持异步调度（`async_scheduling`）

2. **Chunked Prefill**：长 prompt 分块处理
   ```python
   # vllm/v1/core/sched/scheduler.py
   if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
       num_new_tokens = self.scheduler_config.long_prefill_token_threshold
   ```
   - 避免长 prompt 独占批次计算预算

3. **Retraction 机制**：内存不足时回退低优先级请求
   ```python
   def _preempt_request(self, request, timestamp):
       # 将请求从 running 移回 waiting
       # 释放其 KV 缓存块
       self.kv_cache_manager.free(request)
       request.status = RequestStatus.PREEMPTED
       request.num_preemptions += 1
   ```

4. **优先级调度**：支持请求级优先级
   ```python
   # vllm/config/scheduler.py
   policy: SchedulerPolicy = "fcfs"  # 或 "priority"
   ```

5. **Watermark 机制**：预留一定比例的空闲块，避免频繁抢占
   ```python
   # vllm/v1/core/kv_cache_manager.py
   self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
   ```

#### 面试回答模板

"连续批处理解决的核心问题是 **静态批处理的低效和高延迟**。传统方法需要等待整个 batch 完成，导致短请求等待长请求、GPU 利用率低。

vLLM 的实现采用 token 级调度：
1. 每个 iteration 动态决定每个请求处理多少 token
2. 先调度 running 请求，再调度 waiting 请求
3. 内存不足时抢占低优先级请求
4. 自动集成前缀缓存

它的局限性在于：
1. 调度开销增加（每个 iteration 都需要决策）
2. 内存碎片（频繁分配/释放）
3. 延迟波动（batch size 动态变化）
4. 公平性问题（长请求可能被饿死）

改进方向包括：
1. Overlap Scheduling，CPU 调度与 GPU 计算重叠
2. Chunked Prefill，长 prompt 分块处理
3. Retraction 机制，内存不足时回退请求
4. 优先级调度，保证关键请求延迟
5. Watermark 机制，预留空闲块避免频繁抢占"

### 1.3 Prefix Caching（前缀缓存）

#### 解决什么问题？

**问题背景**：在多轮对话、Agent 工作流、system prompt 复用等场景下，多个请求共享相同的前缀。传统方法每次都要重新计算这些共享前缀的 KV Cache。

**具体痛点**：
- 多轮对话中，system prompt + 历史对话每轮都要重新计算
- Agent 的多轮工具调用中，每轮都要重新计算 system prompt
- 批量推理中，相同 system prompt 的请求无法共享 KV Cache

**解决方案**：缓存已计算的 KV Cache 块，当新请求有相同前缀时直接复用。

#### 核心设计原理

vLLM 使用**块哈希**实现前缀缓存（`vllm/docs/design/prefix_caching.md`）：

```python
# 哈希计算方式
Block 1: hash(tuple(block_tokens))
Block 2: hash(tuple(parent_hash, block_tokens))
Block 3: hash(tuple(parent_hash, block_tokens))
```

**示例**：
```
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

**关键设计点**：
1. **只缓存满块**：部分填充的块不缓存
2. **父哈希链**：每个块的哈希包含父块的哈希，确保唯一性
3. **额外键**：支持 LoRA ID、多模态输入哈希、cache_salt 等
4. **LRU 驱逐**：使用双向链表实现 O(1) 驱逐

**缓存驱逐流程**：
```python
# vllm/v1/core/block_pool.py
class FreeKVCacheBlockQueue:
    def evict(self):
        # 1. 从队头弹出 LRU 块
        block = self.free_queue.popleft()
        # 2. 从缓存中移除
        self.cached_block_hash_to_block.pop(block.block_hash, block.block_id)
        # 3. 清除块哈希
        block.block_hash = None
        return block
```

#### 局限性

1. **内存开销**：需要维护块哈希表和双向链表
   - 每个缓存块需要额外的内存存储哈希和链表指针

2. **哈希计算开销**：每个块需要计算哈希
   - 虽然使用 sha256 减少冲突，但计算开销仍存在

3. **只缓存满块**：部分填充的块无法缓存
   - 对于短请求，可能无法利用前缀缓存

4. **驱逐策略单一**：LRU 不一定是最优策略
   - 某些场景下 LFU 或 SLRU 更好

5. **并发控制**：多个请求同时访问缓存需要同步

6. **内存碎片**：虽然 PagedAttention 减少了碎片，但缓存驱逐仍可能产生碎片

#### 改进方法

1. **分层缓存（HiCache）**：
   ```python
   # GPU (L1) -> CPU (L2) -> 分布式存储 (L3)
   class HiRadixCache:
       def prefetch(self, node):  # 从 L2/L3 预取到 L1
       def write_back(self, node):  # 从 L1 写回到 L2/L3
   ```

2. **多级驱逐策略**：
   ```python
   # SLRU (Segmented LRU)：保护高频访问的缓存
   if hit_count < protect_threshold:
       return segment  # probationary segment
   else:
       return segment  # protected segment
   ```

3. **Cache Isolation**：使用 `cache_salt` 隔离不同租户的缓存
   ```json
   {
     "messages": [...],
     "cache_salt": "your-cache-salt"
   }
   ```

4. **内容寻址**：通过 `BlockHash` 实现跨节点的缓存共享

5. **异步缓存更新**：后台线程更新缓存，避免阻塞主线程

#### 面试回答模板

"前缀缓存解决的核心问题是 **多轮对话和 Agent 场景下的前缀重复计算**。传统方法每轮都要重新计算共享前缀的 KV Cache，而前缀缓存通过块哈希实现前缀级别的缓存复用。

vLLM 的实现使用块哈希：
1. 每个块的哈希包含父块的哈希，确保唯一性
2. 只缓存满块，部分填充的块不缓存
3. 使用 LRU 双向链表实现 O(1) 驱逐
4. 支持额外键（LoRA ID、mm hash、cache_salt）

它的局限性在于：
1. 内存开销（需要维护哈希表和链表）
2. 哈希计算开销
3. 只缓存满块
4. LRU 驱逐策略不一定最优

改进方向包括：
1. 分层缓存，扩展到 CPU 和分布式存储
2. 多级驱逐策略（如 SLRU）
3. Cache Isolation，使用 cache_salt 隔离
4. 内容寻址实现跨节点共享"

### 1.4 Chunked Prefill（分块预填充）

#### 解决什么问题？

**问题背景**：当 prompt 很长时，prefill 阶段会独占整个批次的计算预算，导致：
- decode 请求被阻塞，延迟增加
- GPU 利用率不均衡（prefill 时 GPU 满载，decode 时空闲）
- 长 prompt 的首 token 延迟（TTFT）很高

**具体痛点**：
- 一个 10000 token 的 prompt 如果一次性 prefill，会阻塞所有 decode 请求
- GPU 在 prefill 期间满载，但在 decode 期间空闲
- 用户等待首 token 的时间很长

**解决方案**：将长 prompt 分成多个 chunk 处理，每个 chunk 可以与 decode 请求混合调度。

#### 核心设计原理

```python
# vllm/v1/core/sched/scheduler.py
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
```

**配置参数**（`vllm/config/scheduler.py`）：
```python
max_num_partial_prefills: int = 1  # 最大并发部分预填充数
max_long_partial_prefills: int = 1  # 最大并发长部分预填充数
long_prefill_token_threshold: int = 0  # 长 prompt 阈值
enable_chunked_prefill: bool = True  # 是否启用分块预填充
```

**调度流程**：
1. 调度器计算每个请求的 `num_new_tokens`
2. 如果 `num_new_tokens > long_prefill_token_threshold`，则截断
3. 将截断后的 token 与 decode 请求混合调度
4. 下一个 iteration 继续处理剩余的 token

#### 局限性

1. **额外开销**：需要多次调度同一个请求
   - 每个 chunk 都需要调度决策
   - 需要跟踪每个 chunk 的进度

2. **状态管理复杂**：需要跟踪每个请求的 prefill 进度
   - `num_computed_tokens` 需要正确更新
   - 需要处理部分完成的请求

3. **内存管理**：需要为 partial prefill 分配临时存储
   - 部分计算的 KV Cache 需要正确存储

4. **TTFT 可能增加**：对于短 prompt，分块反而增加开销
   - 需要合理设置 `long_prefill_token_threshold`

5. **与投机解码的交互**：分块可能影响投机解码的效率

#### 改进方法

1. **自适应分块**：根据 GPU 负载动态调整 chunk 大小
   ```python
   if gpu_utilization < threshold:
       chunk_size = min(chunk_size * 2, max_chunk_size)
   else:
       chunk_size = max(chunk_size // 2, min_chunk_size)
   ```

2. **优先级调度**：优先处理短 prompt，减少 TTFT
   ```python
   # 调度策略：短 prompt 优先
   waiting_queue.sort(key=lambda r: len(r.prompt_token_ids))
   ```

3. **与投机解码集成**：分块时考虑投机解码的需求
   ```python
   # 预留空间给投机 token
   num_new_tokens -= num_speculative_tokens
   ```

4. **动态阈值**：根据请求分布动态调整 `long_prefill_token_threshold`

5. **预取优化**：在处理当前 chunk 时预取下一个 chunk 的数据

#### 面试回答模板

"Chunked Prefill 解决的核心问题是 **长 prompt 的 prefill 阶段独占批次计算预算**。传统方法一次性处理整个 prompt，会阻塞 decode 请求，导致延迟增加和 GPU 利用率不均衡。

vLLM 的实现：
1. 将长 prompt 分成多个 chunk 处理
2. 每个 chunk 与 decode 请求混合调度
3. 通过 `long_prefill_token_threshold` 控制分块大小
4. 支持 `max_num_partial_prefills` 控制并发数

它的局限性在于：
1. 额外开销（多次调度、状态管理）
2. 内存管理复杂（部分计算的 KV Cache）
3. 对短 prompt 可能增加 TTFT
4. 与投机解码的交互需要特殊处理

改进方向包括：
1. 自适应分块，根据 GPU 负载动态调整
2. 优先级调度，优先处理短 prompt
3. 与投机解码集成
4. 动态阈值调整"

### 1.5 投机解码（Speculative Decoding）

#### 解决什么问题？

**问题背景**：LLM 的 decode 阶段是 memory-bound 的，每步只能生成一个 token，效率低。

**具体痛点**：
- decode 阶段 GPU 计算利用率低，大部分时间在等待内存访问
- 每步只能生成一个 token，吞吐量受限
- 对于长序列，decode 时间很长

**解决方案**：使用小模型（或轻量级方法）快速生成多个候选 token，大模型并行验证。

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

**动态投机解码**：
```python
# vllm/v1/spec_decode/dynamic/
# 支持根据 batch size 动态调整投机 token 数
if speculative_config.num_speculative_tokens_per_batch_size:
    self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
        speculative_config.num_speculative_tokens_per_batch_size,
        vllm_max_batch_size=self.scheduler_config.max_num_seqs,
        vllm_num_speculative_tokens=self.num_spec_tokens,
    )
```

#### 局限性

1. **接受率依赖**：接受率低时，加速效果不明显甚至更慢
   - 创造性任务（如写诗）的接受率通常较低
   - 代码补全等重复性任务的接受率较高

2. **额外开销**：draft 模型的计算和内存开销
   - 需要额外的 GPU 内存存储 draft 模型
   - draft 模型的前向传播有计算开销

3. **实现复杂**：需要协调 draft 和 verify 两个阶段
   - 需要正确处理 draft token 的接受/拒绝
   - 需要维护两个模型的状态

4. **分布一致性**：需要保证输出分布与原模型一致
   - 使用 rejection sampling 保证一致性

5. **与连续批处理的交互**：投机解码影响批次大小
   - 需要为投机 token 预留空间

#### 改进方法

1. **自适应推测**：根据接受率动态调整 draft token 数量
   ```python
   class AdaptiveController:
       def adjust_num_steps(self, accept_rate):
           if accept_rate > threshold_high:
               return num_steps + 1
           elif accept_rate < threshold_low:
               return num_steps - 1
   ```

2. **NGRAM 推测**：无需 draft 模型，基于 n-gram 缓存
   - 适合重复性高的任务（如代码补全）
   - 零内存开销

3. **EAGLE 树验证**：树状 draft + 并行验证，提高接受率
   - 使用树结构生成多个候选路径
   - 并行验证所有路径

4. **热 token 映射**：缩小 draft vocabulary，减少计算量
   - 只预测最可能出现的 token

5. **与 Chunked Prefill 集成**：分块时考虑投机解码

#### 面试回答模板

"投机解码解决的核心问题是 **LLM decode 阶段的 memory-bound 特性导致的低效率**。通过小模型快速生成候选 token，大模型并行验证。

vLLM 支持多种策略：
1. EAGLE：使用 draft 模型生成候选
2. N-gram：基于 n-gram 缓存，无需 draft 模型
3. DFlash：填充式解码
4. 动态投机：根据 batch size 动态调整

它的局限性在于：
1. 接受率依赖（低接受率时加速不明显）
2. 额外开销（draft 模型的计算和内存）
3. 实现复杂（需要协调 draft 和 verify）
4. 分布一致性保证

改进方向包括：
1. 自适应推测，根据接受率动态调整
2. NGRAM 推测，无需 draft 模型
3. EAGLE 树验证，提高接受率
4. 热 token 映射，减少计算量"

---

## 第二类：实验和方案验证能力

> **核心要求**：不仅关注做了什么，更关注**怎么证明它是有效的**，追问实验细节

### 2.1 如何验证 Prefix Caching 的有效性？

#### 实验设计

**指标定义**：
- Cache Hit Rate：前缀缓存命中率
- TTFT (Time To First Token)：首 token 延迟
- Throughput：吞吐量（token/s）
- GPU Memory Usage：GPU 内存使用

**实验方案**：
```bash
# 1. 启动服务，开启 metrics
vllm serve meta-llama/Llama-3-8B \
  --enable-prefix-caching \
  --enable-metrics \
  --port 8000

# 2. 运行 benchmark，模拟多轮对话
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B \
  --enable-prefix-caching

# 3. 使用 benchmark 工具测试
python benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3-8B \
  --dataset-name sharegpt \
  --num-prompts 100

# 4. 查看 Prometheus 指标
curl http://localhost:8000/metrics | grep cache_hit_rate
```

**对比实验**：
| 配置 | Cache Hit Rate | TTFT (P99) | Throughput |
|------|---------------|------------|------------|
| 无 Prefix Caching | 0% | 500ms | 1000 token/s |
| 有 Prefix Caching | 75% | 150ms | 2500 token/s |
| + Chunked Prefill | 75% | 120ms | 2800 token/s |

#### 追问细节应对

**Q: 如何确保实验的公平性？**

A: 控制变量：
1. 相同的请求分布（使用相同的 dataset）
2. 相同的硬件环境（GPU 型号、CUDA 版本）
3. 相同的模型（相同 checkpoint）
4. 相同的并发数
5. 多次运行取平均值，计算标准差

**Q: Cache Hit Rate 是如何计算的？**

A: 在 `KVCacheManager.get_computed_blocks()` 中记录匹配的 token 数量，与总 token 数量的比值。具体实现在 `vllm/v1/metrics/prometheus.py` 中的 `vllm:prefix_cache_hits` 指标。

**Q: 为什么选择 sharegpt 数据集？**

A: sharegpt 数据集包含真实的多轮对话，有较多的共享前缀，适合测试前缀缓存的效果。如果使用随机数据，前缀缓存命中率会很低。

**Q: 如何模拟多轮对话场景？**

A: 
1. 将 sharegpt 数据集按对话轮次拆分
2. 每轮发送相同的 system prompt + 历史对话 + 新问题
3. 统计每轮的 cache hit rate

### 2.2 如何验证 Chunked Prefill 的有效性？

#### 实验设计

**指标定义**：
- TTFT (Time To First Token)：首 token 延迟
- TPOT (Time Per Output Token)：每 token 延迟
- Throughput：吞吐量（token/s）
- GPU Utilization：GPU 利用率

**实验方案**：
```bash
# 1. 基线：无 Chunked Prefill
vllm serve meta-llama/Llama-3-8B \
  --enable-chunked-prefill=False \
  --max-num-batched-tokens 8192

# 2. 实验：有 Chunked Prefill
vllm serve meta-llama/Llama-3-8B \
  --enable-chunked-prefill=True \
  --long-prefill-token-threshold 2048 \
  --max-num-partial-prefills 2

# 3. 测试长 prompt 场景
python benchmarks/benchmark_serving.py \
  --backend openai \
  --base-url http://localhost:8000 \
  --model meta-llama/Llama-3-8B \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 512
```

**对比实验**：
| 配置 | TTFT (P99) | TPOT (P99) | Throughput | GPU Util |
|------|------------|------------|------------|----------|
| 无 Chunked Prefill | 2000ms | 50ms | 800 token/s | 60% |
| Chunked (2048) | 800ms | 45ms | 1200 token/s | 80% |
| Chunked (1024) | 600ms | 42ms | 1300 token/s | 85% |

#### 追问细节应对

**Q: 为什么 Chunked Prefill 能降低 TTFT？**

A: 
1. 长 prompt 被分成多个 chunk，每个 chunk 处理时间更短
2. decode 请求可以在 chunk 之间插入，不会被完全阻塞
3. 用户更快看到第一个 token

**Q: Chunked Prefill 的开销在哪里？**

A:
1. 多次调度同一个请求的开销
2. 状态管理的开销（跟踪 prefill 进度）
3. 部分计算的 KV Cache 的内存管理

**Q: 如何选择 `long_prefill_token_threshold`？**

A:
1. 根据 prompt 长度分布选择：如果大部分 prompt 长度在 2048 左右，设置为 2048
2. 根据延迟要求选择：如果要求 TTFT < 1s，设置较小的值
3. 根据 GPU 负载选择：如果 GPU 负载高，设置较大的值

### 2.3 如何验证投机解码的加速效果？

#### 实验设计

```bash
# 基线：无投机解码
vllm serve meta-llama/Llama-3-8B \
  --speculative-model None

# 实验：EAGLE 投机解码
vllm serve meta-llama/Llama-3-8B \
  --speculative-model eagle \
  --num-speculative-tokens 5 \
  --speculative-max-model-len 4096

# 实验：N-gram 投机解码
vllm serve meta-llama/Llama-3-8B \
  --speculative-model ngram \
  --num-speculative-tokens 5 \
  --ngram-prompt-lookup-max 4
```

**指标对比**：
| 配置 | Decode Throughput | Accept Rate | 额外开销 |
|------|------------------|-------------|---------|
| 无投机解码 | 50 token/s | - | - |
| EAGLE (topk=4) | 90 token/s | 70% | 5% GPU 内存 |
| EAGLE (topk=8) | 110 token/s | 65% | 10% GPU 内存 |
| NGRAM | 80 token/s | 60% | 0% GPU 内存 |

#### 追问细节应对

**Q: Accept Rate 是如何计算的？**

A: 在 verify 阶段，统计被接受的 draft token 数量与总 draft token 数量的比值。具体实现在 `vllm/v1/spec_decode/metrics.py` 中。

**Q: 为什么 topk 越大，Accept Rate 反而越低？**

A: topk 越大，树越宽，verify 阶段需要验证更多 token，但每个分支的接受概率不变。实际上，topk 增大可以提高整体接受的 token 数量，但单个分支的接受率可能下降。

**Q: NGRAM 推测解码为什么不需要 draft 模型？**

A: NGRAM 使用 CPU 端的 n-gram 语料库，通过 BFS 搜索匹配历史 n-gram 作为 draft。适合重复性高的任务（如代码补全），但对创造性任务效果差。

### 2.4 如何验证 KV Cache 量化的有效性？

#### 实验设计

```bash
# 基线：无量化
vllm serve meta-llama/Llama-3-8B \
  --kv-cache-dtype auto

# 实验：FP8 KV Cache
vllm serve meta-llama/Llama-3-8B \
  --kv-cache-dtype fp8 \
  --quantization-param-path kv_cache_scales.json

# 实验：INT8 KV Cache
vllm serve meta-llama/Llama-3-8B \
  --kv-cache-dtype int8
```

**指标对比**：
| 配置 | KV Cache 内存 | 最大并发数 | 质量损失 | 吞吐量 |
|------|--------------|-----------|---------|--------|
| FP16 | 100% | 100 | - | 1000 token/s |
| FP8 | 50% | 200 | < 1% | 1800 token/s |
| INT8 | 50% | 200 | < 2% | 1700 token/s |

#### 追问细节应对

**Q: KV Cache 量化对模型质量的影响？**

A: 
1. FP8 量化：质量损失极小（< 1%），几乎可以忽略
2. INT8 量化：质量损失较小（< 2%），在大部分场景下可以接受
3. INT4 量化：质量损失中等（< 5%），需要根据场景选择

**Q: 如何生成 KV Cache 量化参数？**

A:
1. 使用校准数据集运行模型
2. 统计每层 KV Cache 的动态范围
3. 生成量化缩放因子
4. 保存到 JSON 文件

**Q: KV Cache 量化的开销？**

A:
1. 量化/反量化的计算开销
2. 量化参数的存储开销
3. 可能的精度损失

---

## 第三类：问题定位能力

> **核心要求**：讲清楚**遇到的问题与对应解决方案**，优化思路与排查过程

### 3.1 问题：模型上线后能力突然下降

#### 排查思路

**Step 1：确认问题范围**
- 是所有请求都下降，还是特定类型请求？
- 是突然下降，还是逐渐下降？
- 有没有发布新版本或修改配置？

**Step 2：检查权重同步**
```bash
# 检查模型是否正确加载
vllm serve meta-llama/Llama-3-8B \
  --dtype auto \
  --load-format auto

# 检查日志中的警告
grep "warning\|error" /var/log/vllm/server.log
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
--enable-prefix-caching=False
```

**Step 5：检查日志**
```bash
# 开启详细日志
--max-log-len 1000

# 检查是否有异常
grep "error\|warning\|OOM" server.log
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| 权重加载错误 | 检查 checkpoint 格式，使用 `--load-format auto` |
| 数值精度问题 | 确保 `--dtype` 与训练一致，禁用 fp8 KV Cache |
| 前缀缓存污染 | 使用 `--enable-prefix-caching=False` 排除，或使用 `cache_salt` 隔离 |
| 量化误差 | 使用原始精度模型，或调整量化参数 |
| 配置错误 | 检查所有配置参数，与预期对比 |

#### 面试回答模板

"模型上线后能力突然下降，我的排查思路是：
1. 确认问题范围：是所有请求还是特定类型，是突然还是逐渐
2. 检查权重同步：确认模型是否正确加载
3. 检查数值精度：确保 dtype 与训练一致
4. 检查前缀缓存：禁用缓存排除污染
5. 检查日志：查找异常信息

常见原因包括：权重加载错误、数值精度不匹配、前缀缓存污染、量化误差、配置错误。"

### 3.2 问题：系统上线后突然十分缓慢

#### 排查思路

**Step 1：检查 GPU 利用率**
```bash
nvidia-smi  # 查看 GPU 利用率和显存使用
```

**Step 2：检查队列状态**
```bash
curl http://localhost:8000/metrics | grep -E "num_running_reqs|num_queue_reqs"
```

**Step 3：检查关键指标**
```bash
# 检查 token usage（KV 缓存利用率）
curl http://localhost:8000/metrics | grep kv_cache_usage

# 检查是否有频繁 preemption
curl http://localhost:8000/metrics | grep num_preemptions

# 检查延迟指标
curl http://localhost:8000/metrics | grep e2e_request_latency
```

**Step 4：检查日志**
```bash
# 检查 Decode batch 日志
grep "Decode batch" server.log | tail -20

# 关注：#queue-req, token usage, cuda graph, gen throughput
```

**Step 5：检查网络**
```bash
# 多节点部署时检查网络
ping <other_node>
nc -zv <other_node> <port>
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| KV 缓存满 | 降低 `--gpu-memory-utilization`，或增大 `--max-num-seqs` |
| CUDA Graph 未启用 | 检查 `--enforce-eager`，适当增大 batch size |
| 队列过长 | 增加 DP 实例，或使用 PD 分离 |
| 频繁 preemption | 增大 `--gpu-memory-utilization`，或降低并发 |
| 网络问题 | 检查 NCCL 配置，使用 `--distributed-executor-backend ray` |

#### 面试回答模板

"系统上线后突然十分缓慢，我的排查思路是：
1. 检查 GPU 利用率：确认是否充分利用
2. 检查队列状态：确认是否有请求积压
3. 检查关键指标：token usage、preemption 频率、延迟指标
4. 检查日志：关注 Decode batch 的各项指标
5. 检查网络：多节点部署时检查网络连通性

常见原因包括：KV 缓存满、CUDA Graph 未启用、队列过长、频繁 preemption、网络问题。"

### 3.3 问题：CUDA Out of Memory (OOM)

#### 排查思路

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
--gpu-memory-utilization  # GPU 内存利用率
--max-num-batched-tokens  # 最大批次 token 数
--max-num-seqs  # 最大并发请求数
--max-model-len  # 最大模型长度
```

#### 解决方案

| 阶段 | 原因 | 解决方案 |
|------|------|---------|
| Prefill | 长 prompt | 降低 `--max-num-batched-tokens` |
| Decode | 并发过高 | 降低 `--max-num-seqs` |
| 通用 | GPU 内存利用率过高 | 降低 `--gpu-memory-utilization` |
| 模型加载 | 模型太大 | 使用 TP 并行，或使用量化 |

#### 面试回答模板

"CUDA OOM 的排查思路是：
1. 确认 OOM 阶段：是 Prefill 还是 Decode
2. 检查显存使用：实时监控显存占用
3. 检查配置：gpu-memory-utilization、max-num-batched-tokens、max-num-seqs

解决方案根据阶段不同：
- Prefill OOM：降低 max-num-batched-tokens
- Decode OOM：降低 max-num-seqs
- 通用 OOM：降低 gpu-memory-utilization
- 模型太大：使用 TP 并行或量化"

### 3.4 问题：前缀缓存命中率低

#### 排查思路

**Step 1：检查缓存配置**
```bash
# 确认前缀缓存已启用
--enable-prefix-caching=True

# 检查块大小
--block-size 16
```

**Step 2：检查请求分布**
```bash
# 分析请求是否有共享前缀
# 如果每个请求的前缀都不同，缓存命中率自然低
```

**Step 3：检查缓存大小**
```bash
# 检查 KV 缓存大小
curl http://localhost:8000/metrics | grep kv_cache_size

# 如果缓存太小，频繁驱逐会导致命中率低
```

**Step 4：检查驱逐频率**
```bash
# 检查缓存驱逐频率
curl http://localhost:8000/metrics | grep cache_evictions
```

#### 解决方案

| 原因 | 解决方案 |
|------|---------|
| 请求无共享前缀 | 无法优化，禁用缓存减少开销 |
| 缓存太小 | 增大 `--gpu-memory-utilization` |
| 频繁驱逐 | 增大缓存，或使用分层缓存 |
| 块大小不合适 | 调整 `--block-size` |
| 缓存污染 | 使用 `cache_salt` 隔离 |

#### 面试回答模板

"前缀缓存命中率低的排查思路是：
1. 检查缓存配置：确认前缀缓存已启用
2. 检查请求分布：分析是否有共享前缀
3. 检查缓存大小：确认缓存是否足够
4. 检查驱逐频率：确认是否频繁驱逐

常见原因包括：请求无共享前缀、缓存太小、频繁驱逐、块大小不合适、缓存污染。"

---

## 第四类：工程落地能力

> **核心要求**：理论结合实际，**真正落地生产价值**，关注部署、稳定性、监控、回滚

### 4.1 如何部署一个生产级的 LLM 推理服务？

#### 部署架构

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  vLLM Node 1  │    │  vLLM Node 2  │    │  vLLM Node 3  │
│  (TP=4, DP=1) │    │  (TP=4, DP=1) │    │  (TP=4, DP=1) │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────┴────────┐
                    │   Prometheus    │
                    │   + Grafana     │
                    └─────────────────┘
```

#### 部署 Checklist

**启动前检查**：
1. 模型 checkpoint 完整性
2. GPU 驱动和 CUDA 版本兼容性
3. 网络连通性（多节点部署）
4. 显存估算（模型大小 + KV 缓存）

**启动配置**：
```bash
vllm serve meta-llama/Llama-3-70B \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --long-prefill-token-threshold 2048 \
  --dtype auto \
  --load-format auto \
  --enable-metrics \
  --port 8000
```

**监控配置**：
```bash
# 启动 Prometheus + Grafana
cd examples/monitoring
docker-compose up -d
```

**关键监控指标**：
| 指标 | 告警阈值 | 说明 |
|------|---------|------|
| `vllm:num_queue_reqs` | > 2000 | 队列过长 |
| `vllm:kv_cache_usage` | > 0.95 | KV 缓存接近满 |
| `vllm:num_preemptions` | 增速 > 10/min | 调度过于激进 |
| `vllm:e2e_request_latency_seconds` P99 | > 5s | 延迟过高 |
| `vllm:prefix_cache_hits` | < 0.3 | 缓存命中率低 |

### 4.2 如何保证系统稳定性？

#### 容错机制

**健康检查**：
```bash
# 端点
GET /health

# 响应
{"status": "healthy"}
```

**请求超时**：
```bash
# 环境变量
VLLM_REQUEST_TIMEOUT=300  # 请求超时
```

**优雅关闭**：
```bash
# 发送 SIGTERM 信号
kill -TERM <pid>

# 等待正在处理的请求完成
# 超时后强制关闭
```

#### 故障恢复

**自动重试**：
```python
# 客户端重试逻辑
for attempt in range(max_retries):
    try:
        response = call_vllm(prompt)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(backoff * (2 ** attempt))
        else:
            raise
```

**灰度发布**：
1. 新版本部署到 1 个节点
2. 观察 1 小时，检查关键指标
3. 逐步扩大到所有节点
4. 保留旧版本 24 小时，以便回滚

### 4.3 如何进行性能调优？

#### 调优三板斧

**1. 最大化 batch size**
```bash
# 检查 queue-req 是否保持 100-2000
curl http://localhost:8000/metrics | grep num_queue_reqs

# 如果 queue-req 过低，说明客户端提交太慢
# 如果 queue-req 过高，需要增加 DP 实例
```

**2. KV 缓存最大化**
```bash
# 逐步增加 gpu-memory-utilization
--gpu-memory-utilization 0.85  # 起始值
--gpu-memory-utilization 0.87  # 逐步增加
--gpu-memory-utilization 0.89  # 直到 OOM 后回退
```

**3. CUDA Graph 覆盖**
```bash
# 检查 CUDA Graph 是否启用
# 默认启用，如果禁用需要检查 --enforce-eager

# 适当增大 max-num-seqs
--max-num-seqs 256  # A100
--max-num-seqs 512  # H100
```

#### 场景化调优

| 场景 | 推荐配置 |
|------|---------|
| 多轮对话 | `--enable-prefix-caching` |
| 长 prompt | `--enable-chunked-prefill --long-prefill-token-threshold 2048` |
| 低延迟 | `--max-num-batched-tokens 4096` |
| 高吞吐 | `--max-num-seqs 512 --gpu-memory-utilization 0.95` |
| MoE 模型 | `--tensor-parallel-size 8 --pipeline-parallel-size 2` |

### 4.4 如何保证数据回滚与监控？

#### 数据回滚

**请求日志**：
```bash
--max-log-len 1000
# 输出到文件
--log-requests
```

**指标导出**：
```bash
--enable-metrics
--port 8000
```

#### 监控体系

**Prometheus + Grafana**：
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

**告警规则**：
```yaml
# prometheus.yml
groups:
  - name: vllm_alerts
    rules:
      - alert: HighQueueLength
        expr: vllm:num_queue_reqs > 2000
        for: 5m
        annotations:
          summary: "Queue length is too high"

      - alert: HighPreemptionRate
        expr: rate(vllm:num_preemptions[5m]) > 10
        for: 5m
        annotations:
          summary: "Preemption rate is too high"
```

---

## 第五类：业务与实际场景理解

> **核心要求**：关注**实际场景价值和业务价值**，用户关心什么，上线成本，资源有限时优先优化什么

### 5.1 Prefix Caching 适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 多轮对话 | system prompt + 历史对话共享 | TTFT 降低 50-70% |
| Agent 工作流 | system prompt + 工具定义共享 | 吞吐提升 2-3x |
| 批量推理 | 相同 system prompt 的请求 | 吞吐提升 3-5x |
| RAG 应用 | 相同 context 的请求 | TTFT 降低 40-60% |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 纯生成任务 | 无共享前缀 | 禁用缓存减少开销 |
| 高度个性化 | 每个请求前缀不同 | 使用 `cache_salt` 隔离 |
| 实时性要求极高 | 缓存匹配有开销 | 禁用缓存，使用 FCFS |

#### 用户关心什么？

1. **延迟**：TTFT 和 TPOT 是否满足 SLA
2. **成本**：GPU 成本是否降低
3. **稳定性**：延迟是否稳定，是否有毛刺
4. **可扩展性**：能否水平扩展应对流量增长

#### 上线成本

| 成本项 | 估算 |
|--------|------|
| GPU 成本 | A100 80GB: $2-3/hour |
| 人力成本 | 1-2 周部署 + 调优 |
| 运维成本 | 监控 + 告警 + 故障处理 |
| 存储成本 | 模型 checkpoint + 日志 |

#### 资源有限时优先优化什么？

**优先级排序**：
1. **KV 缓存大小**（`--gpu-memory-utilization`）：直接影响并发能力
2. **CUDA Graph**：直接影响 decode 吞吐
3. **Chunked Prefill**（`--enable-chunked-prefill`）：影响长 prompt 处理能力
4. **Prefix Caching**（`--enable-prefix-caching`）：间接影响缓存命中率
5. **DP 实例**：水平扩展应对流量增长

### 5.2 投机解码适合什么场景？

#### 适合的场景

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 代码补全 | 高重复性，接受率高 | 2-3x 加速 |
| 多轮对话 | 历史上下文可预测 | 1.5-2x 加速 |
| 文本摘要 | 结构化输出 | 1.5-2x 加速 |

#### 不适合的场景

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 创造性写作 | 低重复性，接受率低 | 禁用投机解码 |
| 翻译任务 | 语言差异大 | 使用 NGRAM 推测 |
| 实时性要求极高 | draft 模型有开销 | 禁用投机解码 |

#### 资源有限时优先优化什么？

**优先级排序**：
1. **是否启用投机解码**：根据任务类型决定
2. **draft 模型选择**：EAGLE vs NGRAM vs MTP
3. **num_speculative_tokens**：平衡加速效果和开销
4. **自适应推测**：根据接受率动态调整

### 5.3 如何评估方案的业务价值？

#### 评估框架

**1. 技术指标**
- 延迟：TTFT、TPOT、E2E Latency
- 吞吐：token/s、requests/s
- 资源利用率：GPU 利用率、显存利用率
- 缓存命中率：Prefix Cache Hit Rate

**2. 业务指标**
- 用户满意度：NPS、CSAT
- 任务完成率：Agent 任务成功率
- 成本效率：$/request、$/token
- ROI：投入产出比

**3. 对比实验**
| 方案 | 延迟 (P99) | 吞吐 | 成本 | 业务指标 |
|------|-----------|------|------|---------|
| 基线 | 500ms | 1000 token/s | $1000/天 | NPS 70 |
| 优化后 | 200ms | 3000 token/s | $800/天 | NPS 85 |
| 改进 | -60% | +200% | -20% | +15% |

#### 面试回答模板

"评估方案的业务价值，我会从三个维度考虑：
1. 技术指标：延迟、吞吐、资源利用率、缓存命中率
2. 业务指标：用户满意度、任务完成率、成本效率、ROI
3. 对比实验：与基线方案对比，量化改进幅度

以 Prefix Caching 为例：
- 技术指标：TTFT 降低 60%，吞吐提升 200%
- 业务指标：用户满意度提升 15%，成本降低 20%
- 结论：显著的业务价值，值得投入"

---

## 附录：vLLM 关键配置参数速查

### 模型配置
```bash
--model meta-llama/Llama-3-8B  # 模型路径
--dtype auto  # 数据类型（auto/float16/bfloat16）
--load-format auto  # 加载格式
```

### 并行配置
```bash
--tensor-parallel-size 4  # TP 大小
--pipeline-parallel-size 1  # PP 大小
```

### 内存配置
```bash
--gpu-memory-utilization 0.9  # GPU 内存利用率
--max-model-len 8192  # 最大模型长度
--block-size 16  # KV Cache 块大小
```

### 调度配置
```bash
--max-num-seqs 256  # 最大并发请求数
--max-num-batched-tokens 8192  # 最大批次 token 数
--enable-chunked-prefill  # 启用分块预填充
--long-prefill-token-threshold 2048  # 长 prompt 阈值
```

### 缓存配置
```bash
--enable-prefix-caching  # 启用前缀缓存
--prefix-caching-hash-algo sha256  # 哈希算法
```

### 量化配置
```bash
--kv-cache-dtype auto  # KV Cache 数据类型
--quantization awq  # 量化方法
```

### 投机解码配置
```bash
--speculative-model eagle  # 投机解码模型
--num-speculative-tokens 5  # 投机 token 数
```

### 监控配置
```bash
--enable-metrics  # 启用指标
--port 8000  # 服务端口
```

---

**最后更新**：2026-06-27

**适用场景**：LLM 算法实习面试，针对五类技术问题的深度应对
