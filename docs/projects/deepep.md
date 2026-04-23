---
title: DeepEP v2
date: 2026-04-23
repo: "https://github.com/deepseek-ai/DeepEP"
org: "DeepSeek-AI"
tags: [deepep, moe, ep, nccl, gin, engram, pp, cp, jit, elasticbuffer]
categories:
  - moe
tier: L3
status: active
---

# DeepEP v2：EPv2 + Engram/PP/CP + NCCL GIN 后端

> DeepSeek-AI · PR [#605](https://github.com/deepseek-ai/DeepEP/pull/605) "Introducing EPv2: faster EP, and Engram/PP/CP supports" · 分支 `epv2-release` · 2026-04
> 关联阅读：[GPU-Initiated Networking for NCCL (NCCL GIN)](../papers/posts/2026-04-23-nccl-gin.md)

---

## 1. 一句话要点

DeepEP v2 是对 v1 的**完全重写**：把 HT/LL kernel 合成统一的 `ElasticBuffer`，把 RDMA 后端从 NVSHMEM/IBGDA 切到 **NCCL GIN**，用**解析 SM/QP 公式**替掉离线 autotuner，新增 **Engram / PP / CP** 三个 experimental 通信原语；同 token 规模下峰值快 1.3×、SM 省 4×，scale-out 可撑到 EP=2048。

---

## 2. v1 → v2 变化矩阵

| 维度 | v1 | v2 |
|------|----|----|
| RDMA 后端 | NVSHMEM IBGDA | **NCCL GIN**（header-only，依赖 NCCL ≥ 2.30.4） |
| API 形态 | HT buffer / LL buffer 各自 kernel | 统一 `ElasticBuffer`（内部自动 `dispatch_impl` vs `hybrid_dispatch_impl`） |
| Kernel 编译 | 部分 JIT + 预编译 | **全 JIT，per-config 生成 `.cu` → NVCC → CUBIN 缓存** |
| SM/QP 配置 | 离线 autotune | **解析公式一次算好**（见 §7） |
| Scale | EP ≤ 320 级 | **EP ≤ 2048** |
| 典型 SM 用量（V3-like 训练） | 24 SMs | **4–6 SMs** |
| 新原语 | 无 | **Engram**（RDMA sparse gather）、**PP send/recv**、**AGRS**（CP all-gather via copy engine） |
| 0-SM 变体 | 0-SM LL RDMA EP | Engram / PP with RDMA、CP with Copy Engine（**0-SM LL EP 被删**） |

性能（README verbatim，H100/B100 + CX7）：

| Arch | Topo | Dispatch BW | Combine BW | #SMs |
|------|------|-------------|------------|------|
| SM90 | EP 8×2 | 90 GB/s RDMA | 81 GB/s RDMA | 12 |
| SM90 | EP 8×4 | 61 GB/s RDMA | 61 GB/s RDMA | **6** |
| SM100 | EP 8 (NVLink) | **726 GB/s** | **740 GB/s** | 64 |
| SM100 | EP 8 (NVLink) | 643 GB/s | 675 GB/s | 24 |

依赖：Hopper 或 SM90 PTX；CUDA ≥ 12.3；PyTorch ≥ 2.10；**NCCL ≥ 2.30.4**（对应 NCCL 2.28 Device API + GIN 的正式版）。NVSHMEM 仍保留用于 `legacy` 路径，v2 主路径不再需要。

---

## 3. 源码结构：elastic / legacy / backend 三分

PR 把代码分成三块：

```
csrc/kernels/
  backend/        # nccl.cu · nvshmem.cu · cuda_driver.cu（并列共存，运行时选）
  elastic/        # v2 launcher：api / barrier / dispatch / combine /
                  #            engram / pp_send_recv（都是 .hpp 薄壳）
  legacy/         # v1 的 internode.cu / internode_ll.cu / intranode.cu 原样保留

deep_ep/include/deep_ep/
  common/         # comm.cuh · handle.cuh · layout.cuh · ptx.cuh（header-only 基础层）
  impls/          # 真实 kernel body：dispatch.cuh / combine.cuh /
                  #  hybrid_dispatch.cuh / hybrid_combine.cuh /
                  #  engram_fetch.cuh / pp_send_recv.cuh / barrier.cuh ...

deep_ep/buffers/
  elastic.py      # v2 的 ElasticBuffer：dispatch / combine / engram_* / pp_* / all_gather
  legacy.py       # v1 API 兼容层
```

关键观察：**v2 kernel body 都是 `.cuh`（header-only），没有预编译 `.cu`**——每个 kernel 要跑之前由 JIT 按 `(num_sms, num_channels, num_ranks_scaleup, num_ranks_scaleout, hidden, num_topk, ...)` 组合生成一个 `.cu` 再 NVCC 编译，CUBIN 进磁盘缓存（`kernel_signature = name$$sig$$flags$$code` 做 key，atomic rename 避免并发冲突）。

---

## 4. NCCL GIN 后端接入细节

结合前一篇 [NCCL GIN 论文笔记](2026-04-23-nccl-gin.md) 看，DeepEP v2 怎么用 GIN 有几点**和 paper 里对 v1 集成的描述明显不同**：

### 4.1 自己 `ncclCommInitRank`，不复用 PG comm

README 宣传"reuse existing NCCL communicators"，但 `csrc/kernels/backend/nccl.cu` 里实际做的是：

```cpp
ncclComm_t comm;
NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
return reinterpret_cast<int64_t>(comm);
```

Python 侧从用户 PG 取 unique_id，但**总是建新的 comm**。"复用"意义上是从同一个 unique_id 派生——形式上可以自己传 `nccl_comm` handle 进 `NCCLSymmetricMemoryContext`，但 out-of-the-box 不走这条。要和训练框架的 comm 真正共享要自己改 Python shim。

### 4.2 一个 communicator 多 context（不像 paper 描述的多 communicator）

paper 里写每 communicator 4 个 context、DeepEP 要 24 QPs 所以要开 `ceil(24/4)=6` 个 communicator + 做 `comm_id / ctx_id` 映射。v2 里：

```cpp
reqs.ginContextCount     = num_allocated_qps;       // 65 或 129
reqs.ginExclusiveContexts = true;
reqs.ginQueueDepth       = 1024;
reqs.ginTrafficClass     = sl_idx;
reqs.ginSignalCount      = num_ranks + 2*2;
reqs.ginConnectionType   = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL
                                              : NCCL_GIN_CONNECTION_FULL;
```

**直接把 context 数开到 65/129（hybrid）或 17（non-hybrid）**，single comm 内多路复用，完全跳过 paper 那个"多 communicator 拼 QP"的 workaround——说明 NCCL 2.30 的 GIN 已经放开了 context 数量限制（paper 对应的是 2.28）。

### 4.3 `CONNECTION_RAIL` vs `CONNECTION_FULL`

新拓扑 flag：`RAIL` 只沿 NIC 轨道建连，`FULL` 全互联。Hybrid 模式（scale-out>1，有 NVLink 做 forwarding）走 `RAIL`；纯 RDMA 全互联走 `FULL`。这个抽象正对 hybrid dispatch 里 "forwarder warps 先走 NVLink 再 rail-exit 出 RDMA" 的拓扑假设。

### 4.4 热路径上的 GIN 调用

Dispatch kernel 典型序列：

```cpp
// 小 put：发送端先写 per-rank token count 到接收端元数据
gin.put_value<team_t>(dst_rank_counter, rank_count[i],
                     i, ncclGinOptFlagsAggregateRequests);

// 大 put：发送 token payload
gin.put<team_t>(recv_buffer.get_token_buffer(slot).get_base_ptr(),
                send_buffer_ptr, tma_buffer.get_num_bytes<false>(),
                dst_rank_idx);

// signal：release-acquire，保证之前 put 全部到达后再通知
gin.signal(team, i, ncclGin_SignalInc{rank_idx});

// flush：本地源 buffer 可复用
ncclGin(nccl_dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA).flush(ncclCoopWarp());
```

和 paper 描述的 "zero-byte put + SignalAdd 模拟 release-acquire" 一致，但细节上 v2 的 signal 是有 payload 的 `SignalInc`，更直接。

---

## 5. Unified ElasticBuffer + Hybrid Kernel

### 5.1 warp 分工（hybrid_dispatch.cuh）

```
kNumNotifyWarps     — 归约 per-expert / per-rank token 计数并广播
kNumScaleoutWarps   — 拷 token 到 send buffer、gin.put 到远端、gin.red_add_rel 推 tail
kNumForwardWarps    — poll 远端 signal、消费 token、TMA store 给本节点 scale-up peer
```

Non-hybrid（`num_scaleout_ranks == 1`，纯节点内 NVLink）退回简单 `dispatch_impl`，只有 notify + dispatch 两组 warp。

### 5.2 warp 数公式

```cpp
// combine.hpp
num_warps = min(num_smem_bytes / token_layout_bytes, 32);
num_scaleup_warps = num_forward_warps = num_channels / num_sms;

// dispatch.hpp
num_notify_smem_bytes = align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
num_dispatch_warps = min((num_smem_bytes - num_notify_smem_bytes) / token_layout_bytes,
                         32 - num_notify_warps,
                         ceil_div(512, num_sms));
```

`ceil_div(512, num_sms)` 是 **4× SM 缩减的源头**——shared memory 和 32 warp 上限饱和后，每 SM 多塞 channel 反而更省 SM。

---

## 6. Engram / PP / AGRS 三个新原语

### 6.1 Engram（0-SM sparse gather via RDMA）

名字诱导人想 "memory trace 预存"，**实际是个 RDMA sparse gather**。用法：

```python
buffer.engram_write(local_storage)     # 把本地 slab 注册进 window
hook = buffer.engram_fetch(indices)    # 按 indices 异步 gin.get
fetched = hook()                       # wait
```

kernel（`engram_fetch.cuh`）：每 warp 协作发一个 `gin.get` 把一个 token 拉过来；follow-up kernel 只 `gin.wait()`。"0 SM" 是 GIN+NIC 包办搬字节，GPU 只敲 doorbell——并非字面 0 SM（还是会启一个 1024-thread kernel）。

**适用场景**：KV-cache remote read、embedding table gather、稀疏 attention——任何"按 index 拉远端 token"的模式。

### 6.2 PP send/recv（持久单 warp kernel）

`pp_send_recv.cuh`，`__launch_bounds__(32, 1)`，一个 warp 一个 CTA。

- `pp_send_impl`：等接收端 slot-release signal → TMA 暂存 → `gin.put` → bump send counter
- `pp_recv_impl`：等 arrival signal → TMA 到输出 → 回 slot-free signal → bump recv counter

环形 slot 管理：`slot_idx = send_count % num_max_inflight_tensors`。和 dispatch/combine 共享同一 window、同一组 QP，PP 上下文不额外建 RDMA 栈。

### 6.3 AGRS（CP all-gather / reduce-scatter via Copy Engine）

README 把它叫 "0 SM CP with Copy Engine"，代码里叫 **AGRS**（all-gather / reduce-scatter），**走 NVLink 对称内存**，**不走 RDMA**：

```python
with buffer.agrs_new_session(buffer_size, max_inflight_count):
    gathered = buffer.all_gather(local_tensor)
```

底层调 `cuda_driver::batched_write`/`batched_wait` 直接编程 copy engine。"0 SM" 在这里是真的——DMA 引擎跑，SM 不动。跟 EP 正交，另起 session、另备 buffer，所以可以和 dispatch/combine 重叠。

---

## 7. 解析 SM/QP 计算（杀掉 autotuner 的核心）

来自 `deep_ep/buffers/elastic.py` 的 `get_theoretical_num_sms`：

```python
# 每 token 期望命中的远端 expert 数（group-limited gate）
def get_expected_topk(num_groups):
    return num_groups * (1 - comb(N_exp - N_exp/num_groups, num_topk) / comb(N_exp, num_topk))

# HBM 读写算力需求
sm_read  = 1/topk + scaleout_topk/topk
sm_write = 1/topk + (1/topk)*(scaleout_topk/num_scaleout) + 1

# 线路需求
rdma_traffic   = (1/topk) * scaleout_topk * (1 - 1/num_scaleout)
nvlink_traffic = 1 - 1/num_scaleup

# 用最慢那条线定 SM，HBM 带宽匹配得上即可
bounded_gbs, bounded_traffic = argmax_by_time(rdma, nvlink)
num_sms = max(
    bounded_gbs/bounded_traffic * sm_read  / sm_read_gbs,    # 200 GB/s default
    bounded_gbs/bounded_traffic * sm_write / sm_write_gbs,   # 50  GB/s default
)
num_sms = align(max(4, ceil(num_sms * 1.25)), 2)   # 25% headroom, 偶数
```

QP 数：

```python
num_qps = num_sms * 16 + 1    if allow_hybrid_mode else min(num_sms, 8 + 1)
num_qps = min(num_qps, num_allocated_qps)     # 65 或 129 (hybrid) / 17 (non-hybrid)
```

Hybrid 下 **每 SM 驱动 16 条 channel 走独立 QP**，所以 context 数要 65/129。`+1` 是 barrier/flush 专用 QP。

QP ↔ SM 分配（`common/comm.cuh`）：

```cpp
if (kNumSMs <= kNumAvailableQPs) {
  // 每 SM 独享若干 QP，CTA-local 共享
  return {kQPStartIdx + sm_idx + ..., kSharingCTA};
} else {
  // SM 多过 QP，取模共享，grid-level 共享
  return {kQPStartIdx + (global_channel_idx % kNumAvailableQPs), kSharingGrid};
}
```

这一套公式替掉了 v1 的离线 autotune——是 v2 能往 EP=2048 扩的根本原因。

---

## 8. JIT 编译系统

`csrc/jit/` 新目录。每个 runtime class（`DispatchRuntime` / `CombineRuntime` / `EngramFetchRuntime` / `PPSendRuntime` / ...）继承 `jit::LaunchRuntime<Args>`，实现 `generate_impl()` 返回一段 `.cu` 源码——内容就是 `#include <deep_ep/impls/XXX.cuh>` + 用编译期参数模板实例化一个 kernel。

cache key：

```
kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code)
```

所以**改 kernel name / 编译器版本 / 编译 flag / 生成的源文本 任何一项都触发重编译**。磁盘 cache + atomic rename，多进程并发安全。

代价：首次调用某 config 组合会卡住做 NVCC 编译（典型几秒到十几秒），之后从缓存走。好处：每 config 的 kernel 里所有维度都是编译期常量，寄存器分配更紧。

---

## 9. Barrier with Timeout + PTX trap

`common/comm.cuh` 三种 barrier：

- `nvlink_barrier_wo_local_sync` — 纯 NVLink + workspace atomic
- `gin_barrier_wo_local_sync` — `gin.signal` + shadow pointer poll **带超时**
- `scaleup_barrier_wo_local_sync` — 视拓扑选一个

`gpu_barrier()` 分两层：SM0 做 scale-up，其余 SM 做 scale-out，grid sync 串联。**超时到阈值直接 PTX `trap` 挂掉**——v1 卡死的毛病从此有 fail-fast。是个实打实的容错改进。

---

## 10. 我的评估

### 10.1 真正的架构级变化

1. **NCCL GIN 取代 NVSHMEM/IBGDA 做主 RDMA 路径**——部署依赖少一套 runtime，训练框架和 DeepEP 共享 NCCL 基础设施；验证了 NCCL GIN paper 里 "生态统一" 的承诺可落地。
2. **Unified ElasticBuffer + hybrid 分 warp**——是 4× SM 缩减的物理来源，不是包装层改动。
3. **Analytical SM/QP 公式替代 autotuner**——是 EP=2048 scale 的使能。
4. **Per-config JIT**——每次调用编译期常量化所有维度，寄存器和 shmem 布局最优化；代价是首次编译延迟。
5. **AGRS with Copy Engine**——CP all-gather 走 copy engine 而不是 RDMA，这是全新的通信路径（不是改写旧路径）。

### 10.2 增量改良（没那么颠覆）

6. **Engram** = `gin.get` + batcher + `wait()`，名字浪漫，实现小巧。价值在于复用 GIN 管道给用户一个稀疏 gather 原语。
7. **PP send/recv** = 持久单 warp kernel，复用 dispatch/combine 的 window + QP，省了一套独立 RDMA 栈。
8. **Barrier with timeout+trap** = 生产级容错补丁，不是新机制。

### 10.3 使用者要注意的坑

- **NCCL ≥ 2.30.4** 刚发出来不久，供应链风险：NCCL 版本锁得很死；一旦自家训练栈被钉在老版本，v2 用不了。
- "reuse existing NCCL communicator" 目前**只能从 unique_id 派生**，想和 PG 的 `ncclComm_t` 真共享得自己改 shim——还没达到 paper 里宣传的"直接挂训练框架 comm"的理想形态。
- **Buffer 占用显著增大**：hybrid 的 4D `dst_buffer_slot_idx { channel × scaleout_rank × max_tokens_per_channel × topk }` + scale-up 收发 + scale-out 收发，EP=2048 时不可忽略——README 自己承认。
- **0-SM 有水分**：Engram/PP 都还是小 kernel（1024 thread 或 1 persistent warp），只有 AGRS 是字面 0 SM。
- **0-SM RDMA LL EP 被砍**：v1 有、v2 没，decode-only 推理部署若依赖这条要再评估。

### 10.4 对我工作的启示

- **AFD / disagg**：Engram（RDMA sparse gather）语义正对 KV 跨实例读取，比 NIXL 那套 block-level transfer 粒度更细；值得把 Engram 拆出来单用。
- **PP 共享 QP 池**：给 AFD 里 Attention↔FFN 的逐层通信提供了一个参考——PP send/recv 和 dispatch/combine 用同一 window、同一组 QP，AFD 的 comm 路径也可以这么收敛。
- **AGRS copy engine 路径**：CP 不走 RDMA 改走 copy engine 是个很聪明的架构选择，思路可以推广——节点内任何对称内存的大块搬运都可以绕 SM。
- **Analytical 公式 vs autotune**：disagg serving 的 SM/QP 配置也值得做 analytical 化，autotune 本身在 scale 大了之后 cost 爆炸。

### 10.5 相对 NCCL GIN paper 的实证价值

Paper 里是 DeepEP v1 的集成数据，描述了 "每 comm 4 context → 多开 comm" 的 workaround。DeepEP v2 直接跳过这个 workaround，说明 **NCCL 2.30 的 GIN 已经放开 context 限制**（paper 对应的是 2.28）。可见 GIN 仍在快速演进，paper 里列的 "当前限制" 条目在 v2 时间点已经被消化掉了两条（context 数量、对称 window 大小相关的部分）。

---

## 11. 代码导航

| 关心什么 | 去哪 |
|----------|------|
| README / 特性矩阵 / 性能表 | `README.md` |
| NCCL GIN 后端 | `csrc/kernels/backend/nccl.cu`、`deep_ep/include/deep_ep/common/comm.cuh` |
| v2 dispatch/combine 内核 | `deep_ep/include/deep_ep/impls/{dispatch,combine,hybrid_dispatch,hybrid_combine}.cuh` |
| Engram / PP | `deep_ep/include/deep_ep/impls/{engram_fetch,pp_send_recv}.cuh` |
| AGRS / CP | `deep_ep/buffers/elastic.py::all_gather / agrs_*` + `csrc/kernels/backend/cuda_driver.cu` |
| 解析 SM/QP 公式 | `deep_ep/buffers/elastic.py::get_theoretical_num_{sms,qps}` |
| JIT 系统 | `csrc/jit/` + 各 runtime 的 `generate_impl()` |
| Barrier | `deep_ep/include/deep_ep/impls/barrier.cuh` + `common/comm.cuh` |
| v1 兼容 | `csrc/kernels/legacy/` + `deep_ep/buffers/legacy.py` |

相关笔记：[通信与网络 →](../communication/index.md) · [MoE Systems →](../moe/index.md) · [NCCL GIN paper →](../papers/posts/2026-04-23-nccl-gin.md)
