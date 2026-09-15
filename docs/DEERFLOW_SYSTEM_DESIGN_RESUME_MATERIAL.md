# DeerFlow 系统设计与实现说明（简历素材版）

> 用途：作为后续生成简历项目经历、面试讲稿和项目介绍的事实底稿。
>
> 使用前请补充“个人职责与成果”。本文描述的是当前仓库具备的系统能力，不应默认全部写成个人独立完成，也不要虚构业务指标。

## 1. 项目定位

DeerFlow 是一个基于 LangGraph/LangChain 的全栈 AI Agent Runtime。它不是单一聊天机器人，而是把模型调用、工具执行、任务规划、子 Agent、Sandbox、长短期记忆、Skills、MCP、流式传输、持久化和多渠道接入统一到同一套线程与 Run 生命周期中。

当前项目额外实现了一套 Skill 自进化系统：从真实 Agent Run 中采集可验证证据，经过确定性压缩、结果验证、结构化语义提取、跨轨迹聚类和 Skill 蒸馏，最终生成或更新 Skill。当前部署使用 direct publication 模式，在达到聚类阈值后跳过在线 Replay Evaluation 和人工审批，直接进入受安全校验和版本控制保护的发布流程。

## 2. 解决的问题

- 模型、工具、上下文、文件系统和执行环境耦合，难以扩展和替换。
- 长任务产生大量消息与工具结果，容易超过上下文窗口。
- Agent Run 是异步长事务，需要处理流式输出、断线重连、取消、恢复和并发冲突。
- 工具和 Skill 来源复杂，需要统一权限、隔离、审计和安全检查。
- 单次任务中的成功经验只停留在对话历史，不能自动沉淀为可复用能力。
- 直接让 LLM 总结轨迹容易产生幻觉、泄漏敏感信息，且输出不可复现。

核心目标是把 Agent 从一次性模型调用提升为可持久化、可扩展、可恢复、可审计，并能从真实执行轨迹中持续进化的运行系统。

## 3. 总体架构

    Browser / IM / Python Client
                |
                v
          Nginx :2026
           /          \
          v            v
    Next.js :3000   FastAPI Gateway :8001
                        |
                        +-- LangGraph-compatible API
                        +-- RunManager / Run Worker
                        +-- StreamBridge / SSE
                        +-- REST Management APIs
                        |
                        v
                 LangGraph Lead Agent
                 /       |        \
                v        v         v
            Toolset   Middleware  Subagents
                |        |         |
                +--------+---------+
                         |
                Sandbox / Skills / MCP
                         |
              SQLite or PostgreSQL / Redis

### 3.1 服务拓扑

| 服务 | 默认端口 | 职责 |
|---|---:|---|
| Nginx | 2026 | 统一入口，转发前端、REST 和 LangGraph-compatible API |
| Gateway | 8001 | FastAPI、Agent Runtime、Run 生命周期、SSE、持久化和管理 API |
| Frontend | 3000 | Next.js/React 对话与工作区界面 |
| Provisioner | 8002 | 可选的远程/Kubernetes Sandbox 资源管理 |

Gateway 内嵌 Agent Runtime，不需要额外部署 LangGraph Server；同时保留 LangGraph SDK 兼容接口和 langgraph.json，便于 Studio 或外部 LangGraph 客户端接入。

### 3.2 Monorepo 分层

- backend/packages/harness/deerflow：可复用 Agent Harness，负责 Agent、工具、Sandbox、Skills、Memory、MCP、Subagent 和 Runtime。
- backend/app：应用层，负责 FastAPI Gateway、REST 路由和 IM Channel。
- frontend：Next.js App Router 前端。
- skills：公共、自定义和集成 Skill。
- contracts：跨模块 JSON Contract。
- docker、scripts：部署、启动、诊断和运维脚本。

Harness 不允许反向依赖 app，并通过测试固化依赖方向，避免框架层与应用层循环耦合。

## 4. Agent 运行链路

    创建 Run
      -> RunManager 做线程级并发控制和状态持久化
      -> Worker 构造 RunnableConfig 与运行上下文
      -> make_lead_agent 动态组装模型、工具、Skill 和 Middleware
      -> LangGraph agent.astream 执行模型/工具循环
      -> RunJournal 记录消息、工具、Token 和审计事件
      -> StreamBridge 发布 values/messages/custom 事件
      -> SSE Consumer 推送前端或 IM Channel
      -> 持久化终态、交付回执、轨迹和检查点

### 4.1 Run 生命周期

RunManager 统一管理 queued、running、success、error、interrupted 等状态，并处理：

- 同一线程只允许一个活动 Run 的数据库约束。
- interrupt/rollback 等多任务策略。
- 多 Worker 下的 owner lease、heartbeat、接管和孤儿 Run 恢复。
- 非 owner Worker 接收取消请求后，将请求持久化并交给实际 owner 执行。
- Run 结束后的 Token、状态、交付回执和线程状态同步。

### 4.2 Lead Agent 动态组装

make_lead_agent 根据请求上下文和配置动态决定：

- 使用哪个模型以及是否开启 thinking/reasoning。
- 是否启用 Plan Mode 和 write_todos。
- 是否启用 Ultra Mode 和 task 子 Agent。
- 当前自定义 Agent 的工具组、Skill allowlist 和模型覆盖。
- 是否暴露视觉、Memory、MCP、ACP、Skill 管理等条件工具。
- Authorization/Guardrail 是否允许模型看到和执行某个工具。

模型通过反射式工厂加载，配置可切换 OpenAI-compatible、Anthropic、DeepSeek、Google、Ollama 等 LangChain ChatModel 实现。

### 4.3 Middleware 体系

系统使用 30+ 个有序 Middleware 处理横切能力，主要包括：

- 输入与远程工具结果清洗。
- 工具输出预算控制和大结果外置。
- 用户/线程目录初始化与上传文件注入。
- Sandbox 获取、回收和审计。
- Read-before-write 文件写入保护。
- 工具异常标准化、无进展检测和循环检测。
- Skill 激活、动态工具策略和持久上下文投影。
- 对话摘要、Todo、Token 统计、标题和 Memory 更新。
- MCP 延迟工具过滤和按路由自动提升。
- 子 Agent 并发与总量限制。
- 模型长度、安全终止、空响应恢复和用户澄清。

Middleware 顺序本身是系统契约。例如 Skill 激活必须先于 Skill 工具策略，持久上下文捕获必须先于摘要压缩，Clarification 必须位于尾部以截获工具调用并中断图执行。

## 5. 工具与扩展系统

工具由四类来源统一组装并按名称去重：

1. config.yaml 中声明的 Python/LangChain 工具。
2. DeerFlow 内置工具。
3. MCP Server 动态发现的工具。
4. 外部 ACP Agent 适配工具。

当前配置实际启用 14 个基础工具：

- 网络：web_search、web_fetch、image_search。
- 文件：ls、read_file、glob、grep、write_file、str_replace。
- 执行：bash。
- 内置：present_files、ask_clarification、review_skill_package、list_uploaded_files。

Pro 模式额外注入 write_todos；Ultra 模式再注入 task。当前 Qwen3 模型不支持视觉，所以不注入 view_image。

### 5.1 MCP

框架支持 stdio、HTTP/SSE MCP Server、OAuth、工具名前缀、调用超时、自定义拦截器和路由提示。MCP 工具可以全部绑定，也可以通过 tool_search 延迟暴露 Schema，减少大量工具对模型上下文和工具选择准确率的影响。

当前实例没有 extensions_config.json，因此没有实际启用 MCP Server；tool_search 也处于关闭状态。这属于“框架已支持、当前部署未接入”的能力。

### 5.2 Skill

Skill 是带 Frontmatter 的 SKILL.md 能力包，可附带 references、scripts 和 assets。系统支持：

- 公共 Skill、用户自定义 Skill、历史 Skill 和托管集成 Skill。
- /skill-name 显式激活。
- 延迟发现与 describe_skill。
- Skill 级 allowed-tools，在模型 Schema 可见性和执行阶段双重限制。
- Skill 安装、启停、版本历史、安全扫描和 Sandbox 投影刷新。

Skill 正文按需加载，不直接永久塞入基础系统 Prompt，以降低上下文成本。

## 6. Sandbox 与文件隔离

系统定义抽象 SandboxProvider/Sandbox 接口，文件工具和 Bash 面向统一虚拟路径工作：

    /mnt/user-data/workspace
    /mnt/user-data/uploads
    /mnt/user-data/outputs
    /mnt/skills

底层可切换 Local、Docker/AIO、E2B、Boxlite、Tenki 或 Provisioner/Kubernetes 等实现。线程数据按 user_id/thread_id 隔离，防止跨用户或跨会话文件串读。

关键实现包括：

- 路径规范化、虚拟路径映射和目录边界校验。
- read_file、write_file、str_replace 等统一工具协议。
- 写前读取哈希校验，防止模型基于旧内容覆盖并发修改。
- Shell 高风险命令审计与工具结果大小限制。
- 输出文件必须位于 outputs 目录，并通过 present_files 和 durable delivery receipt 完成交付闭环。

当前开发配置使用 LocalSandboxProvider 且显式开启 host bash，适合可信单用户本地环境，但不是生产级安全隔离。生产部署应使用容器或远程 Sandbox。

## 7. 流式传输设计

系统保留两条不同消费者模型的流式路径：

| 路径 | 执行模型 | 消费者 | 传输方式 |
|---|---|---|---|
| Gateway | async | Web、IM、LangGraph SDK | agent.astream -> StreamBridge -> SSE |
| DeerFlowClient | sync | Python/Jupyter/测试 | agent.stream -> generator yield |

LangGraph 的三类核心事件：

- values：节点完成后的完整 State 快照。
- messages：模型 Token 增量和工具消息。
- custom：任务进度、子 Agent 生命周期等应用事件。

Gateway 使用 Memory 或 Redis StreamBridge 解耦 Agent Producer 和 SSE Consumer，支持多订阅者、心跳和 Last-Event-ID 恢复。保留窗口被裁剪时发送显式 gap 事件，客户端重新加载持久化状态，而不是静默消费不完整历史。

前端不会订阅 Subgraph 的完整状态；子 Agent 进度通过根命名空间的 task_* custom event 返回，避免子图状态覆盖主线程视图。

## 8. 状态与持久化

### 8.1 ThreadState

在 LangGraph AgentState 基础上扩展：

- sandbox、thread_data、uploaded_files。
- artifacts、viewed_images、todos。
- goal、delegations、skill_context。
- summary_text、线程标题和延迟工具提升状态。

自定义 reducer 负责去重、合并、终态不可降级和容量上限。

### 8.2 Checkpoint

支持 Memory、SQLite 和 PostgreSQL Checkpointer。消息 Channel 支持：

- full：每个 Checkpoint 保存完整消息。
- delta：保存增量写入，并按配置周期生成完整快照，降低长会话 O(N²) 存储放大。

模式在进程内冻结并带兼容性检查，避免使用 full 模式误读 delta 数据。回滚、上下文压缩和 regenerate 通过统一 State Accessor 和 Mutation Graph 写入，避免直接操作底层 Checkpoint 破坏 LangGraph 继承关系。

### 8.3 当前存储配置

- 主数据库：SQLite。
- Checkpoint Channel：full。
- RunEvent：数据库持久化，避免服务重启后丢失进化轨迹。
- Agent 定义：文件存储。
- 多 Worker 扩展：支持 PostgreSQL、Redis StreamBridge 和 lease-based ownership。

## 9. Memory 与上下文工程

当前启用 DeerMem，运行在 middleware 模式：

- 每轮结束后异步提取用户摘要、历史摘要和 Agent 事实。
- 用户全局摘要保存在 JSON，Agent Facts 使用独立 Markdown 文件。
- 使用锁、revision、journal 和原子替换避免并发丢失更新。
- 默认通过 SQLite FTS5/BM25 建立可重建的派生索引。
- Prompt 注入有 Token 上限；长对话由 SummarizationMiddleware 压缩为 summary_text + recent messages。

系统也支持 tool 模式，向模型暴露 memory_search/add/update/delete，但当前部署未启用该模式。

## 10. 子 Agent 系统

Ultra 模式通过 task 工具把边界明确的任务委派给子 Agent。设计重点包括：

- Provider tool_call_id 用于消息、SSE 和前端关联。
- 服务端独立生成全局 execution_id，用于进程注册、取消、超时和清理。
- 子 Agent 拥有独立上下文和执行预算，禁止嵌套调用 task。
- 支持同一响应内有限并发和单 Run 总委派上限。
- 步骤事件批量写入 RunEventStore，避免深任务逐条写库。
- 工具结果中的 Token Usage 回传主 Agent，并通过消息标记保证幂等计费。

路由策略强调“收益驱动”：只有并行延迟收益、专业能力或上下文隔离收益明显时才委派；存在数据依赖或共享可变状态时禁止并行。

## 11. Skill 自进化系统

### 11.1 核心链路

    RunEvent
      -> EvolutionTraceSnapshot
      -> OutcomeEvidence
      -> EligibilityDecision
      -> DeterministicExtraction
      -> EvolutionEvent
      -> Candidate Retrieval / Cluster Confirmation
      -> EvolutionCluster (K >= 3 independent runs)
      -> SkillProposal (create or patch)
      -> Direct Publication
      -> Version Snapshot / Rollback

### 11.2 轨迹采集与确定性压缩

Run 结束后，Worker 从持久化 RunEvent 中构建 EvolutionTraceSnapshot：

- 源读取上限为 10,000 条事件。
- 快照默认保留第一条可见任务输入和最新事件尾部，总计最多 256 条。
- 根据 tool_call_id 配对模型工具调用和工具结果。
- 提取任务输入、最终回答、用户纠正、工具状态、Skill 使用和 Artifact。
- 过滤隐藏上下文、Summary 和非 Lead Agent 消息。
- 请求密钥按值脱敏，敏感字段按 key 脱敏，宿主路径映射为虚拟路径。
- 已读取的 SKILL.md 正文不进入快照，只保留路径和 SHA-256。
- 使用稳定 JSON 排序与 SHA-256 生成可复现 snapshot_hash。

这一阶段不调用 LLM。它本质上是有界、可重放、可审计的确定性投影，而不是自然语言摘要。

### 11.3 结果验证与准入

OutcomeVerifier 使用 Python 规则聚合运行终态、测试命令退出码、Artifact 校验、环境奖励和显式用户反馈。模型在最终回答中声称“任务成功”不属于权威证据。

Eligibility 要求：

- 结果为高置信成功，当前阈值为 0.8。
- 命中复杂度信号，例如工具调用达到 5 次、经历可恢复错误、用户纠正、非平凡工作流或显式记忆请求。
- 排除安全截断、资源上限等不适合学习的终止原因。

不满足成功证据，或既不满足复杂度信号也没有明确记忆请求的任务，不会进入后续高成本 LLM 流水线。

### 11.4 确定性证据选择

pre_extract_evolution 再把 Snapshot 压成供 LLM 使用的证据集合：

- 工具片段最多 32 条，优先保留失败工具，再用最新工具补足。
- 用户纠正最多 8 条，Artifact 片段最多 16 条，采用头尾采样。
- 单候选片段最多 2,000 字符。
- 工具参数最多 700 字符，结果最多 1,000 字符。
- Base64、Data URI 和疑似二进制内容省略正文，只保留哈希。
- 完整工具顺序仍保留结构、状态和参数/结果哈希。
- 每个片段有稳定 segment_id 和 content_hash。

### 11.5 结构化 LLM 提取

LLM 从这一阶段才开始介入，负责把有限证据转换为 EvolutionEvent：

- 任务目标与任务签名。
- 成功执行路径。
- 失败尝试和对应经验。
- 用户纠正及有效变化。
- 可复用经验。
- 已使用 Skill 的能力缺口。
- 新 Skill 或 Patch 的候选目标。

模型必须输出严格 JSON，并且每项结论引用已有 segment_id。系统通过 Pydantic Schema、语义约束和证据引用校验拒绝幻觉；无对应失败 Tool 或 Correction 证据的可选结论会被丢弃。

### 11.6 聚类与 K 次确认

EvolutionEvent 先使用确定性任务指纹分组，再可选使用 Ollama Embedding 和 Qdrant 做语义候选召回，最后由结构化 LLM 确认是否属于同一类可复用任务。

当前阈值：

- 至少 3 个确认事件。
- 至少来自 3 个独立 Run。
- 单 Cluster 最多保存 20 个事件。
- 不允许存在冲突或尚未确认的候选。

因此单次偶然成功不能直接修改 Skill，必须积累跨任务、跨 Run 的重复证据。

### 11.7 Skill 蒸馏

Ready Cluster 分为两条分支：

- Create：生成新的完整 SKILL.md 和可选支持文件。
- Patch：解析源 Run 实际使用的 Skill Hash，针对当前版本生成证据关联的 Patch。

蒸馏阶段要求每条指导可追溯到多条事件证据，过滤一次性观察；路径穿越、绝对路径、保留路径和不安全支持文件在进入发布前被拒绝。

### 11.8 当前 Direct Publication 策略

系统同时实现了 Replay Evaluation、质量评分、人工审批和自动审批基础设施，但当前配置是 skill_evolution.publication.mode = direct。

Direct 模式的语义是：Cluster 达到 K=3 并完成 Proposal 蒸馏后，跳过在线 Replay Evaluation 和 Approval，直接发布真实 Skill。这样降低在线链路时延和环境复现成本。

Direct 不等于无保护。发布仍执行：

- Skill 名称、包结构和路径校验。
- Native SkillScan。
- LLM 安全审核，失败时 fail closed。
- Base Skill Hash 和 Package Hash 的 CAS 校验。
- 每用户、每 Skill 串行化。
- 临时目录 staging、原子目录替换和失败恢复。
- 发布前后完整二进制安全 Snapshot。
- 版本历史和精确 Rollback。

Direct 模式还会从 Lead Agent 中移除 skill_manage，避免在线 Agent 绕过 K-ready 后台流程直接修改 Skill。

### 11.9 Durable Coordinator

自进化流程运行在请求链路之外：

- Run 终态先幂等写入 Snapshot，再持久化 EvolutionJob。
- SQL Job 是事实来源，内存 Queue 只负责唤醒。
- Job 使用确定性幂等键、claim token、revision CAS 和可续期 lease。
- 支持有限并发、指数退避、启动恢复和有界 shutdown drain。
- 旧 Worker 不能提交已被新 Worker 接管的结果。
- 进化失败为 fail-open，不改变已经完成的用户 Run 结果。

## 12. 安全与可靠性设计

- 所有用户资源以 user_id 分区，线程文件再以 thread_id 隔离。
- Authorization 在工具组装时过滤 Schema，在执行前再次检查。
- Guardrail、SandboxAudit、路径校验和 Sandbox 形成分层防护。
- 远程网页内容和用户输入会中和伪造的系统标签与边界标记。
- 大工具输出外置到文件，只把摘要和读取引用放回上下文。
- Skill 写入统一经过 SkillMutationService，禁止绕过扫描、锁和历史。
- 数据库结构由 Alembic 管理；SQLite 用于单节点，PostgreSQL 用于多 Worker。
- 关键写入采用幂等键、唯一索引、CAS、原子替换和 lease，而不是依赖进程内锁。
- RunEvent、Checkpoint、业务表和文件 Artifact 分层持久化，各自承担审计、恢复或大对象存储职责。

## 13. 当前部署快照

以下是当前工作区配置，不代表框架能力上限：

| 项目 | 当前值 |
|---|---|
| 主模型 | 本地 Ollama qwen3:8b |
| 模型上下文 | 40,960 tokens |
| Thinking | 支持并默认可启用 |
| Vision | 不支持 |
| Sandbox | LocalSandboxProvider，允许 host bash |
| Database | SQLite |
| Checkpoint | full mode |
| RunEvent | DB 持久化 |
| Memory | DeerMem middleware mode |
| MCP | 框架支持，当前未配置 Server |
| Skill Evolution | enabled |
| 聚类阈值 | 3 events / 3 distinct runs |
| 发布模式 | direct |
| Embedding | 配置启用 Ollama nomic-embed-text，服务不可用时降级 |
| Vector Store | 配置启用 Qdrant，服务不可用时降级 |

## 14. 技术栈

- 后端：Python 3.12、FastAPI、LangGraph 1.2、LangChain 1.3、Pydantic 2、SQLAlchemy 2、Alembic。
- 前端：Next.js 16、React 19、TypeScript 5、LangGraph SDK。
- 数据：SQLite/PostgreSQL、Redis Streams、SQLite FTS5、Qdrant。
- 模型：Ollama/Qwen，以及可插拔的 OpenAI-compatible、Anthropic、DeepSeek、Google 等模型。
- 扩展：MCP、ACP、Python Plugin、Agent Middleware、Skills。
- 执行隔离：Local、Docker/AIO、E2B、Boxlite、Tenki、Kubernetes Provisioner。
- 工程化：pytest、Ruff、pnpm、Docker Compose、Nginx、SSE、结构化日志和 LangSmith/Langfuse/Monocle tracing。

## 15. 可用于简历的技术亮点

以下句子是候选素材，只保留本人真实负责和能够在面试中解释源码的部分：

1. 设计并实现基于 LangGraph 的全栈 Agent Runtime，将模型、工具、Sandbox、Memory、Skills 和 Subagent 统一到线程级 Run 生命周期，并通过 FastAPI/SSE 向 Web 与 IM 客户端提供实时执行状态。
2. 构建基于 RunEvent 的 Skill 自进化闭环，通过确定性脱敏压缩、规则化结果验证、严格 Schema 的 LLM 语义提取和跨 Run 聚类，将真实成功轨迹沉淀为可复用 Skill。
3. 设计 K=3 独立运行确认机制和 Durable Evolution Coordinator，使用幂等键、数据库 lease、revision CAS、指数退避与启动恢复，确保后台进化任务可重试且不影响在线用户请求。
4. 实现安全的 Skill 发布与回滚机制，通过 SkillScan、LLM moderation、包级 Hash CAS、原子目录替换和二进制安全版本快照，支持已有 Skill Patch 和新 Skill 创建。
5. 设计 Agent 流式传输层，基于 Memory/Redis StreamBridge 支持多订阅者、心跳、SSE 断线续传和显式 gap 恢复，并隔离主 Agent 与 Subgraph 状态流。
6. 构建动态工具与权限体系，统一加载配置工具、内置工具、MCP 和 ACP，通过组装期 Schema 过滤、执行期授权、Skill allowed-tools 和延迟工具发现控制模型权限与上下文成本。
7. 实现线程级文件隔离和可插拔 Sandbox，统一虚拟路径、写前读取校验、命令审计、Artifact 交付回执，并兼容本地、容器和远程执行环境。
8. 优化长会话状态管理，引入摘要持久上下文和 full/delta Checkpoint 模式，避免消息历史增长导致的上下文和存储放大问题。

## 16. 面试展开建议

建议重点准备以下三个故事，每个都按“问题 -> 约束 -> 方案 -> 取舍 -> 验证”展开。

### 故事 A：为什么轨迹压缩不用 LLM

- 问题：原始轨迹大、含敏感信息，直接让 LLM 总结不可复现且可能幻觉。
- 方案：固定事件选择、工具配对、字段脱敏、长度预算、稳定 Hash。
- 取舍：会损失预算外语义，因此保留 truncated，再由后续有限证据上的 LLM 做语义提取。
- 结果：同一输入产生相同 Snapshot，可审计、可重试、可做证据引用校验。

### 故事 B：为什么自进化采用 K 次确认

- 问题：单次任务可能是偶然成功，直接写 Skill 容易过拟合和污染能力库。
- 方案：任务指纹、语义召回、结构化确认、3 个独立 Run 门槛。
- 取舍：学习速度变慢，但提高可复用性和证据质量。
- 当前策略：选择 direct publication 降低 Replay 成本，但保留安全扫描、CAS、原子发布和回滚。

### 故事 C：长任务如何保证可靠性

- 问题：Agent Run 可能断连、取消、跨 Worker、服务重启或产生长时间后台任务。
- 方案：RunManager + durable state + StreamBridge + owner lease + idempotency + recovery。
- 取舍：单机模式简单；多 Worker 需要 PostgreSQL、Redis 和统一 ownership 配置。

## 17. 需要本人补充的数据

- 本人负责的模块：[填写]
- 项目周期和团队规模：[填写]
- 代码量或核心 PR 数量：[填写]
- 实际部署环境和用户规模：[填写]
- 日均 Run 数、并发量、平均任务时长：[填写]
- 优化前后的时延、成功率、Token、存储或成本变化：[填写]
- 线上故障、技术债或性能问题及解决结果：[填写]
- 是主导架构、独立实现、协作实现还是负责其中一段：[填写]

如果没有可信业务指标，只写架构事实，例如“支持 3 次独立 Run 聚类确认”“轨迹最多 256 事件”“工具证据最多 32 段”，不要把配置上限包装成业务收益。

## 18. 给 GPT-6 的建议提示词

    请根据我提供的 DeerFlow 系统设计材料，为我生成中文技术简历中的一段项目经历。

    目标岗位：LLM Agent 平台 / AI Infra / Python 后端研发
    项目角色：[填写]
    项目周期：[填写]
    团队规模：[填写]
    本人真实负责范围：[填写]
    可信量化成果：[填写，没有则明确写“无”]

    要求：
    1. 先输出 1 句项目简介，再输出 4-6 条项目职责与技术成果。
    2. 每条采用“技术问题 + 我的方案 + 可验证结果”的结构。
    3. 突出 LangGraph Agent Runtime、Run 生命周期、流式传输、Sandbox、Skill 自进化和可靠性设计。
    4. 不要把整个仓库能力默认写成我个人完成，只使用“本人真实负责范围”中的内容。
    5. 不虚构 QPS、用户数、性能提升百分比或商业收益。
    6. 可以使用系统中的真实架构参数作为技术复杂度证据，但不要把配置上限写成线上指标。
    7. 语言简洁、专业，避免“负责了”“参与了”这类空泛表达。

    以下是系统设计事实材料：
    [粘贴本文第 1-16 节]

## 19. 关键代码索引

- Agent 组装：backend/packages/harness/deerflow/agents/lead_agent/agent.py
- 工具组装：backend/packages/harness/deerflow/tools/tools.py
- Run 生命周期：backend/packages/harness/deerflow/runtime/runs/
- RunEvent：backend/packages/harness/deerflow/runtime/events/、runtime/journal.py
- 流式桥接：backend/packages/harness/deerflow/runtime/stream_bridge/
- Sandbox：backend/packages/harness/deerflow/sandbox/、community/*sandbox*/
- Subagent：backend/packages/harness/deerflow/subagents/
- Memory：backend/packages/harness/deerflow/agents/memory/
- Skill 加载与存储：backend/packages/harness/deerflow/skills/
- Skill 自进化：backend/packages/harness/deerflow/skill_evolution/
- 自进化持久化：backend/packages/harness/deerflow/persistence/skill_evolution/
- Gateway：backend/app/gateway/
- 前端：frontend/src/

## 20. 事实边界

- 当前配置使用 direct publication，因此在线自进化主链路不执行 Replay Evaluation 和人工 Approval；这些模块已实现，但属于备用或实验能力。
- 当前没有配置 MCP Server，不能描述为“线上已接入 MCP 生态”，只能描述为“实现 MCP 扩展框架”。
- 当前使用 LocalSandbox 并允许 host bash，不能描述为“当前环境具备强容器隔离”；容器和远程 Sandbox 是框架可选能力。
- 当前数据库是 SQLite、StreamBridge 默认是进程内模式；PostgreSQL/Redis 是多 Worker 扩展方案。
- 本文没有业务规模和性能收益数据，正式简历必须由本人补充并核实。
