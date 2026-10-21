# 森语 · 森林康养基地智能咨询与地图路线规划

基于 Vue 3、FastAPI、LangChain 1.x（`create_agent` / LangGraph）、RAG（Chroma）与 MySQL 的森林康养环境咨询原型，示范对象为福州国家森林公园。项目提供设施/分区问答、森林与一般健康科普、读者注册登录、管理员账号与质检管理、多模型切换、虚拟二维基地分区路线规划、SSE 流式回答和滚动摘要记忆。

在 Agent 能力上，项目内置 5 个工具（知识检索、虚拟地图、疗养路线推荐、步道计算、分区查询），支持同一轮组合调用并在聊天区直接渲染虚拟地图；附带 15 条金标集的 Agent 评测模块，以及服务启动时必跑的 PyTorch“工具路由 + 回答质量判定”本地训练流程。

## 能力说明

- **多工具 Agent**：LangChain `create_agent` 挂载 5 个工具——知识检索、虚拟地图展示、疗养路线推荐、两分区步道计算、单分区信息查询。模型可在同一轮调用多个工具；调用 `show_virtual_map` 时，聊天区会收到 SSE `map` 事件并渲染虚拟二维图。回答优先注明检索资料来源。
- **流式对话与逐轮质检**：`POST /api/chat` 以 SSE 返回 `status`、`tool`、`map`、`token`、`eval`、`summary`、`done` 或 `error` 事件。每轮回答都会做一次规则化质检（回答长度、禁用表述、资料引用、路线免责声明、健康安全提示），并将问题截断至 512 字、工具调用、质量分和检查明细写入 MySQL 的 `eval_turn_records` 表，供管理员“测试报告”查看。
- **滚动摘要记忆**：当前页面内最近 16 条消息作为短期上下文；回答完成后模型把本轮对话压缩为仅含关键要点的滚动摘要（目标、偏好、健康安全限制、路线选择和未解决问题，最多 10 条短句、800 字以内），不保留寒暄和逐句复述。摘要存入 MySQL，刷新或重新登录后对话栏清空，只载入摘要；用户可主动清除。
- **个性化疗养顺序**：路线页询问活动偏好、可用时长和步行能力，再从已配置线路中按偏好匹配度与时长筛选最多 3 条候选；虚拟二维图支持选择分区起终点，并用 Dijkstra 计算配置步道网络中的最短示意路线。点位、步道与距离均为开发演示数据，不用于实地导航。
- **RAG 知识库**：支持 TXT、Markdown、PDF，按文件 SHA-256 哈希做增量索引，存入本地 Chroma；检索使用 MMR 减少重复片段，并为文本块写入 `title`、`source`、`topic` 元数据。
- **多模型**：`config/models.yml` 配置通义千问、DeepSeek、OpenAI、智谱、Moonshot 共 5 家供应商的模型端点；服务端只返回 `.env` 中已配置 API Key 的模型。Embedding 固定使用 DashScope `text-embedding-v4`，切换对话模型不改变向量空间。
- **身份与权限**：MySQL 保存账号、角色、启用状态、滚动摘要和逐轮质检记录，不保存完整聊天正文。密码使用 PBKDF2-HMAC-SHA256（310,000 次迭代 + 16 字节随机盐）哈希。服务端使用 Starlette `SessionMiddleware` 签发同源会话 Cookie；页面每次加载（含刷新、新开标签页）都会主动调用注销接口，必须重新登录。
- **管理员功能**：管理员不提供咨询对话入口，可创建/停用/改角色/删除账号、索引知识库、查看普通读者的滚动摘要，以及查看按读者汇总的测试报告与逐轮质检明细；管理员不能删除当前登录的自身账号。
- **Agent 金标评测**：`agent/golden_cases.json` 包含 15 条用例，评测器按工具召回、必备事实、禁用表述和资料引用打分并输出 JSON 报告。
- **本地模型训练**：每次服务启动都会用项目内置监督样本、Agent 金标集及已有的本地对话记录训练轻量“工具路由 + 回答质量判定”模型；下一轮对话把高置信度工具建议作为参考注入上下文。训练只在启动时执行一次，聊天请求只做 CPU 推理；该模块**不微调云端大模型**，也不改变医疗安全边界。
- **咨询功能分区**：聊天分为森林知识、身体健康、健康生活建议三个领域。疗养路线页提供完整筛选与地图交互；聊天中的路线/地图问题会通过工具按需回答，并保留演示数据免责声明。

> 虚拟地图与路线长度均为开发演示数据，不代表真实基地内部测绘或现场步道，不是导航服务；不包含实时开放状态、施工、天气、客流、救援设施或无障碍信息。路线规划不能代替医疗意见。

## 技术栈与环境要求

| 类别 | 内容 |
| --- | --- |
| 运行环境 | Python 3.10+，MySQL 8.x |
| Web 后端 | FastAPI、Uvicorn、Starlette Session |
| Agent / LLM | LangChain 1.x、LangGraph、LangChain OpenAI 兼容接口 |
| 向量检索 | Chroma、DashScope `text-embedding-v4`、MMR |
| 前端 | Vue 3.5（unpkg CDN，无需 Node.js 构建）、原生 CSS / SVG |
| 本地训练 | PyTorch 2.2+（启动时在 CPU 上训练轻量分类器） |
| 网络依赖 | 聊天/Embedding 需访问对应供应商 API；前端 Vue 与 Google Fonts 默认从 CDN 加载 |

## 本地环境

PowerShell：

```powershell
Set-Location D:\PythonProject\Forest-agent
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

若使用 Windows 服务版 MySQL，可先检查并启动服务（服务名依安装版本而异）：

```powershell
Get-Service *mysql*
Start-Service MySQL84
```

确认 `.env` 中 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE` 均对应本机 MySQL，并将 `DASHSCOPE_API_KEY` 设置为有效的百炼 API Key。若 `.env` 已存在，不要用模板覆盖，也不要在 README、终端输出或版本控制中暴露密钥；仅补齐缺失项。默认 MySQL 地址为 `127.0.0.1:3306`、账号 `root`、数据库 `forest_wellness`，可按本机安装调整。

### 环境变量

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_USER` / `MYSQL_PASSWORD` / `MYSQL_DATABASE` | 是 | MySQL 连接信息；密码为空时服务无法初始化数据库 |
| `DASHSCOPE_API_KEY` | 是（聊天与 RAG） | 百炼聊天模型和 `text-embedding-v4` Embedding 共用；虚拟路线规划不依赖它 |
| `EMBEDDING_MODEL_NAME` | 否 | 默认 `text-embedding-v4` |
| `APP_SESSION_SECRET` | 生产必填 | 会话签名密钥；未配置时每次启动随机生成，重启后旧会话失效 |
| `APP_COOKIE_SECURE` | 否 | 设为 `true` 时 Cookie 仅通过 HTTPS 发送 |
| `APP_HOST` / `APP_PORT` / `APP_OPEN_BROWSER` | 否 | `run.py` 的默认监听地址、端口和自动打开浏览器开关 |
| `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ZHIPUAI_API_KEY` / `MOONSHOT_API_KEY` | 否 | 配置后对应模型才会出现在模型选择器 |
| `TRAINING_CAPTURE` | 否 | 设为 `true` 时把每轮原始问答写入本地 JSONL |
| `TRAINING_LOG_PATH` / `TOOL_ROUTER_MODEL` / `TOOL_ROUTER_HINT_THRESHOLD` | 否 | 训练样本路径、模型检查点路径、工具提示阈值（默认 0.35） |

### 初始化数据库与演示账号

```powershell
python -m db.init_db
python -m db.demo_accounts
```

演示账号固定为 `admin`（管理员）、`reader` 和 `guest`（读者），三者的本地演示密码均为 `123456`。种子脚本会把旧的 `owner`、`demo_admin`、`demo_reader`、`demo_guest` 迁移到上述账号并更新演示密码；遇到同名的非演示账号会跳过，不覆盖。该短密码仅用于本机演示，部署到共享或公网环境前必须删除/改密。普通注册和管理员创建仍要求至少 10 位密码。

如果数据库里已有一个不可登录的同名 `admin` 账号（例如早先通过“初始化管理员”创建、密码已遗忘），普通种子脚本会报告冲突并跳过。此时在本机演示库中执行：

```powershell
python -m db.demo_accounts --reset
```

`--reset` 会把同名账号强制重置为演示账号（密码 `123456`、角色恢复），并停用同时存在的旧演示账号（如 `owner`），旧账号的历史数据仍保留。该操作会覆盖现有密码，请勿在共享或生产环境使用。

`python -m db.init_db` 会在首次连接时创建数据库和 4 张表：`users`、`chat_messages`、`conversation_memory`、`eval_turn_records`。

### 启动

```powershell
python run.py
```

启动后终端会输出可点击的本机网址 `http://127.0.0.1:8000`；在 VS Code、Windows Terminal 等支持链接的终端中 Ctrl+Click（或单击）即可打开浏览器。API 文档位于 `http://127.0.0.1:8000/docs`。

常用参数：

```powershell
python run.py --open          # 服务就绪后自动打开浏览器
python run.py --no-browser    # 不自动打开浏览器
python run.py --host 0.0.0.0  # 允许局域网访问
python run.py --reload        # 开发热重载
```

监听地址和端口也可通过 `.env` 的 `APP_HOST`、`APP_PORT` 配置。生产环境应配置随机的 `APP_SESSION_SECRET`，并启用 HTTPS 后设置 `APP_COOKIE_SECURE=true`。

首次启动如果尚无管理员，登录页会显示“初始化管理员”入口；也可以通过 `python -m db.demo_accounts` 创建演示管理员。

## 会话与权限

- 登录成功后，用户信息保存在服务端会话 Cookie 中；前端不再持久化登录态。
- 页面每次加载（包括刷新、重新打开或新开标签页）都会调用 `POST /api/auth/logout` 清除会话并回到登录页，因此**每次重新加载页面都需要重新登录**。同一浏览器中新开页面会结束已有会话，其他已打开标签页的后续请求会返回 401。
- 角色只有 `reader`（读者）和 `admin`（管理员）。管理员调用 `POST /api/chat` 会返回 403；管理员入口默认进入“账号管理”。
- `GET /api/auth/me` 只返回用户信息、空的消息列表和摘要，不返回历史聊天正文；调用时会清理 `chat_messages` 中的短期记录。

## 知识库

`data/knowledge/` 当前包含 8 篇结构化资料：

| 文件 | 覆盖主题 |
| --- | --- |
| `森林环境与健康科普.txt` | 森林浴、负氧离子、植物精气、证据强度与敏感人群 |
| `身体活动与慢病安全.txt` | 活动量、强度判断、高血压/糖尿病/心肺疾病安全 |
| `康复支持与就医边界.txt` | 康复原则、术后活动、跌倒预防、急症边界 |
| `基地分区与路线说明.txt` | 5 个规划分区、路线筛选规则、无障碍与运营信息边界 |
| `森林生态与动植物观察.txt` | 森林层次、常见植物、蛇虫蜱蜂、无痕山林 |
| `户外环境风险与急救常识.txt` | 中暑、失温、扭伤、蛇咬、雷电、山洪、迷路 |
| `健康信息辨识与常见误区.txt` | 负氧离子/排毒/保健品等误区、证据等级、AI 边界 |
| `特殊人群森林活动建议.txt` | 老年人、儿童、孕期、慢病、行动不便、过敏人群 |

索引配置位于 `config/rag.yml`：文本块 700 字、重叠 120 字，检索返回 6 个片段（MMR，`fetch_k` 至少为 `k * 4`）。每个文本块会补充 `主题：…` 前缀并写入 `title`、`source`、`topic` 元数据，引用时返回资料主题和来源。

索引方式：

- 管理员登录后进入“知识资料”，点击“索引 / 更新资料”；
- 或在终端运行：

```powershell
python -c "from rag.vector_store import VectorStoreService; print(VectorStoreService().index_documents())"
```

新/改文档会按文件哈希增量写入 Chroma；修改知识内容后再次执行索引即可。**改变 embedding 模型后必须全量重建**：停止服务，整体删除 `chroma_db/` 后重新执行索引。不要在同一集合混用不同向量空间。迁移/备份时需整体保留 `chroma_db/`。

知识资料是基于 WHO、CDC、NHS 与同行评审资料整理的摘要，末尾保留来源链接。健康内容仅作一般教育；不得用于诊断、治疗、停药或制定个体康复处方。

## 地图与疗养路线数据

路线页使用 `data/base/base_zones.json` 中的 `map_position` 与 `trails` 渲染本地 SVG 虚拟二维地图，并基于 6 条已配置步道、5 个规划分区和 3 条示范线路计算疗养顺序与最短示意路线。相关说明见 `data/base/README_MAP.md`。

- 路线筛选条件：预计时长不超过用户输入、步行能力对应的最大强度、轮椅/行动不便时只接受 `wheelchair_accessible=true` 的线路；当前 3 条线路的该字段均为 `null`，因此不会推荐轮椅路线。
- 候选线路按“偏好匹配数降序、时长升序”排序，最多返回 3 条。
- 两分区步行路线使用 `services/virtual_map.py` 中的 Dijkstra 最短路，距离和时间均为配置演示值。

该地图不调用外部底图、地理编码或真实导航服务，也不表达实地地理关系。投入真实运营前必须由基地运营方替换并核实分区、内部步道、距离、开放状态及无障碍信息；当前虚拟坐标和连接不能作为实际疗养导航。康养路线也不替代医疗评估。

## Vue 前端

唯一前端入口使用 FastAPI 同源提供 Vue 3 页面和 `/api` 接口，不需要 Node.js 构建步骤。页面静态文件位于 `frontend/`：

- `index.html`：单页入口，从 unpkg 加载 Vue 3.5.13，从 Google Fonts 加载字体；
- `app.js`：登录、疗养路线、森林咨询、知识资料、管理员视图与 SSE 解析；
- `style.css`、`virtual-map.css`、`chat-categories.css`：页面、虚拟地图与咨询分类样式。

前端通过服务端会话 Cookie 调用 Python 业务层，并用 `localStorage` 记住上次选择的对话模型。由于 Vue 和字体默认来自 CDN，完全离线部署时需要把这两个资源改成本地文件。

## HTTP API

除 `/api/status`、登录、注册、首任管理员创建和静态页面外，其余接口均要求登录；管理员接口还要求 `admin` 角色。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | 数据库就绪状态、是否需要初始化管理员（公开） |
| POST | `/api/auth/login` | 登录并写入会话 |
| POST | `/api/auth/register` | 注册读者账号 |
| POST | `/api/auth/first-admin` | 无管理员时创建首任管理员 |
| POST | `/api/auth/logout` | 清除会话 |
| GET | `/api/auth/me` | 当前用户、空消息列表与滚动摘要 |
| GET | `/api/models` | 已配置密钥的模型列表 |
| POST | `/api/chat` | SSE 流式咨询（仅读者） |
| DELETE | `/api/chat` | 清除短期消息与滚动摘要 |
| GET | `/api/chat/evaluations` | 当前用户最近 50 条质检记录 |
| GET | `/api/routes/map` | 虚拟地图分区与步道数据 |
| POST | `/api/routes/recommend` | 按偏好、时长、步行能力筛选线路 |
| POST | `/api/routes/walking` | 两分区最短示意路线 |
| GET | `/api/knowledge` | 知识文件列表 |
| POST | `/api/knowledge/index` | 增量索引知识库（管理员） |
| GET/POST | `/api/admin/users` | 账号列表 / 创建账号（管理员） |
| PATCH/DELETE | `/api/admin/users/{user_id}` | 更新状态、角色、密码 / 删除账号（管理员） |
| GET | `/api/admin/summaries` | 普通读者滚动摘要（管理员） |
| GET | `/api/admin/reports` | 测试报告汇总与最近 200 条质检记录（管理员） |

## Agent 工具与评测

工具定义位于 `agent/tools.py`，系统提示位于 `prompts/system_prompt.txt`：

- `search_forest_knowledge`：本地知识库检索，返回带主题和来源的资料片段。
- `show_virtual_map`：返回分区概要，并向聊天 SSE 流发送 `map` 事件供前端渲染。
- `recommend_wellness_routes`：按偏好、时长和步行能力筛选已配置线路。
- `plan_walking_route`：按分区名称或 id 计算最短示意步道。
- `get_zone_information`：查询单个分区的功能、标签和核验状态。

金标集位于 `agent/golden_cases.json`（15 条用例，覆盖地图、路线、知识、健康安全、特殊人群和拒答边界）。运行真实模型评测（会产生模型 API 调用费用）：

```powershell
python -m agent.evaluation --model qwen-plus
python -m agent.evaluation --limit 5
python -m agent.evaluation --case map-and-route --case wheelchair-route
```

评测器只检查可观察行为，不调用 LLM 评审：工具召回占 40%、必备事实覆盖率占 35%、无禁用表述占 15%、资料引用占 10%，总分不低于 0.70 且满足各用例阈值才算通过。终端输出摘要，详细结果写入 `reports/agent_eval_<时间>.json`（已被 `.gitignore` 排除）。

离线测试不会调用真实模型；如需在 pytest 中做一次真实模型冒烟测试：

```powershell
$env:RUN_LIVE_AGENT_TESTS="1"
python -m pytest tests/test_live_agent.py -q
```

## 本地 PyTorch 训练

说明：本项目的对话模型来自云端 API，**不能在本地用 PyTorch 微调这些模型**。这里的 PyTorch 模块是一个本地辅助分类器：

1. **工具路由头**：根据用户问题预测应优先考虑哪些工具（多标签分类）；
2. **质量判定头**：根据问题和回答预测该轮是否满足引用、安全和免责声明规则。

每次运行 `python run.py` 时，服务启动过程都会强制完成一次本地训练，再开始接受请求。训练数据由 `data/training/router_samples.jsonl` 中维护的多样化工具问句与质量正反例、`agent/golden_cases.json` 金标用例，以及存在时的 `data/training/turns.jsonl` 组成。内置样本不需要网络或用户数据；项目不会因检查点已存在而跳过训练。缺少 PyTorch 或训练失败时，服务启动会报错，不会静默降级。

训练使用固定随机种子、小型哈希 n-gram `EmbeddingBag`、Adam 权重衰减、梯度裁剪和验证集早停，训练/验证分离的流程参考 [PyTorch 官方优化教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。工具建议采用轻量分类提示而非强制调度，符合 [LangChain Router 模式](https://docs.langchain.com/oss/python/langchain/multi-agent/router)中分类器负责路由、Agent 继续编排的分工；本地模型不会覆盖 Agent 对实际工具调用的判断。CPU 训练仅发生在服务启动，运行中的聊天只进行轻量 CPU 推理。训练数据或训练逻辑变更后也可手动运行：

```powershell
python -m training.train
```

`training.train` 支持 `--data`、`--output`、`--epochs`、`--batch-size`、`--buckets`；内置样本和金标集始终参与训练，不提供跳过选项。训练产物默认为 `data/training/tool_router.pt`，通过临时文件替换方式保存。模型会把置信度超过 `TOOL_ROUTER_HINT_THRESHOLD`（默认 0.35）的工具建议作为参考注入上下文，并由质量头提供每轮辅助质量分；规则化质检仍是权威结果。

隐私提示：`TRAINING_CAPTURE` 默认关闭。开启后本地 JSONL 会保存用户原始问题和回答，并在后续启动训练时被读取；请先取得用户同意、避免记录敏感身份信息，并在共享/公网部署前评估合规要求。内置训练集 `router_samples.jsonl` 随项目维护；用户对话日志和生成的模型检查点已被 `.gitignore` 排除。

## 数据存储与隐私

- **MySQL**：`users` 保存账号、角色和状态；`conversation_memory` 保存每用户滚动摘要；`eval_turn_records` 保存每轮质检记录（问题前 512 字、工具、质量分、检查明细）；`chat_messages` 为短期消息表，当前 Web 对话流程不写入完整正文，用户登录时会清理。
- **Chroma**：`chroma_db/` 保存知识库向量索引和 `source_manifest.json` 文件哈希清单；迁移时需整体保留。
- **本地文件**：`data/training/router_samples.jsonl` 是随项目维护的训练集；`turns.jsonl` 仅在启用记录后生成；`tool_router.pt` 在每次服务启动训练后生成，后二者默认不进入版本控制。
- **日志/报告**：`logs/` 与 `reports/` 已加入 `.gitignore`。

## 测试

```powershell
python -m pytest -q
```

当前 `tests/` 包含 17 个测试文件，主要覆盖：密码哈希与认证失败路径、演示账号迁移、管理员对话限制、读者摘要权限与查询范围、会话清理、工具注册与参数解析、聊天地图/工具事件、金标集评分、知识库结构、虚拟地图数据、路线偏好筛选与步道最短路、模型配置、摘要更新、内置训练样本覆盖、模型前向、训练与检查点加载。PyTorch 是必需依赖，缺失时训练测试不会跳过。真实 MySQL 登录/CRUD 和真实模型评测属于集成验证，需要可用的 `.env` 与 API Key。

## 目录

```text
Forest-agent/
├── agent/
│   ├── agent.py            # create_agent 智能体组装
│   ├── tools.py            # 5 个智能体工具与地图载荷
│   ├── route_planner.py    # 线路筛选与推荐文本
│   ├── evaluation.py       # 金标集评测器与命令行入口
│   └── golden_cases.json   # 15 条 Agent 评测用例
├── config/
│   ├── models.yml          # 多供应商模型端点
│   ├── rag.yml             # Chroma / 分块 / 检索配置
│   └── settings.py         # 环境变量与路径
├── data/
│   ├── base/               # base_zones.json 与地图说明
│   ├── knowledge/          # 8 篇结构化康养与安全资料
│   └── training/           # 内置路由样本，以及本机对话记录和模型产物
├── db/
│   ├── connection.py       # PyMySQL 连接与查询封装
│   ├── init_db.py          # 建库建表
│   ├── repository.py       # 用户、摘要、质检数据访问
│   └── demo_accounts.py    # 演示账号种子脚本
├── frontend/
│   ├── index.html          # Vue 单页入口
│   ├── app.js              # 登录、问答、路线、管理逻辑
│   ├── style.css
│   ├── virtual-map.css
│   └── chat-categories.css
├── memory/conversation.py  # token 估算、摘要与历史压缩
├── model/factory.py        # OpenAI 兼容模型工厂
├── prompts/system_prompt.txt
├── rag/vector_store.py     # Chroma 索引、MMR 检索
├── services/virtual_map.py # 虚拟步道网络最短路径
├── tests/                  # 17 个离线测试文件
├── training/               # 必需的 PyTorch 工具路由/质量模型
│   ├── schema.py           # 每轮对话记录与规则质检
│   ├── capture.py          # JSONL 训练样本落盘
│   ├── features.py         # 哈希字符/词 n-gram 特征
│   ├── model.py            # EmbeddingBag 多任务模型
│   ├── train.py            # 训练命令行入口
│   └── inference.py        # 懒加载推理与工具建议
├── utils/security.py       # PBKDF2 密码哈希
├── api.py                  # FastAPI 接口与静态页面入口
├── run.py                  # 启动脚本：终端输出可点击的本机网址
├── requirements.txt        # 全部依赖
└── .env.example            # 环境变量模板
```

## 设计参考

本项目在 Agent 工具注册、金标集评测、检索质量与训练闭环上的做法参考了这些公开实践：

- AgentUnit：pytest 风格的多框架 Agent 场景/指标评测框架：https://pypi.org/project/agentunit/
- RAGAS：RAG 检索与生成质量评估指标：https://docs.ragas.io/
- Anthropic Contextual Retrieval：为文本块补充上下文以降低检索失败率：https://www.anthropic.com/news/contextual-retrieval
- AWS SageMaker + MLflow LangGraph Agent 评测示例：https://github.com/aws-samples/genai-ml-platform-examples
- 阿里云开发者社区，《AI 智能体的开发与测试》：https://developer.aliyun.com/article/1735225

## 参考来源

- WHO 身体活动：https://www.who.int/zh/news-room/fact-sheets/detail/physical-activity
- CDC 成人活动：https://www.cdc.gov/physical-activity-basics/guidelines/adults.html
- CDC 老年人活动：https://www.cdc.gov/physical-activity-basics/guidelines/older-adults.html
- CDC 慢性病/残障人士活动：https://www.cdc.gov/physical-activity-basics/guidelines/chronic-health-conditions-and-disabilities.html
- NHS 物理治疗：https://www.nhs.uk/conditions/physiotherapy/
- WHO 空气质量与健康：https://www.who.int/health-topics/air-pollution
- WHO 生物多样性与健康：https://www.who.int/news-room/fact-sheets/detail/biodiversity-and-health
- NHS First aid：https://www.nhs.uk/conditions/first-aid/
- 国家林业和草原局：https://www.forestry.gov.cn/
- 森林浴综述 DOI：https://doi.org/10.3390/ijerph14080851
