# AI Knowledge Management System

## 系统概述
搭建一个基于多智能体协作（MPC）的AI知识管理系统，帮助进行知识的整理、归纳和文档化处理。

## 核心工作流程

### 1. BookChapter Agent
- **功能**: 根据用户指定的主题创建标准化文章
- **输入**: 文章主题、文件类型（Markdown/LaTeX）、文件名
- **输出**: 在 `Book/` 文件夹中创建新的文章文件夹和文件
- **流程**: 
  1. 从 `Templates/` 文件夹选择合适的模板
  2. 在 `Book/` 下创建以内容命名的新文件夹
  3. 初始化并配置文章文件

### 2. Polish Agent
- **功能**: 对生成的文章进行格式化和润色
- **选项**:
  1. 格式规范化：确保符号符合 LaTeX/Markdown 规范
  2. 上下文一致性：统一术语、符号和格式
  3. 语法检查：修正语法错误和表达不清的地方
  4. 学术润色：提升学术表达的准确性和专业性

## 系统架构

### 文件夹结构
```
CC-Agent-Book/
├── Book/                    # 生成的文章存放地
│   ├── 20250810-SVD-Rt/    # 按日期-主题命名的文章文件夹
│   └── 20250811-Diffusion/  # 包含 .tex/.md 和相关资源
├── Templates/              # 文章模板
│   ├── latex-template.tex  # LaTeX 模板
│   └── markdown-template.md # Markdown 模板
├── Agents/                 # AI 智能体实现
│   ├── book_chapter_agent.py
│   ├── polish_agent.py
│   ├── template_manager.py
│   └── workflow_orchestrator.py
├── Config/                 # 系统配置
│   ├── agent_config.yaml
│   └── template_config.yaml
├── Scripts/               # 自动化脚本
├── Tests/                 # 测试框架
├── Data/                  # 知识库和中间数据
└── README.md
```

### 子智能体设计

#### 1. BookChapterAgent
- **职责**: 文章创建和初始化
- **能力**: 
  - 理解用户需求并选择合适模板
  - 创建文件夹结构
  - 生成初始内容框架
  - 处理不同的文件格式

#### 2. PolishAgent  
- **职责**: 内容质量优化
- **能力**:
  - 格式检查和修正
  - 术语一致性检查
  - 语法和拼写检查
  - 学术表达优化

#### 3. TemplateManager
- **职责**: 模板管理
- **能力**:
  - 模板版本控制
  - 模板定制化
  - 新模板创建
  - 模板验证

#### 4. WorkflowOrchestrator
- **职责**: 工作流协调
- **能力**:
  - 任务分发和调度
  - 智能体间通信
  - 质量控制
  - 错误处理

## MPC (Multi-Agent Collaboration) 机制

### 协作模式
- **管道模式**: BookChapter → Polish 的顺序处理
- **反馈循环**: Polish Agent 可以请求 BookChapter Agent 修改内容
- **质量门控**: 每个阶段都有质量检查，不合格则返回上一阶段

### 通信协议
- **消息队列**: 基于 Task 工具的异步通信
- **状态共享**: 通过文件系统共享中间状态
- **错误恢复**: 异常处理和重试机制

### 扩展性设计
- **插件化架构**: 可以轻松添加新的 Agent 类型
- **工作流配置**: 可以通过配置文件调整工作流
- **负载均衡**: 支持多个同类 Agent 并行工作

## 使用示例

### 创建新文章
```
用户: "请创建一篇关于SVD分解的LaTeX文章，文件名为SVD-Decomposition"
系统: 
1. BookChapterAgent 接收需求
2. 从 Templates/ 选择 latex-template.tex
3. 创建 Book/20250831-SVD-Decomposition/ 文件夹
4. 生成初始 LaTeX 文件
5. PolishAgent 进行格式化和质量检查
6. 返回完成状态
```

### 文章润色
```
用户: "请对刚刚创建的文章进行学术润色"
系统:
1. WorkflowOrchestrator 分配任务给 PolishAgent
2. PolishAgent 分析现有内容
3. 应用学术写作标准进行改进
4. 更新文件并返回结果
```

## 技术栈建议

### 核心技术
- **Agent Framework**: 基于 Claude Code 的 Task 工具
- **文件操作**: 系统文件 API
- **配置管理**: YAML/JSON 配置文件
- **模板引擎**: Jinja2 或类似模板系统

### 质量保证
- **自动化测试**: 每个 Agent 的单元测试
- **集成测试**: 完整工作流的端到端测试
- **性能监控**: Agent 执行时间和成功率监控

## 未来扩展

### 新增 Agent 类型
- **ResearchAgent**: 文献检索和综述生成
- **CitationAgent**: 引用管理和格式化
- **TranslationAgent**: 多语言翻译支持
- **VisualizationAgent**: 图表和数据可视化

### 高级功能
- **知识图谱**: 构建主题间的关联关系
- **版本控制**: 文章的版本管理和回滚
- **协作编辑**: 多用户同时编辑支持
- **发布集成**: 自动发布到不同平台