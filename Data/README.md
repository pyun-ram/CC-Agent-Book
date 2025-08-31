# Data文件夹说明

此文件夹用于存储AI知识管理系统的运行时数据和缓存文件。

## 文件结构

```
Data/
├── workflow_logs.json      # 工作流执行日志
├── template_changes.log    # 模板变更记录
├── temp/                   # 临时文件
├── cache/                  # 缓存文件
├── knowledge_base/         # 知识库数据
└── backups/               # 数据备份
```

## 文件说明

### workflow_logs.json
记录所有工作流的执行历史，包括：
- 工作流ID
- 执行时间
- 参数信息
- 执行结果
- 错误信息

### template_changes.log
记录模板文件的变更历史，用于版本控制和回滚。

### temp/
临时文件存储目录，系统会定期清理。

### cache/
缓存文件存储目录，提高系统性能。

### knowledge_base/
存储知识库数据，包括：
- 术语库
- 写作规范
- 引用格式
- 主题分类

### backups/
重要数据的备份目录。

## 注意事项

1. 此文件夹中的文件可以安全删除，系统会自动重新生成
2. 定期清理临时文件和缓存文件
3. 重要数据建议定期备份
4. 不要手动修改workflow_logs.json文件