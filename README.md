# 需求—测试用例追踪链接恢复

本项目用 TRIAD 的 EBT 子集评估需求到现有测试用例的追踪链接恢复。检索方法读取需求文本和完整用例库，离线评分使用 TRIAD 提供的参考链接。

## 导入数据

使用 Python 3.10 或更新版本。导入脚本只使用标准库。在仓库根目录运行：

```powershell
python scripts/import_ebt.py
```

原始文件位于 `data/raw/ebt/`。来源提交为 `524929e892dd39b8b370ffeef17b4c6962210e07`。脚本先校验 SHA-256，再检查表头、主键、文本、链接重复和两端 ID。仓库保留上游的许可文件和原始文件字节。 [TRIAD 源提交及 EBT 数据](https://github.com/huiAlex/TRIAD/tree/524929e892dd39b8b370ffeef17b4c6962210e07/dataset/ebt)

| 生成文件（`data/processed/ebt/`） | 用途 |
| --- | --- |
| `requirements.csv` | 41 条需求，列为 `id`、`text` |
| `test_cases.csv` | 完整的 25 条候选用例，列为 `id`、`text` |
| `reference_links.csv` | 51 条参考链接，列为 `requirement_id`、`test_case_id`，供离线评分 |
| `import_report.json` | 源路径、SHA-256、导入统计与无参考链接的需求 ID |

导入把文本中的连续空白统一为空格，并移除测试用例标题末尾、前置条件之前的需求章节引用，例如 `(2.1.1)(2.2.1)`。用例正文里的步骤、前置条件和结果保留在原有单列中。检索和重排以 `text` 为输入，`id` 用于建立结果索引。

参考矩阵覆盖 39 条需求和全部 25 条用例。需求 100 和 101 可用于原型检索，正式评分使用其余 39 条有参考链接的需求。需求 120 有 6 条参考链接。

## 主实验设置

BM25 提供关键词基线。语义检索使用 `sentence-transformers/all-MiniLM-L6-v2`，正式运行时记录所用权重的版本。重排组复用同一份语义检索排名，每条需求用一次请求重排前 10 个候选。主指标为 39 条需求的宏平均 Recall@5，并报告 MAP 和逐需求的排序变化。

重排模型定为 DeepSeek V4.1 Flash，API 模型 ID 为 `deepseek-flash`。2026-11-24 查到的服务端版本为 `DeepSeek-V4.1-Flash-1017`。主实验显式关闭思考模式，记录正式运行日期、模型版本与每次请求的 token 用量，并保存响应以便离线复评分。 [DeepSeek 模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) · [思考模式设置](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)
