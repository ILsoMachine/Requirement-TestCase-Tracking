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

BM25 提供关键词基线。语义检索使用固定提交的 `sentence-transformers/all-MiniLM-L6-v2`，通过 ONNX Runtime 在 CPU 上执行。向量对序列的非填充位置取均值并做 L2 归一化。跨多个片段时取最大相似度。重排组在一次请求中评估每条需求的前 10 个语义检索候选，校验引文并按等级重新排序。同等级沿用语义检索顺序，第 11 名及之后沿用原排名。主指标为 39 条需求的宏平均 Recall@5，并报告 MAP 和逐需求的变化。

重排模型定为 DeepSeek V4.1 Flash，API 模型 ID 为 `deepseek-flash`。2026-11-24 查到的服务端版本为 `DeepSeek-V4.1-Flash-1017`。主实验显式关闭思考模式，记录正式运行日期、模型版本与每次请求的 token 用量，并保存响应以便离线复评分。 [DeepSeek 模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/) · [思考模式设置](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)

## 运行与复现

用 Python 3.12 创建虚拟环境，从项目根目录运行。已有实验目录可直接离线重新评分，评分阶段只依赖 Python 标准库。

```powershell
python -m experiment.run score --run-dir results/ebt-20261124
```

如需重新执行全部实验，在激活的虚拟环境中运行以下命令。模型下载可使用当前终端的 `HTTPS_PROXY`。DeepSeek 密钥通过 `DEEPSEEK_API_KEY` 环境变量或 Git 忽略的 `model.local.json` 中的 `api_key` 读取。另选一个运行目录可以保留已发布的实验。

```powershell
python -m pip install -r requirements.txt
python -m experiment.download
python -m experiment.run smoke --run-dir results/ebt-reproduction
python -m experiment.run baselines --run-dir results/ebt-reproduction
python -m experiment.run rerank --run-dir results/ebt-reproduction
python -m experiment.run score --run-dir results/ebt-reproduction
```

重排阶段在每条需求返回后立即保存请求正文、原始响应、校验结果和最终排名。中断后可在同一运行目录重启重排，已保存的查询会被跳过。延迟以本机语义向量索引已建立为起点，重排组计入 API 请求与校验。评分阶段只读取已经保存的排名和参考矩阵。

| 路径 | 内容 |
| --- | --- |
| `config/ebt.json` | 模型提交和校验和、BM25 参数、候选数、随机种子、重试规则 |
| `prompts/` | 评分准则、任务提示和引文校验要求 |
| `experiment/` | 权重下载、两组检索、API 重排与离线评分脚本 |
| `results/ebt-20261124/` | 本轮配置快照、排名、请求和响应、逐需求对比及汇总 |

本轮指标和代表性案例见 [实验结果分析](results/ebt-20261124/analysis.md)。

## 三项目扩展实验

| 项目 | 可评分需求 | 候选用例 | 参考链接 | 数据出处 |
| --- | ---: | ---: | ---: | --- |
| EBT | 39 | 25 | 51 | [TRIAD 复现数据](https://github.com/huiAlex/TRIAD/tree/524929e892dd39b8b370ffeef17b4c6962210e07/dataset/ebt) |
| Mozilla 前端 | 184 | 184 | 184 | [Burak Yolacan，2024，Mozilla browser front-end software requirements and test scripts dataset，V1](https://doi.org/10.17632/jw5bgshtvg.1)，CC BY 4.0 |
| Eclipse EDC | 156 | 177 | 176 | [Jorge Gavilan，2025，Requirements Traceability Matrix - EDC，v1](https://doi.org/10.5281/zenodo.17546974)，CC BY 4.0 |

Mozilla 的源 CSV 以分号分隔。根据发布者的同编号约定生成一对一参考链接，编号仅用于存储和评分。EDC 需求使用标题、正文和验收条件，用例使用描述字段。评分链接取工作簿需求页和用例页一致的映射，排除一条映射冲突的需求，保留全部用例参与检索。导入统一空白并去除显式制品 ID。原始资料和导出的预处理版本均按项目存放，导入报告记录版本、SHA-256 和筛选口径。 [CC BY 4.0 许可正文](https://creativecommons.org/licenses/by/4.0/)

每个项目有五组结果。四个主实验组为 BM25、MiniLM、BM25 前 10 + DeepSeek、MiniLM 前 10 + DeepSeek。第五组重排 MiniLM 前 30 个候选，EBT 按实际候选库重排 25 个。各深度的重排以单次请求评估整个窗口，前 30 组的输出上限为 6,600 tokens。每个处理组沿用基线的窗口外排名和同等级次序。EBT 已有的两组检索和 MiniLM 前 10 重排通过原配置与请求校验后复用。

结果保存在 `results/expansion-20261124/`。其中 `summary.csv` 为 15 个项目与方法组合，`contrasts.csv` 包含各对照的配对变化和按需求重采样的 95% Bootstrap 区间。各项目目录中的 `per-query.csv`、`paired.csv` 和重排子目录保存排名、指标和完整响应。运行方法与冻结配置写入该目录下的 JSON 文件。[汇总分析](results/expansion-20261124/analysis.md)列出主要差异、案例与 API 用量。旧 EBT 结果保存在 `results/ebt-20261124/`，扩展评分从该目录读取复用记录。

离线复评分使用 Python 标准库。

```powershell
python -m experiment.expansion score
```

从固定原件重新导入并复跑，请在已按上文安装依赖和下载 MiniLM 权重的虚拟环境中使用新的结果目录。重排每完成一条便保存，重启同一目录会检查并跳过已写入的记录。

```powershell
python -m experiment.import_expansion mozilla
python -m experiment.import_expansion edc
python -m experiment.expansion prepare --run-dir results/expansion-reproduction
python -m experiment.expansion rerank --run-dir results/expansion-reproduction
python -m experiment.expansion score --run-dir results/expansion-reproduction
```
