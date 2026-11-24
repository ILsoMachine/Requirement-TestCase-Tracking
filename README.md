# 需求—测试用例追踪链接恢复

本项目比较传统关键词检索、向量检索和大模型重排，在三个项目中恢复自然语言需求与已有测试用例之间的追踪链接。

## 数据

| 项目 | 可评分需求 | 候选用例 | 参考链接 | 数据出处 |
| --- | ---: | ---: | ---: | --- |
| EBT | 39 | 25 | 51 | [TRIAD 复现数据](https://github.com/huiAlex/TRIAD/tree/524929e892dd39b8b370ffeef17b4c6962210e07/dataset/ebt) |
| Mozilla 前端 | 184 | 184 | 184 | [Burak Yolacan，2024，Mozilla browser front-end software requirements and test scripts dataset，V1](https://doi.org/10.17632/jw5bgshtvg.1)，CC BY 4.0 |
| Eclipse EDC | 156 | 177 | 176 | [Jorge Gavilan，2025，Requirements Traceability Matrix - EDC，v1](https://doi.org/10.5281/zenodo.17546974)，CC BY 4.0 |

Mozilla 的源 CSV 以分号分隔。根据发布者的同编号约定生成一对一参考链接，编号仅用于存储和评分。EDC 需求使用标题、正文和验收条件，用例使用描述字段。评分链接取工作簿需求页和用例页一致的映射，排除一条映射冲突的需求，保留全部用例参与检索。导入统一空白并去除显式制品 ID。原始资料和导出的预处理版本均按项目存放，导入报告记录版本、SHA-256 和筛选口径。 [CC BY 4.0 许可正文](https://creativecommons.org/licenses/by/4.0/)

EBT 的导入脚本验证上游文件的 SHA-256、主键和参考链接，去除测试用例标题中的章节交叉引用。需求 100 和 101 保留供检索，其余 39 条有参考链接的需求参与评分。原件、清洗文本、参考矩阵和导入报告均以项目名归档。

## 实验设置

[config/experiment.json](config/experiment.json)统一定义数据筛选、模型版本、随机种子、候选窗口和重试规则。[prompts/](prompts/)保存评分准则、任务提示和引文校验要求。

每个项目运行 BM25、MiniLM、BM25 前 10 + DeepSeek、MiniLM 前 10 + DeepSeek 和 MiniLM 前 30 + DeepSeek。EBT 的最后一组按候选库实际大小重排全部 25 条用例。MiniLM 通过 ONNX Runtime 在 CPU 上执行，对非填充位置取均值并归一化，跨片段取最大相似度。DeepSeek 使用 `deepseek-flash` 并关闭思考模式。重排按校验后的等级排序，同等级和窗口外候选沿用原有顺序。评分按项目报告宏平均 Recall@5、MAP 和逐需求配对变化。 [DeepSeek 模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

## 结果结构

唯一的正式结果目录是 [results/experiment-20261124/](results/experiment-20261124/)。

| 路径 | 内容 |
| --- | --- |
| `config.json`、`prompts.json`、`manifest.json`、`execution.json` | 冻结配置、提示词、各批次来源和文件校验和 |
| `summary.csv`、`summary.json`、`contrasts.csv` | 15 个项目与方法组合的指标、合计和分组 API 用量、证据校验统计与配对区间 |
| `{ebt,mozilla,edc}/` | 各项目的基线排名、逐需求评分、配对变化和三组原始 API 请求／响应 |

[实验汇总及案例](results/experiment-20261124/analysis.md)

## 离线复评分

在仓库根目录运行以下命令。这个步骤使用已归档排名和参考链接，只依赖 Python 标准库。

```powershell
python -m experiment.run score
```

## 从原件复跑全部三项目

使用 Python 3.12。在激活的虚拟环境中执行以下命令。新目录会针对三个项目重新计算基线，并按冻结设置提交全部重排请求。模型下载可使用当前终端的 `HTTPS_PROXY`。DeepSeek 密钥通过 `DEEPSEEK_API_KEY` 或本地忽略文件 `model.local.json` 的 `api_key` 读取。

```powershell
python -m pip install -r requirements.txt
python scripts/import_ebt.py
python -m experiment.import_expansion mozilla
python -m experiment.import_expansion edc
python -m experiment.download
python -m experiment.run prepare --run-dir results/reproduction
python -m experiment.run rerank --run-dir results/reproduction
python -m experiment.run score --run-dir results/reproduction
```

重排按需求保存请求、原始响应、校验结果和最终排名。中断后再次运行相同的 `rerank` 命令，可以继续处理缺少记录的需求。云端模型响应可能随服务版本变化，历史指标可通过离线复评分精确复现。
