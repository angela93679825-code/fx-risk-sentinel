# 上传说明

建议仓库名称：`fx-risk-sentinel`。

建议简介：`FX volatility forecasting, model explanations and historical portfolio reconstruction in Python.`

建议 Topics：`python`、`xgboost`、`time-series`、`shap`、`portfolio-optimization`、`reproducible-research`。

解压后，上传本目录里面的 README.md、run.py、requirements.txt、RIGHTS.md、.gitignore 以及 fxrisk、scripts、data/demo、docs、results、tests 等目录到新仓库根目录。不要把ZIP文件本身当成项目源代码上传，也不要额外套一层同名目录。

本包已经排除真实输入、个人电脑路径、诊断日志和临时缓存；保留了可运行的独立模拟数据及真实数据运行的汇总结果。真实输入导入到data/private并被.gitignore忽略。不要为了展示研究成绩把模拟数据的结果写成真实回测。

README已说明历史代码的限制，不需要把尚未验证的风险过滤写成有效策略，也不要标注不存在的收益提升。原项目为合作项目，具体分工应只补充你能够确认的内容。

如使用命令行，在本目录按实际当前工作进行一次正常初始提交即可；无需补造历史提交日期或贡献记录。
