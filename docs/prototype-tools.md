# 可运行的基础工具

这些工具是研究准备与公式核查原型，不包含 YOLO、神经网络特征采样、自动求导、训练循环或官方数据集评测实现。所有测试和演示使用合成框及手写数字，不代表模型性能。

本轮代码实际编写、验证于 **2026-10-21**。若 Git 作者日期被安排到历史日期，该元数据用于历史贡献整理，不表示代码、实验或结论当时已经存在；实际实施时间以本说明及提交者日期为准。

## 运行

在仓库根目录使用 Python 3.10 或更新版本，无需第三方运行时依赖：

```console
python -m crir_kd demo
python -m unittest discover -s tests -v
python -m crir_kd yolo path/to/labels.txt --classes 10
python -m crir_kd coco path/to/annotations.json
python -m crir_kd splits path/to/splits.json
python -m crir_kd run-record path/to/run.json
python -m crir_kd summarize path/to/runs.json
```

命令只读取指定文件并向标准输出写 JSON，不修改数据、不下载资源。退出码 `0` 表示检查通过，`1` 表示审计发现问题，`2` 表示输入、参数或读文件错误。空 YOLO 标签文件可以表示负样本。检查器不会自动裁剪或删除可疑标注，也不会代替 VisDrone 忽略区域规则或 AI-TOD-v2 官方评测。

## 几何与可靠性

- `Box` 使用连续像素坐标、`xyxy` 角点及严格正宽高；`from_xywh` 显式转换格式，IoU 不加 `1`。
- `ViewTransform` 记录两个方向的缩放、补边及水平翻转，可正向与逆向映射框。`letterbox` 是连续尺寸的理想中心补边；接入实际图片处理时，必须记录图像库取整后的实际缩放与边距。它不处理图片，也不提供裁剪增强。
- `match_quality` 在同类别候选中按 IoU、分数、输入索引排序，贪心一对一匹配。`q = IoU × score`，未匹配 GT 的 q 为零。它不是最优二分图匹配或官方 AP 实现。
- `instance_stability` 接收已逆变换到共同原图坐标的框；不可见视图不计分母，可见但漏检的视图计为缺失。变化量是相对参考预测的中心和宽高绝对变化，分别按 GT 宽高（带下限）归一化后取均值。`s = coverage × exp(-variation / tau)`；没有参考预测或有效匹配时为零。
- `reliability_weights` 提供 uniform、quality、stability、joint 四种对照。joint 采用 `floor + (1-floor) × q × s`；`matched_mean_control` 用等平均权重的统一权重控制整体 KD 强度。
- `weighted_kd_loss` 使用 `mean(weight × instance_loss)`，分母为全部实例数量；不会除以权重和。`total_loss` 保留独立传入的完整检测损失。这里只核查标量公式，不验证梯度。
- 对面积相加或损失求和溢出的极端有限输入，IoU 和损失均值使用缩放计算；最终总损失若仍超出浮点表示范围，会明确抛出 `ValueError`，不返回无穷大。
- `size_diagnostics` 按显式提供的 `sqrt(box area)` 边界分组，区间为左闭右开。每个目标（含未匹配者）都计入分母；必须命名坐标空间。输出不是官方 tiny/small AP 或召回。

这些定义是可检查的起点，不是已经选定的最终研究方法。冻结教师、神经网络梯度、RoIAlign/投影和导出学生模型仍待实现。

## 划分清单

路径相对于同一个数据根目录；可选 SHA-256 必须来自实际文件内容，`source_group` 可记录原视频或拍摄序列：

```json
{
  "train": [{"path": "train/0001.jpg", "source_group": "sequence-1"}],
  "val": [{"path": "val/0002.jpg", "source_group": "sequence-2"}]
}
```

工具检查重复路径、重复内容摘要及跨集合的同源组，给出清单指纹。它不读取图像、不自动计算内容摘要、不识别近重复画面；路径大小写保留，需按数据集的实际命名约定统一。

## 运行记录与多种子报告

`run-record` 所需文本字段包括 `run_id`、`dataset_name`、`dataset_version`、`split`、`config_path`、`student`、`teacher`、`python_version`、`framework_version`、`device`、`evaluation_command`、`raw_log_path`。还需 40 位 `code_commit`、64 位 `split_fingerprint`/`config_sha256`、非负整数 `seed`、`student_input`/`teacher_input` 的 `[宽, 高]` 数组、布尔 `synthetic`，以及 `data_use`（`train`、`validation` 或 `test-evaluation-only`）。通过结构校验不表示工具已经核实文件存在或实验真实发生。

`summarize` 的输入为数组，每项需要 `method`、`protocol_fingerprint`、`seed`、`synthetic` 和数值 `metrics`。协议指纹应由调用方对控制变量的规范化配置计算 SHA-256；只改变方法/种子时应保持相同协议，改变数据版本、师生结构、输入尺寸、预算或评测口径时必须更新。不同协议、不同方法以及合成/实测结果分组统计，不混合。单次运行的样本标准差为 `null`；同组种子不允许重复，各种子必须报告相同指标。输出只作描述性统计，不产生显著性或优越性结论。
