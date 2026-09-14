# 相关论文与官方资源

以下资源用于定位已有研究和后续复现。列入本页不表示本项目已经复现了它们，也不表示作者代码可以直接用于任意 YOLO 版本。

## 与选题直接相关

1. **Multi-Scale Aligned Distillation for Low-Resolution Detection (MSAD), CVPR 2021**\
   高低分辨率目标检测与多尺度特征对齐已有先例。\
   [原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Qi_Multi-Scale_Aligned_Distillation_for_Low-Resolution_Detection_CVPR_2021_paper.html) · [作者代码](https://github.com/dvlab-research/MSAD)

2. **CRKD-YOLO: Cross-Resolution Knowledge Distillation for Low-Resolution Remote Sensing Image Object Detection, IEEE TIM 2025**\
   与跨分辨率 YOLO 最直接相邻。不能把高分辨率教师与低分辨率学生的组合本身当成新贡献。\
   [作者仓库与论文 PDF](https://github.com/Jianfantasy/CRKD-YOLO) · [DOI](https://doi.org/10.1109/TIM.2025.3559616)

3. **End-to-End Semi-Supervised Object Detection With Soft Teacher, ICCV 2021**\
   半监督设定下利用框扰动评估定位指导可靠性；需要区分本项目的全监督、跨分辨率问题与已有可靠性筛选。\
   [原论文](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html) · [作者代码](https://github.com/microsoft/SoftTeacher)

4. **AvatarKD, ACM Multimedia 2023**\
   教师特征扰动与不确定性加权已有相关研究，需检查机制重合。\
   [论文](https://arxiv.org/abs/2305.02722) · [作者代码](https://github.com/Gumpest/AvatarKD)

5. **Localization Distillation for Dense Object Detection, CVPR 2022**\
   定位分布蒸馏与有效定位区域的相关工作。\
   [原论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html) · [作者代码](https://github.com/HikariTJU/LD)

6. **CrossKD: Cross-Head Knowledge Distillation for Object Detection, CVPR 2024**\
   研究预测蒸馏中教师指导与真实标注之间的监督冲突。\
   [作者代码与方法说明](https://github.com/jbwang1997/CrossKD)

7. **Knowledge Distillation via Hierarchical Matching for Small Object Detection, JCST 2024**\
   小目标分层特征匹配蒸馏已有先例。\
   [期刊论文页](https://www.sciopen.com/article/10.1007/s11390-024-4158-5)

## 数据集

- [VisDrone 官方数据集仓库](https://github.com/VisDrone/VisDrone-Dataset)：本项目初始计划使用 DET 图像检测任务。
- [AI-TOD-v2 官方项目页](https://chasel-tsui.github.io/AI-TOD-v2/)：微小目标检测基准与 NWD 相关研究。
- [AI-TOD-v2 与 NWD-RKA 官方实现](https://github.com/Chasel-Tsui/mmdet-aitod)：核实数据版本、评价规则和方法实现。

## 阅读时记录的问题

- 监督设定是全监督还是半监督？训练时是否可以访问真实框？
- 师生输入分辨率是否不同？学生推理结构是否改变？
- 蒸馏的是特征、类别预测还是定位分布？
- 可靠性如何定义？额外教师计算量如何处理？
- 对 tiny 目标和教师漏检的处理是否引入偏差？
- 哪些实验可以真正区分本项目的候选假设与已有方案？
