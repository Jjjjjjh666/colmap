# 更新内容

增加了个人文件夹，可以把自己的进度或者学习内容整理写在里面





COLMAP README文件
======

关于
-----

COLMAP 是一个通用的结构从运动（Structure-from-Motion，简称SfM）和多视图立体视觉（Multi-View Stereo，简称MVS）处理流程，它提供了图形和命令行界面。它为有序和无序图像集合的重建提供了广泛的功能。该软件在新的BSD许可证下授权。如果您将此项目用于研究，请引用以下文献：

    @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }
    
    @inproceedings{schoenberger2016mvs,
        author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
        title={Pixelwise View Selection for Unstructured Multi-View Stereo},
        booktitle={European Conference on Computer Vision (ECCV)},
        year={2016},
    }

如果您使用了图像检索/词汇树引擎，请同时引用：

    @inproceedings{schoenberger2016vote,
        author={Sch\"{o}nberger, Johannes Lutz and Price, True and Sattler, Torsten and Frahm, Jan-Michael and Pollefeys, Marc},
        title={A Vote-and-Verify Strategy for Fast Spatial Verification in Image Retrieval},
        booktitle={Asian Conference on Computer Vision (ACCV)},
        year={2016},
    }

最新的源代码可在 https://github.com/colmap/colmap 获取。COLMAP 建立在现有工作之上，当您在 COLMAP 中使用特定算法时，请根据源代码中的指定引用原始作者，并考虑引用相关的第三方依赖项（尤其是 ceres-solver, poselib, sift-gpu, vlfeat）。

下载
--------

* Windows 的二进制文件和其他资源可以从 https://github.com/colmap/colmap/releases 下载。
* Linux/Unix/BSD 的二进制文件可在 https://repology.org/metapackage/colmap/versions 获取。
* 预构建的 Docker 镜像可在 https://hub.docker.com/r/colmap/colmap 获取。
* Python 绑定可在 https://pypi.org/project/pycolmap 获取。
* 如需从源代码构建，请访问 https://colmap.github.io/install.html。

快速开始
---------------

1. 下载预构建的二进制文件或从源代码构建。
2. 从 https://demuc.de/colmap/datasets/ 下载提供的其中一个数据集，或使用您自己的图像。
3. 使用**自动重建**功能，只需一键或一条命令即可轻松构建模型。

文档
-------------

文档可在 https://colmap.github.io/ 查阅。

支持
-------

请使用 GitHub 讨论区 https://github.com/colmap/colmap/discussions 提出问题，以及 GitHub 问题跟踪器 https://github.com/colmap/colmap 报告错误、功能请求/添加等。

致谢
---------------

COLMAP 最初由 [Johannes Schönberger](https://demuc.de/) 编写，其博士导师 Jan-Michael Frahm 和 Marc Pollefeys 提供资金支持。

PyCOLMAP 中的 Python 绑定最初由 [Mihai Dusmanu](https://github.com/mihaidusmanu)、[Philipp Lindenberger](https://github.com/Phil26AT) 和 [Paul-Edouard Sarlin](https://github.com/Skydes) 添加。

该项目还从无数社区贡献中受益，包括错误修复、改进、新功能、第三方工具和社区支持（特别感谢 [Torsten Sattler](https://tsattler.github.io)）。

贡献
------------

非常欢迎贡献（错误报告、错误修复、改进等），应以新问题和/或 GitHub 上的拉取请求形式提交。

许可
-------

COLMAP 库在新的 BSD 许可证下授权。请注意，此文本仅指 COLMAP 本身的许可证，与其第三方依赖项分开授权。使用这些依赖项构建 COLMAP 可能会影响最终的 COLMAP 许可证。

    Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
    
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
    
        * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
          its contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.


​    
    Copyright (c) 2023, ETH Zurich 和 UNC Chapel Hill。
    版权所有。
    
    在满足以下条件的情况下，允许在源代码和二进制形式下，无论是否修改，重新分发和使用：
    
        * 重新分发源代码必须保留上述版权声明、此条件列表和以下免责声明。
    
        * 以二进制形式重新分发必须在随软件提供的文档和/或其他材料中复制上述版权声明、此条件列表和以下免责声明。
    
        * 未经特定事先书面许可，不得使用 ETH Zurich 和 UNC Chapel Hill 的名称及其贡献者的名字来支持或推广衍生自该软件的产品。
    
    本软件由版权持有者和贡献者“按原样”提供，不提供任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证。在任何情况下，即使被告知可能发生此类损害的可能性，版权持有者和贡献者也不对任何直接的、间接的、偶然的、特殊的、惩罚性的或间接损害（包括但不限于替代商品或服务的采购、使用、数据或利润的损失，或业务中断）负责，无论是基于合同责任、严格责任还是侵权行为（包括疏忽或其他原因）引起的，因使用本软件而引起的任何方式的任何责任，即使被告知此类损害的可能性。
