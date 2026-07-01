# KIE ComfyUI Nodes

KIE API 的 ComfyUI 自定义节点，包含 Veo、Grok 和 Seedance 系列视频模型的统一异步工作流。

## 安装

将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：

```bash
git clone https://github.com/xiaohuangzhengbang/KIE.git
```

重启 ComfyUI 后，在 `KieAI` 分类中添加节点。

## 视频工作流

1. `Kie 视频统一异步提交`：选择模型并提交任务，支持最多 9 张图像、1 个视频和 1 个音频输入；程序会按模型规则自动校验。
2. `Kie 视频系列异步查询`：无需任务 ID，一次查询本地记录的全部视频任务，并显示生成中、成功待下载、已下载、失败及错误原因。
3. `Kie 通用结果下载` 或 `Kie 视频结果下载`：无需任务 ID 和结果 URL，自动下载下一个成功且尚未下载的视频。

下载成功的任务会标记为已下载，下次运行会继续下载下一个任务。三个节点均支持强制重复运行。

## 本地数据

任务记录保存在 `kie_universal_history.json`。该文件包含本机任务历史，已加入 `.gitignore`，不会提交到仓库。

