# 本地训练数据目录

- `turns.jsonl`：启用 `TRAINING_CAPTURE=true` 后逐轮追加的对话记录，包含用户问题、回答、工具调用和规则质检结果。
- `tool_router.pt`：`python -m training.train` 生成的 PyTorch 检查点。

这两个文件默认被 `.gitignore` 排除。原始对话属于本地数据，请勿提交到版本控制、共享目录或公网环境；启用记录前应取得用户同意并避免敏感身份信息。

未启用记录时，仍可使用 `agent/golden_cases.json` 中的金标样本训练基础模型。
