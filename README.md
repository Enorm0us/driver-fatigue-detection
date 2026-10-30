# 驾驶员疲劳检测（dlib 68 点 + 眼部面积占比 + MLP）

用 dlib 的人脸检测与 68 点关键点模型，把「睁眼 / 闭眼」这两件事变成一个**二维数值特征**问题：左右眼轮廓面积占人脸面积的比例。再用一个轻量 MLP 做二分类，判断驾驶员是否疲劳。

不直接跑卷积网络检测眼睛，是因为这个任务的重点在**特征怎么来的**，而不是模型有多深 —— 两个浮点数就能表达眼睛的闭合程度，模型只需要学一条决策边界。整套流程在普通 CPU 上几秒就能训完。

这是《人工智能开发框架应用》课程实训的个人独立作业。五个 notebook 按编号顺序执行。

## 结果

| 指标 | 数值 |
| --- | --- |
| 数据量 | 睁眼 1,226 张 + 闭眼 1,002 张 = 2,228 张 |
| 切分 | 8:2 → 训练 1,783 / 验证 445 |
| 特征维度 | 2（左眼占比、右眼占比） |
| 训练轮数 | 200 epoch，batch = 32 |
| 验证准确率 | **73.93%**（第 200 轮） |
| 峰值 | **75.28%**（第 132 轮） |
| 训练耗时 | 每 epoch 约 0.3s（纯 CPU） |

## 核心发现：验证准确率卡在 54.83%

前 64 个 epoch，`val_accuracy` 一动不动地钉在 **0.5483**（即 244/445，几乎等于「全猜同一类」的基线），`val_loss` 从 0.6922 缓慢降到 0.6871 —— 模型在学，但学不到能越过决策边界的程度。

```
Epoch   1  val_loss 0.6922  val_acc 0.5483
Epoch   2  val_loss 0.6914  val_acc 0.5483
...
Epoch  64  val_loss 0.687x  val_acc 0.5483   <- 连续 64 轮纹丝不动
Epoch  65  val_loss 0.6677  val_acc 0.5528   <- 开始松动
Epoch  72  val_loss 0.6620  val_acc 0.5888
Epoch  90  val_loss 0.6432  val_acc 0.6854
Epoch 118  val_loss 0.6087  val_acc 0.7393
Epoch 132  val_loss 0.5937  val_acc 0.7528   <- 峰值
Epoch 200  val_loss 0.5630  val_acc 0.7393
```

原因不是数据坏了、也不是网络太浅，而是 **Adam 的默认学习率 1e-3 对这类「只有两个特征、量级在 1e-3」的输入偏大**：梯度更新一步就跨过了最优解所在的窄谷，参数在谷底两侧反复弹跳，loss 几乎不降。

把学习率改成 `Adam(0.0001)` 之后，第 65 个 epoch 左右模型才开始明显收敛，验证准确率一路爬到 75% 附近。这个排查过程说明：**超参数不能无脑用默认值**，尤其是特征维度极低时，学习率的量级要和特征量级匹配。

## 为什么用「面积占比」而不是直接喂关键点

68 个点的绝对坐标会随人脸在画面中的远近、位置、大小剧烈变化 —— 同一张闭眼照片，人在远处时点和近处时完全不是一组数。所以做了两层归一化：

1. **减原点**：所有点减去人脸检测框左上角 `d.left()` / `d.top()`，把坐标变成**相对于人脸框**的位置，消除平移。
2. **除以人脸面积**：`cv2.contourArea(眼轮廓) / d.area()`，得到无量纲的占比，消除缩放。

这样同一双眼睛在 1 米和 3 米处算出来的特征基本一致。实测睁眼样本的左右眼占比都在 0.003~0.009 量级，闭眼会显著更小。

## 流程

```
1_DataCleaning.ipynb    dlib 过滤掉检测不到人脸的废图
2_FaceLandmark.ipynb    提取 68 点相对坐标 + 人脸面积
3_DataIntegration.ipynb 算左右眼占比，打标签，8:2 切分导出 csv
4_Training.ipynb        DenseFeatures + MLP，200 epoch
5_Predict.ipynb         对单张图片推理
```

**1_DataCleaning**：`glob` 扫 `*.jpg`，逐张 `cv2.imread`。注意 `cv2.imread` 读出来是 **BGR**，而 dlib 期待 **RGB**，不转通道会检测失败或框到莫名其妙的位置 —— 代码里显式做了 `cv.cvtColor(frame, cv.COLOR_BGR2RGB)`。`detector(img, 1)` 的第二个参数 `upsample=1` 表示先上采样一次再检测，小脸更容易被找到。检测不到人脸的图 `shutil.move` 到 `*Failed/` 目录。

实跑：睁眼 1,226 张 2 分 11 秒（9.34 it/s），闭眼 1,002 张 1 分 40 秒（9.93 it/s）。

**2_FaceLandmark**：`dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")` 对每张脸出 68 个点。`d = dets[0]` 只取第一张脸（这个数据集每张图只有一个人）。循环里把 `shape.part(i).x - oriX` 写进 `facePoints`，最终 `data` 形状 `(N, 68, 2)`，`area` 形状 `(N,)`，面积用 `d.area()`。

**3_DataIntegration**：左眼取点 36–41、右眼取点 42–47（dlib 68 点的标准眼周索引），`cv2.contourArea` 算轮廓面积。**`contourArea` 要求点集是 `int32`**，直接传 `float64` 会抛异常，所以先 `astype(np.int32)`。睁眼标 `isTired=0`、闭眼标 `isTired=1`，`np.random.permutation` 打乱后按 8:2 切分，导出 `train.csv` / `eval.csv`，三列 `leftEye, rightEye, isTired`。

## 模型

```
DenseFeatures(feature_columns)   # 两个 numeric_column
Dense(128, relu)
Dense(64,  relu)
Dense(1,  sigmoid)
```

损失 `binary_crossentropy`，优化器 **`Adam(0.0001)`**，指标 `accuracy`。数据管道用 `tf.data.experimental.CsvDataset(paths, [tf.float32, tf.float32, tf.int32], header=True)`，`.map(parseCSV)` 把每行包成 `{'leftEye':…, 'rightEye':…}` 字典，`.batch(32)`。

训练时 TensorFlow 会刷一堆 `WARNING:tensorflow:Layers in a Sequential model should only have a single input tensor` —— 因为 `DenseFeatures` 输入是字典，却塞在 `Sequential` 里。不影响结果，但按提示改成 Functional API 就能消掉。

模型存为 `./model/keras_nn`（SavedModel 格式，不是单个 `.h5`）。

## 推理

`5_Predict.ipynb` 对单张图：检测人脸 → 提 68 点 → 算左右眼占比 → `model.predict({'leftEye': …, 'rightEye': …})` → 输出 < 0.5 判"未疲劳"，≥ 0.5 判"疲劳"。

`samples/open.jpg` 与 `samples/closed.jpg` 是两张实测样图。

## 已知问题（代码里留着的真实 bug）

`5_Predict.ipynb` 的 `calEyeArea` 把 `cv.contourArea(leftEye) / d.area()` 写在了 `for point in [36..41]` 循环**内部**，每加一个点就重算一次面积。结果碰巧正确 —— 因为最后一轮循环时 `leftEye` 已经累积了全部 6 个点，最后一次计算的就是完整轮廓的面积 —— 但逻辑是错的，且多算了 5 次。正确写法是把这两行挪到循环结束后。

仓库里的代码**保留了原始版本**（和实训提交的一致），这个 bug 正在上面的位置，属于我事后复盘时发现并记录下来的问题，没有回头改 notebook 里的历史输出。

## 目录

```
1_DataCleaning.ipynb    dlib 过滤废图
2_FaceLandmark.ipynb    68 点相对坐标 + 人脸面积
3_DataIntegration.ipynb 眼面积占比 + 标签 + 切分
4_Training.ipynb        MLP 训练（含完整 200 epoch 日志）
5_Predict.ipynb         单图推理
model/keras_nn/         SavedModel 格式的成品模型
samples/                open.jpg / closed.jpg 两张样图
```

> `model/shape_predictor_68_face_landmarks.dat`（约 95 MB）是 dlib 官方模型，**不在仓库里**，需要自己下载后放到 `model/` 下。

## 运行

```bash
pip install tensorflow numpy opencv-python dlib pandas tqdm
```

Windows 下 dlib 直接 `pip install dlib` 常需要编译，建议下对应 Python 版本的离线 whl：

```bash
pip install dlib-19.22.99-cp39-cp39-win_amd64.whl   # Python 3.9
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl    # Python 3.8
```

数据目录 `./data/OpenFace/` 与 `./data/ClosedFace/` 需自行准备。

## 可以继续改的地方

- 只用左右眼两个特征，丢掉了眼睑开合之外的姿态信息。可以把 68 点直接喂给一个浅层网络（先归一化），或补上嘴巴张开度、眉毛高度等特征。
- 每张图只用 `dets[0]`。换成真实车载视频时会有一帧多人的情况，应该取面积最大的那张脸。
- `2_FaceLandmark` 里用 `np.concatenate` 逐点拼接，1,226 张图要 2 分半。改成先存 list 再一次性 `np.array` 会快一个数量级。
- 没有做时序平滑。真实场景里单帧误判很常见，加一个 5~10 帧的滑动窗口投票会稳得多。
