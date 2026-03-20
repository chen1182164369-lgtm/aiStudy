# PyTorch 与环境配置学习笔记

## 🛠 一、 环境安装与配置

### 1. Miniconda 安装与基础指令
建议从 [Anaconda 官网](https://www.anaconda.com/download) 下载 Miniconda（更轻量）。

```zsh
# 1. 检查 Conda 版本
conda --version

# 2. 创建虚拟环境 (建议环境名不要加尖括号)
conda create --name study python=3.10 

# 3. 激活环境
conda activate study

# 4. 安装核心库 (针对 Mac/Cuda 不同设备)
conda install pytorch==2.5.0 -c pytorch
pip install transformers==4.52.0
```

### 2. 算力平台
* **魔搭 ModelScope**: [我的控制台](https://www.modelscope.cn/my/overview)

---

## 🔥 二、 PyTorch 核心：张量 (Tensor)

### 1. 初始化方式
| 方式 | 说明 | 备注 |
| :--- | :--- | :--- |
| `torch.tensor()` | 函数式创建 | **推荐**，会将数据拷贝一份 |
| `torch.Tensor()` | 类实例化 | 创建空张量时速度快，数据为随机噪声 |
| `np.random.randn(2,3)` | NumPy 生成 | 2行3列随机数组 |
| `torch.arange(0, 10, 1)` | 步长生成 | 类似 Python 的 `range()` |
| `torch.linspace(0, 11, 10)` | 等分生成 | 将 0-11 等分为 10 份 |

> **💡 提示**：使用 `torch.manual_seed(100)` 可以固定随机数起点，保证实验可复现。



### 2. 数据类型与设备
* **默认类型**：`torch.randn` 默认 `float32`；从 NumPy 转换通常为 `float64`。
* **设备转移**：
  ```python
  data = data.cuda()   # 转移到 GPU
  data = data.cpu()    # 转移到 CPU (默认)
  print(data.device)   # 查看当前所在设备
  ```

---

## ⚡ 三、 基本运算

### 1. 算术运算
* **常用函数**：`add()`, `sub()`, `mul()`, `div()`, `neg()`, `abs()`。
* **就地修改 (In-place)**：方法名带下划线的会改变原数据。
  * `data.add(10)`：返回新张量，原数据不变。
  * `data.add_(10)`：**直接修改原张量**。

### 2. 矩阵乘法对比
| 运算类型 | 符号 / 函数 | 特点 |
| :--- | :--- | :--- |
| **阿达玛积** | `*` 或 `torch.mul` | 逐元素相乘，形状必须完全一致 |
| **普通矩阵乘法** | `@` 或 `torch.matmul` | 支持**广播机制**，最常用 |
| **标准矩阵乘法** | `torch.mm` | 不支持广播，要求维度严格匹配 |
| **批矩阵乘法** | `torch.bmm` | 专门用于 3 维张量 `(B, n, m) @ (B, m, p)` |



---

## 🏗 四、 张量操作

### 1. 拼接 (Join)
* **`torch.cat()`**：在原有维度拼接，**不增加维度**。
* **`torch.stack()`**：在新的维度堆叠，**会增加一个维度**。
  * *例子*：两个 `[2, 3]` 拼接，`cat` 后是 `[4, 3]`，`stack` 后是 `[2, 2, 3]`。

### 2. 索引 (Indexing)
* **范围索引**：`data[:3, :2]` (前3行前2列)。
* **布尔索引**：`data[data[:, 2] > 5]` (筛选出第3列大于5的所有行)。

### 3. 形状变换 (Shape Manipulation)
* **`reshape(1, 6)`**：改变形状，最常用。
* **`transpose(dim0, dim1)`**：交换两个维度（浅拷贝）。
* **`permute(0, 2, 1)`**：同时重新排列多个维度。
* **`view()` vs `reshape()`**：
  * **`view()`**：要求内存必须连续（`is_contiguous()`）。
  * **`reshape()`**：更智能，不连续时会自动调用 `.contiguous()`。

---

## 📂 五、 存储与深浅拷贝
* **浅拷贝 (共享内存)**：
  * `data_tensor.numpy()`
  * `torch.from_numpy(data_numpy)`
* **深拷贝 (独立内存)**：
  * `torch.tensor(data_numpy)`
* **标量转换**：`data.item()` 将仅含一个元素的张量转为 Python 标量。

---
