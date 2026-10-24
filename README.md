# NumPy for C

NumPy 的纯 C 语言移植版。

本仓库的目标：把 NumPy 中常用函数与魔术方法（矩阵加减乘除、矩阵乘、切片、转置、规约等）
以**纯C语言**实现，强调**可读性与直观计算**，而非性能。

## 设计约束

- **纯C语言**，仅依赖标准库（`stdlib.h` / `string.h` / `math.h` / `stdio.h`），
  不依赖任何 Windows 或 Linux 专有 API，可在任意符合 C99 的编译器上编译。
- **支持多维数组**：`ndim` 可为 0（标量）、1、2 乃至更高维，不仅是二维矩阵。
- **函数签名允许与NumPy不一致**，以最清晰的命名方式表达（如 `np_matmul`、`np_sum_axis`）。
- **复杂广播暂不处理**：元素级运算支持「同形 / 标量 / 尾维对齐」三种广播；
  维数不同且非标量时返回空数组（`ndim==0`）表示不支持。
- 不做性能优化，矩阵乘等按最直观的三重循环实现，便于阅读与教学。

## 编译与运行

```shell
gcc np.c main.c -o main.exe
main.exe
```

## 核心数据结构

```c
typedef struct {
    int     ndim;     /* 维数，0 表示标量 */
    int*    shape;    /* 各维长度 */
    double* data;     /* 数据区 */
} np_array;
```

> 结构体只保存形态信息；过程量不随对象存储：`np_size(a)` 返回元素总数，`np_stride(a, i)` 返回第 `i` 维的行主序步长，按需计算。
> 数组采用 C 行主序存储；多维下标与扁平下标通过 `np_offset` 互转。
> 函数以 `np_array` 传参，`const` 表示只读；底层 `data`/`shape` 通过共享指针访问。

## API 一览

| 类别 | 函数 | 说明 |
| --- | --- | --- |
| 创建 | `np_empty` / `np_zeros` / `np_ones` / `np_full` / `np_scalar` | 各种初始化 |
| 创建 | `np_eye(n)` | 2D 单位矩阵 |
| 辅助 | `np_size` / `np_stride` | 计算元素总数 / 各维步长 |
| 创建 | `np_arange(start, stop, step)` | 1D 等差序列 |
| 视图 | `np_copy` / `np_reshape` | 复制 / 改变形状（总数须一致） |
| 访问 | `np_get` / `np_set` / `np_offset` | 元素读写 |
| 元素级 | `np_add` / `np_sub` / `np_mul` / `np_div` | 含简单广播 |
| 矩阵乘 | `np_matmul` / `np_dot` | 1D·1D=标量, 2D·2D=2D, 1D·2D/2D·1D=1D |
| 切片 | `np_slice(start, end, step)` | 各维 `start/end/step`，`-1` 表示默认 |
| 转置 | `np_transpose` | 逆序所有轴（2D 即标准转置） |
| 规约 | `np_sum` / `np_mean` / `np_max` / `np_min` | 全数组规约 |
| 规约 | `np_sum_axis` / `np_mean_axis` | 沿指定轴规约 |
| 调试 | `np_print` / `np_free` | 打印 / 释放 |

## 示例

```c
int shape[2] = {2, 3};
np_array a = np_zeros(2, shape);
np_set(a, (int[]){0, 0}, 1.0);    /* 设置 a[0][0] = 1.0 */

np_array b = np_ones(2, shape);
np_array c = np_add(a, b);        /* 元素级相加，自动广播 */
np_array m = np_matmul(a, c);     /* 矩阵乘 */

np_print(m);
np_free(&a); np_free(&b); np_free(&c); np_free(&m);
```
