/*
 * np.h - 一个纯 C 语言实现的、类似 NumPy 的多元数组库
 *
 * 实现目标：
 *   1. 纯 C 语言实现，仅依赖标准库。
 *   2. 支持多维数组。
 *   3. 支持实现 numpy 的常用函数，包括加、减、乘、除、矩阵乘、点乘、切片、转置、规约等计算方法。
 *   4. 以可读性与直观计算为优先，不做性能优化。
 *   5. 函数签名可能与 numpy 不一致，以最清晰的方式命名。
 *
 * 数组采用 C 行主序（row-major）存储：data[k] 与多维下标通过 np_offset 互转。
 * 结构体只保存形态（ndim/shape/data），过程量（size、strides）由 np_size / np_stride 按需计算。
 */

#ifndef NP_H
#define NP_H

/* 多元数组结构体：基本形态信息 */
typedef struct {
    int     ndim;     /* 维数，0 表示标量 */
    int*    shape;    /* 各维长度，长度 = ndim */
    double* data;     /* 数据区 */
} np_array;

/* ---------- 辅助 ---------- */
int  np_size  (const np_array a);          /* 元素总数 = shape 各维乘积（标量返回 1） */
int  np_stride(const np_array a, int i);   /* 行主序下第 i 维步长 = shape[i+1..] 乘积，末维为 1 */

/* ---------- 创建 / 释放 ---------- */
np_array np_empty (int ndim, const int* shape);
np_array np_zeros (int ndim, const int* shape);
np_array np_ones  (int ndim, const int* shape);
np_array np_full  (int ndim, const int* shape, double value);
np_array np_scalar(double value);                 /* 0 维标量 */
np_array np_eye   (int n);                        /* 2D 单位矩阵 */
np_array np_arange(double start, double stop, double step); /* 1D 等差序列 */
np_array np_copy  (const np_array a);
np_array np_reshape(const np_array a, int ndim, const int* shape);
void    np_free  (np_array* a);                  /* 取指针：释放后置空 shape/data，避免悬空 */

/* ---------- 元素访问 ---------- */
int     np_offset(const np_array a, const int* index); /* 多维下标 -> 扁平下标 */
double  np_get   (const np_array a, const int* index);
void    np_set   (np_array a, const int* index, double value);

/* ---------- 元素级运算（含简单广播：同形 / 标量 / 尾维对齐） ---------- */
np_array np_add(const np_array a, const np_array b);
np_array np_sub(const np_array a, const np_array b);
np_array np_mul(const np_array a, const np_array b);
np_array np_div(const np_array a, const np_array b);

/* ---------- 矩阵乘（@）：1D·1D=标量, 2D·2D=2D, 1D·2D / 2D·1D=1D ---------- */
np_array np_matmul(const np_array a, const np_array b);
double  np_dot   (const np_array a, const np_array b); /* 1D 内积 */

/* ---------- 切片 / 转置 ---------- */
np_array np_slice(const np_array a, const int* start, const int* end, const int* step);
np_array np_transpose(const np_array a);

/* ---------- 规约（全数组） ---------- */
double  np_sum (const np_array a);
double  np_mean(const np_array a);
double  np_max (const np_array a);
double  np_min (const np_array a);
np_array np_sum_axis (const np_array a, int axis);
np_array np_mean_axis(const np_array a, int axis);

/* ---------- 调试输出 ---------- */
void    np_print(const np_array a);

#endif /* NP_H */
