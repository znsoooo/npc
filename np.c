/*
 * np.c - NumPy 的 C 语言实现
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "np.h"

/* ---------- 辅助 ---------- */
int np_size(const np_array a) {
    int s = 1;
    for (int i = 0; i < a.ndim; i++) s *= a.shape[i];
    return s;   /* 标量 ndim==0 时循环不执行，返回 1 */
}

int np_stride(const np_array a, int i) {
    int st = 1;
    for (int k = i + 1; k < a.ndim; k++) st *= a.shape[k];
    return st;
}

/* 多维下标 -> 扁平下标（行主序） */
int np_offset(const np_array a, const int* index) {
    int off = 0;
    for (int i = 0; i < a.ndim; i++) off += index[i] * np_stride(a, i);
    return off;
}

/* 创建未初始化数组（分配 data 内容未定义） */
np_array np_empty(int ndim, const int* shape) {
    np_array a;
    a.ndim  = ndim;
    a.shape = ndim > 0 ? (int*)malloc(sizeof(int) * ndim) : NULL;
    for (int i = 0; i < ndim; i++) a.shape[i] = shape[i];
    int n = np_size(a);
    a.data = (double*)malloc(sizeof(double) * (n > 0 ? n : 1));
    return a;
}

np_array np_zeros(int ndim, const int* shape) {
    np_array a = np_empty(ndim, shape);
    int n = np_size(a);
    for (int i = 0; i < n; i++) a.data[i] = 0.0;
    return a;
}

np_array np_ones(int ndim, const int* shape) {
    np_array a = np_empty(ndim, shape);
    int n = np_size(a);
    for (int i = 0; i < n; i++) a.data[i] = 1.0;
    return a;
}

np_array np_full(int ndim, const int* shape, double value) {
    np_array a = np_empty(ndim, shape);
    int n = np_size(a);
    for (int i = 0; i < n; i++) a.data[i] = value;
    return a;
}

np_array np_scalar(double value) {
    np_array a = np_empty(0, NULL);
    a.data[0] = value;
    return a;
}

np_array np_eye(int n) {
    int shape[2] = {n, n};
    np_array a = np_zeros(2, shape);
    for (int i = 0; i < n; i++) {
        int idx[2] = {i, i};
        a.data[np_offset(a, idx)] = 1.0;
    }
    return a;
}

np_array np_arange(double start, double stop, double step) {
    if (step == 0.0) step = 1.0;
    int n = (int)((stop - start) / step);
    if (n < 0) n = 0;
    np_array a = np_empty(1, &n);
    for (int i = 0; i < n; i++) a.data[i] = start + i * step;
    return a;
}

np_array np_copy(const np_array a) {
    np_array b = np_empty(a.ndim, a.shape);
    memcpy(b.data, a.data, sizeof(double) * np_size(a));
    return b;
}

np_array np_reshape(const np_array a, int ndim, const int* shape) {
    int size = 1;
    for (int i = 0; i < ndim; i++) size *= shape[i];
    if (size != np_size(a)) {
        /* 元素总数不一致，返回空数组表示错误 */
        np_array empty = np_empty(0, NULL);
        return empty;
    }
    np_array b = np_empty(ndim, shape);
    memcpy(b.data, a.data, sizeof(double) * np_size(a));
    return b;
}

void np_free(np_array* a) {
    free(a->shape);
    free(a->data);
    a->shape = NULL;   /* 置空，释放后对象不可用，再次 np_free 安全（free(NULL) 无害） */
    a->data  = NULL;
    a->ndim  = 0;
}

/* ---------- 元素访问 ---------- */
double np_get(const np_array a, const int* index) {
    return a.data[np_offset(a, index)];
}

void np_set(np_array a, const int* index, double value) {
    a.data[np_offset(a, index)] = value;
}

/* ---------- 广播 ---------- */
/* 将扁平下标 flat 还原为多维下标写入 index（针对 r 的形状） */
static void flat_to_index(const np_array r, int flat, int* index) {
    for (int i = r.ndim - 1; i >= 0; i--) {
        index[i] = flat % r.shape[i];
        flat /= r.shape[i];
    }
}

/* 计算 a 与 b 的广播结果形状，成功返回 1，否则返回 0。
 * out_shape 需至少 max(a.ndim, b.ndim) 大小。 */
static int broadcast_shape(const np_array a, const np_array b, int* out_ndim, int* out_shape) {
    if (a.ndim == 0) {
        *out_ndim = b.ndim;
        for (int i = 0; i < b.ndim; i++) out_shape[i] = b.shape[i];
        return 1;
    }
    if (b.ndim == 0) {
        *out_ndim = a.ndim;
        for (int i = 0; i < a.ndim; i++) out_shape[i] = a.shape[i];
        return 1;
    }
    if (a.ndim != b.ndim) return 0;  /* 维数不同且非标量，暂不支持 */
    *out_ndim = a.ndim;
    for (int i = 0; i < a.ndim; i++) {
        if (a.shape[i] == b.shape[i])       out_shape[i] = a.shape[i];
        else if (a.shape[i] == 1)           out_shape[i] = b.shape[i];
        else if (b.shape[i] == 1)           out_shape[i] = a.shape[i];
        else                                return 0;  /* 不兼容，无法广播 */
    }
    return 1;
}

/* 由输出多维下标 out_index 推算输入 in 的对应下标 in_index（广播展开） */
static void broadcast_index(const int* out_index, int out_ndim, const np_array in, int* in_index) {
    if (in.ndim == 0) return;  /* 标量：下标恒为 0 */
    int offset = out_ndim - in.ndim;
    for (int i = 0; i < in.ndim; i++) {
        int oi = i + offset;
        in_index[i] = (in.shape[i] == 1) ? 0 : out_index[oi];
    }
}

/* 通用元素级运算 */
static np_array np_apply(const np_array a, const np_array b, double (*op)(double, double)) {
    int out_ndim = (a.ndim > b.ndim) ? a.ndim : b.ndim;
    int* out_shape = (int*)malloc(sizeof(int) * out_ndim);
    if (!broadcast_shape(a, b, &out_ndim, out_shape)) {
        free(out_shape);
        return np_empty(0, NULL);  /* 不支持的广播：返回空数组 */
    }
    np_array r = np_empty(out_ndim, out_shape);
    int n = np_size(r);
    int* oi = (int*)malloc(sizeof(int) * (out_ndim > 0 ? out_ndim : 1));
    int* ai = (int*)malloc(sizeof(int) * (a.ndim   > 0 ? a.ndim   : 1));
    int* bi = (int*)malloc(sizeof(int) * (b.ndim   > 0 ? b.ndim   : 1));
    for (int flat = 0; flat < n; flat++) {
        flat_to_index(r, flat, oi);
        broadcast_index(oi, out_ndim, a, ai);
        broadcast_index(oi, out_ndim, b, bi);
        r.data[flat] = op(a.data[np_offset(a, ai)], b.data[np_offset(b, bi)]);
    }
    free(oi); free(ai); free(bi); free(out_shape);
    return r;
}

static double op_add(double x, double y) { return x + y; }
static double op_sub(double x, double y) { return x - y; }
static double op_mul(double x, double y) { return x * y; }
static double op_div(double x, double y) { return x / y; }

np_array np_add(const np_array a, const np_array b) { return np_apply(a, b, op_add); }
np_array np_sub(const np_array a, const np_array b) { return np_apply(a, b, op_sub); }
np_array np_mul(const np_array a, const np_array b) { return np_apply(a, b, op_mul); }
np_array np_div(const np_array a, const np_array b) { return np_apply(a, b, op_div); }

/* ---------- 矩阵乘 ---------- */
double np_dot(const np_array a, const np_array b) {
    double s = 0.0;
    for (int i = 0; i < a.shape[0]; i++) s += a.data[i] * b.data[i];
    return s;
}

/* 仅处理 2D·2D；1D 情形由 wrapper 升维后调用 */
static np_array matmul_2d(const np_array a, const np_array b) {
    int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    int shape[2] = {m, n};
    np_array r = np_zeros(2, shape);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int t = 0; t < k; t++) {
                int ai[2] = {i, t};
                int bi[2] = {t, j};
                sum += a.data[np_offset(a, ai)] * b.data[np_offset(b, bi)];
            }
            int ri[2] = {i, j};
            r.data[np_offset(r, ri)] = sum;
        }
    }
    return r;
}

np_array np_matmul(const np_array a, const np_array b) {
    /* 1D · 1D -> 标量 */
    if (a.ndim == 1 && b.ndim == 1) {
        return np_scalar(np_dot(a, b));
    }
    /* 2D · 2D */
    if (a.ndim == 2 && b.ndim == 2) {
        return matmul_2d(a, b);
    }
    /* 1D · 2D：把 a 视作行向量 (1, k) */
    if (a.ndim == 1 && b.ndim == 2) {
        int s1[1] = {a.shape[0]};
        np_array a2 = np_reshape(a, 1, s1);
        np_array r  = matmul_2d(a2, b);
        np_array out = np_reshape(r, 1, &r.shape[1]);
        np_free(&a2); np_free(&r);
        return out;
    }
    /* 2D · 1D：把 b 视作列向量 (k, 1) */
    if (a.ndim == 2 && b.ndim == 1) {
        np_array b2 = np_reshape(b, 2, (int[]){b.shape[0], 1});
        np_array r  = matmul_2d(a, b2);
        np_array out = np_reshape(r, 1, &r.shape[0]);
        np_free(&b2); np_free(&r);
        return out;
    }
    /* 不支持的高维情形 */
    return np_empty(0, NULL);
}

/* ---------- 切片 ---------- */
np_array np_slice(const np_array a, const int* start, const int* end, const int* step) {
    int* s = (int*)malloc(sizeof(int) * a.ndim);
    int* e = (int*)malloc(sizeof(int) * a.ndim);
    int* st = (int*)malloc(sizeof(int) * a.ndim);
    int* new_shape = (int*)malloc(sizeof(int) * a.ndim);
    for (int i = 0; i < a.ndim; i++) {
        s[i]  = (start[i] >= 0) ? start[i] : 0;
        e[i]  = (end[i]   >= 0) ? end[i]   : a.shape[i];
        st[i] = (step[i]  > 0)  ? step[i]  : 1;
        if (s[i] < 0) s[i] = 0;
        if (e[i] > a.shape[i]) e[i] = a.shape[i];
        if (e[i] < s[i]) e[i] = s[i];
        new_shape[i] = (e[i] - s[i] + st[i] - 1) / st[i];
        if (new_shape[i] < 0) new_shape[i] = 0;
    }
    np_array r = np_empty(a.ndim, new_shape);
    int n = np_size(r);
    int* oi = (int*)malloc(sizeof(int) * a.ndim);
    int* ii = (int*)malloc(sizeof(int) * a.ndim);
    for (int flat = 0; flat < n; flat++) {
        flat_to_index(r, flat, oi);
        for (int i = 0; i < a.ndim; i++) ii[i] = s[i] + oi[i] * st[i];
        r.data[flat] = a.data[np_offset(a, ii)];
    }
    free(s); free(e); free(st); free(new_shape); free(oi); free(ii);
    return r;
}

/* ---------- 转置（逆序所有轴） ---------- */
np_array np_transpose(const np_array a) {
    int* new_shape = (int*)malloc(sizeof(int) * a.ndim);
    for (int i = 0; i < a.ndim; i++) new_shape[i] = a.shape[a.ndim - 1 - i];
    np_array r = np_empty(a.ndim, new_shape);
    int n = np_size(r);
    int* oi = (int*)malloc(sizeof(int) * a.ndim);
    int* ii = (int*)malloc(sizeof(int) * a.ndim);
    for (int flat = 0; flat < n; flat++) {
        flat_to_index(r, flat, oi);
        for (int i = 0; i < a.ndim; i++) ii[i] = oi[a.ndim - 1 - i];
        r.data[flat] = a.data[np_offset(a, ii)];
    }
    free(new_shape); free(oi); free(ii);
    return r;
}

/* ---------- 规约（全数组） ---------- */
double np_sum(const np_array a) {
    double s = 0.0;
    int n = np_size(a);
    for (int i = 0; i < n; i++) s += a.data[i];
    return s;
}
double np_mean(const np_array a) {
    int n = np_size(a);
    return n ? np_sum(a) / n : 0.0;
}
double np_max(const np_array a) {
    double m = a.data[0];
    int n = np_size(a);
    for (int i = 1; i < n; i++) if (a.data[i] > m) m = a.data[i];
    return m;
}
double np_min(const np_array a) {
    double m = a.data[0];
    int n = np_size(a);
    for (int i = 1; i < n; i++) if (a.data[i] < m) m = a.data[i];
    return m;
}

/* 沿 axis 轴求和/求平均，结果去掉该轴 */
np_array np_sum_axis(const np_array a, int axis) {
    int* new_shape = (int*)malloc(sizeof(int) * (a.ndim - 1));
    for (int i = 0, j = 0; i < a.ndim; i++)
        if (i != axis) new_shape[j++] = a.shape[i];
    np_array r = np_zeros(a.ndim - 1, new_shape);
    int* oi = (int*)malloc(sizeof(int) * a.ndim);
    int* ii = (int*)malloc(sizeof(int) * a.ndim);
    int* ri = (int*)malloc(sizeof(int) * (a.ndim - 1));
    int n = np_size(a);
    for (int flat = 0; flat < n; flat++) {
        flat_to_index(a, flat, ii);
        double v = a.data[flat];
        /* 输出下标：去掉 axis 维 */
        for (int i = 0, j = 0; i < a.ndim; i++)
            if (i != axis) ri[j++] = ii[i];
        int rflat = 0;
        for (int i = 0; i < r.ndim; i++) rflat += ri[i] * np_stride(r, i);
        r.data[rflat] += v;
    }
    free(new_shape); free(oi); free(ii); free(ri);
    return r;
}

np_array np_mean_axis(const np_array a, int axis) {
    np_array r = np_sum_axis(a, axis);
    double count = a.shape[axis];
    int n = np_size(r);
    for (int i = 0; i < n; i++) r.data[i] /= count;
    return r;
}

/* ---------- 调试输出 ---------- */

/* 打印以 (fixed, fixed + 1) 两维构成的“二维块”。
 * indent：该块外层 [ 所在的列号，块内后续行据此对齐缩进。 */
static void print_2d_block(const np_array a, int* idx, int fixed, int indent) {
    int rows = a.shape[fixed];
    int cols = a.shape[fixed + 1];
    printf("[");
    for (int i = 0; i < rows; i++) {
        if (i > 0) printf("\n%*s", indent + 1, "");  /* 行首缩进 1 格，对齐首行 [ */
        printf("[");
        idx[fixed] = i;
        for (int j = 0; j < cols; j++) {
            idx[fixed + 1] = j;
            if (j > 0) printf(", ");
            printf("%g", a.data[np_offset(a, idx)]);
        }
        printf("]");
        if (i < rows - 1) printf(",");
    }
    printf("]");
}

/* 递归打印：axis 为当前遍历的轴；indent 为当前 [ 所在列号。
 *   - 轴 == ndim：叶子（标量元素）
 *   - 轴 == ndim - 2：最后一二维，打成二维块
 *   - 其余：逐元素包 [ ]，顶层（axis==0）块间留一空行 */
static void np_print_rec(const np_array a, int* idx, int axis, int indent) {
    if (axis == a.ndim) {
        printf("%g", a.data[np_offset(a, idx)]);
        return;
    }
    if (axis == a.ndim - 2) {
        print_2d_block(a, idx, axis, indent);
        return;
    }
    printf("[");
    int count = a.shape[axis];
    for (int i = 0; i < count; i++) {
        if (i > 0) {
            if (axis == 0) printf(",\n\n%*s", indent + 1, ""); /* 顶层块间空行 */
            else           printf(",\n%*s",   indent + 1, "");
        }
        idx[axis] = i;
        np_print_rec(a, idx, axis + 1, indent + 1);
    }
    printf("]");
}

void np_print(const np_array a) {
    if (a.ndim == 0) { printf("[%g]\n", a.data[0]); return; }
    if (np_size(a) == 0) { printf("[]\n"); return; }
    if (a.ndim == 1) {                 /* 1D：一行 Python 列表 */
        printf("[");
        int n = a.shape[0];
        for (int i = 0; i < n; i++) {
            if (i > 0) printf(", ");
            printf("%g", a.data[i]);
        }
        printf("]\n");
        return;
    }
    int* idx = (int*)calloc(a.ndim, sizeof(int));
    np_print_rec(a, idx, 0, 0);
    printf("\n");
    free(idx);
}
