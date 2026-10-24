/*
 * main.c - NumPy for C 的演示与自测
 * 编译：gcc np.c main.c -o main.exe
 */

#include <stdio.h>
#include "np.h"

int main(void) {
    printf("===== demo start =====\n");
    printf("\n");

    /* 1. 创建 2D 矩阵 A(2x3) 并填充 1..6 */
    int sh2[2] = {2, 3};
    np_array a = np_zeros(2, sh2);
    double v = 1.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            int idx[2] = {i, j};
            np_set(a, idx, v++);   /* [[1, 2, 3], [4, 5, 6]] */
        }
    }
    printf("A =\n"); np_print(a);
    printf("\n");

    /* 2. 元素级乘 A * A */
    np_array a2 = np_mul(a, a);    /* A * A */
    printf("np_mul(A, A) =\n"); np_print(a2);
    printf("\n");

    /* 3. 标量广播：A + 10 */
    np_array ten = np_scalar(10.0);
    np_array ap = np_add(a, ten);
    printf("np_add(A, 10) =\n"); np_print(ap);
    printf("\n");

    /* 4. 矩阵乘 A(2x3) · B(3x2) */
    int shb[2] = {3, 2};
    np_array b = np_zeros(2, shb);
    v = 1.0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            int idx[2] = {i, j};
            np_set(b, idx, v++);   /* [[1, 2], [3, 4], [5, 6]] */
        }
    }
    printf("B =\n"); np_print(b);
    printf("\n");
    np_array c = np_matmul(a, b);  /* (2x3)·(3x2) = 2x2 */
    printf("np_matmul(A, B) =\n"); np_print(c);
    printf("\n");

    /* 5. 1D 内积 */
    np_array x = np_arange(1, 4, 1);  /* [1, 2, 3] */
    np_array y = np_arange(4, 7, 1);  /* [4, 5, 6] */
    double d = np_dot(x, y);       /* 1 * 4 + 2 * 5 + 3 * 6 = 32 */
    printf("np_dot([1, 2, 3], [4, 5, 6]) = %g (expected 32)\n", d);
    printf("\n");

    /* 6. 转置 */
    np_array at = np_transpose(a);   /* 2x3 -> 3x2 */
    printf("np_transpose(A) =\n"); np_print(at);
    printf("\n");

    /* 7. 切片 A[0:2, 1:3] = 前两行，第2~3列 */
    int start[2] = {0, 1}, end[2] = {2, 3}, step[2] = {1, 1};
    np_array sl = np_slice(a, start, end, step);
    printf("np_slice(A, [0, 1], [2, 3], [1, 1]) =\n"); np_print(sl);
    printf("\n");

    /* 8. 规约 */
    printf("np_sum(A) = %g (expected 21)\n", np_sum(a));
    printf("np_mean(A) = %g (expected 3.5)\n", np_mean(a));
    np_array row_sum = np_sum_axis(a, 1);  /* 沿列求和 -> 每行的和 */
    printf("np_sum_axis(A, 1) = "); np_print(row_sum);
    printf("\n");

    /* 9. reshape：先变 1D，再变回 2D 矩阵 */
    np_array r = np_reshape(a, 1, (int[]){6});
    printf("np_reshape(A, 1, 6) =\n"); np_print(r);
    printf("\n");
    np_array r2 = np_reshape(a, 2, (int[]){3, 2});  /* 2x3 -> 3x2 二维矩阵 */
    printf("np_reshape(A, 2, [3, 2]) =\n"); np_print(r2);
    printf("\n");

    /* 10. 单位矩阵 */
    np_array e = np_eye(3);
    printf("np_eye(3) =\n"); np_print(e);
    printf("\n");

    printf("===== demo end =====\n");
    return 0;
}
