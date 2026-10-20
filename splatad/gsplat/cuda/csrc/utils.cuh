#ifndef GSPLAT_CUDA_UTILS_H
#define GSPLAT_CUDA_UTILS_H

#include "helpers.cuh"
#include "third_party/glm/glm/glm.hpp"
#include "third_party/glm/glm/gtc/type_ptr.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#define RAD_TO_DEG 57.2957795131f

inline __device__ void depth_compensation_from_cov3d(const glm::mat3 cov3d, const float eps2d, glm::vec2 &depth_compensation) {
    // cov3d is a 3x3 matrix [a1, a2, a3; b1, b2, b3; c1, c2, c3]
    // but glm::mat3 is column major
    // depth_compensation is a 2x1 vector [dc1; dc2]
    // in rowmajor terms: [invcov3d[2,0]/invcov3d[2,2]; invcov3d[2,1]/invcov3d[2,2]]
    // we can compute this by inverting the 3x3 matrix and taking the 3rd column
    // hence, we don't have to compute the full inverse
    float a1 = cov3d[0][0] + eps2d;
    float a2 = cov3d[1][0];
    float b1 = cov3d[0][1];
    float b2 = cov3d[1][1] + eps2d;
    float c1 = cov3d[0][2];
    float c2 = cov3d[1][2];

    float invD = 1/(a1*b2-a2*b1);
    depth_compensation = glm::vec2(
        (b1*c2-b2*c1) * invD,
        (a2*c1-a1*c2) * invD
    );
}

inline __device__ void depth_compensation_from_cov3d_vjp(const glm::mat3 cov3d, const float eps2d, const glm::vec2 v_depth_compensation, glm::mat3 &v_cov3d) {
    float a1 = cov3d[0][0] + eps2d;
    float a2 = cov3d[1][0];
    float b1 = cov3d[0][1];
    float b2 = cov3d[1][1] + eps2d;
    float c1 = cov3d[0][2];
    float c2 = cov3d[1][2];

    float invD = 1/(a1*b2-a2*b1);

    float dc1 = (b1*c2-b2*c1) * invD;
    float dc2 = (a2*c1-a1*c2) * invD;

    v_cov3d[0][0] += (v_depth_compensation[0] * (-dc1*b2) + v_depth_compensation[1] * (-c2-dc2*b2))*invD;
    v_cov3d[1][0] += (v_depth_compensation[0] * (dc1*b1) + v_depth_compensation[1] * (c1+dc2*b1))*invD;
    //v_cov3d[2][0] += 0.f;
    v_cov3d[0][1] += (v_depth_compensation[0] * (c2 + dc1*a2) + v_depth_compensation[1] * (dc2*a2))*invD;
    v_cov3d[1][1] += (v_depth_compensation[0] * (-c1 - dc1*a1) + v_depth_compensation[1] * (-dc2*a1))*invD;
    //v_cov3d[2][1] += 0.f;
    v_cov3d[0][2] += (v_depth_compensation[0] * (-b2) + v_depth_compensation[1] * (a2))*invD;
    v_cov3d[1][2] += (v_depth_compensation[0] * (b1) + v_depth_compensation[1] * (-a1))*invD;
    //v_cov3d[2][2] += 0.f;
}

inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rnorm4df(x, y, z, w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return glm::mat3((1.f - 2.f * (y2 + z2)), (2.f * (xy + wz)),
                     (2.f * (xz - wy)), // 1st col
                     (2.f * (xy - wz)), (1.f - 2.f * (x2 + z2)),
                     (2.f * (yz + wx)), // 2nd col
                     (2.f * (xz + wy)), (2.f * (yz - wx)),
                     (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}

inline __device__ void quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R,
                                          glm::vec4 &v_quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rnorm4df(x, y, z, w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    glm::vec4 v_quat_n = glm::vec4(
        2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
               z * (v_R[0][1] - v_R[1][0])),
        2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
               z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])),
        2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
               z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])),
        2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
               2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])));

    glm::vec4 quat_n = glm::vec4(w, x, y, z);
    v_quat += (v_quat_n - glm::dot(v_quat_n, quat_n) * quat_n) * inv_norm;
}

inline __device__ void quat_scale_to_covar_preci(const glm::vec4 quat,
                                                 const glm::vec3 scale,
                                                 // optional outputs
                                                 glm::mat3 *covar, glm::mat3 *preci) {
    glm::mat3 R = quat_to_rotmat(quat);
    if (covar != nullptr) {
        // C = R * S * S * Rt
        glm::mat3 S =
            glm::mat3(scale[0], 0.f, 0.f, 0.f, scale[1], 0.f, 0.f, 0.f, scale[2]);
        glm::mat3 M = R * S;
        *covar = M * glm::transpose(M);
    }
    if (preci != nullptr) {
        // P = R * S^-1 * S^-1 * Rt
        glm::mat3 S = glm::mat3(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f,
                                0.f, 0.f, 1.0f / scale[2]);
        glm::mat3 M = R * S;
        *preci = M * glm::transpose(M);
    }
}

inline __device__ void quat_scale_to_covar_vjp(
    // fwd inputs
    const glm::vec4 quat, const glm::vec3 scale,
    // precompute
    const glm::mat3 R,
    // grad outputs
    const glm::mat3 v_covar,
    // grad inputs
    glm::vec4 &v_quat, glm::vec3 &v_scale) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = scale[0], sy = scale[1], sz = scale[2];

    // M = R * S
    glm::mat3 S = glm::mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    glm::mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    glm::mat3 v_M = (v_covar + glm::transpose(v_covar)) * M;
    glm::mat3 v_R = v_M * S;

    // grad for (quat, scale) from covar
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] += R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2];
    v_scale[1] += R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2];
    v_scale[2] += R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2];
}

inline __device__ void quat_scale_to_preci_vjp(
    // fwd inputs
    const glm::vec4 quat, const glm::vec3 scale,
    // precompute
    const glm::mat3 R,
    // grad outputs
    const glm::mat3 v_preci,
    // grad inputs
    glm::vec4 &v_quat, glm::vec3 &v_scale) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    float sx = 1.0f / scale[0], sy = 1.0f / scale[1], sz = 1.0f / scale[2];

    // M = R * S
    glm::mat3 S = glm::mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    glm::mat3 M = R * S;

    // https://math.stackexchange.com/a/3850121
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    // so
    // for D = M * Mt,
    // df/dM = df/dM + df/dMt = G * M + (Mt * G)t = G * M + Gt * M
    glm::mat3 v_M = (v_preci + glm::transpose(v_preci)) * M;
    glm::mat3 v_R = v_M * S;

    // grad for (quat, scale) from preci
    quat_to_rotmat_vjp(quat, v_R, v_quat);

    v_scale[0] +=
        -sx * sx * (R[0][0] * v_M[0][0] + R[0][1] * v_M[0][1] + R[0][2] * v_M[0][2]);
    v_scale[1] +=
        -sy * sy * (R[1][0] * v_M[1][0] + R[1][1] * v_M[1][1] + R[1][2] * v_M[1][2]);
    v_scale[2] +=
        -sz * sz * (R[2][0] * v_M[2][0] + R[2][1] * v_M[2][1] + R[2][2] * v_M[2][2]);
}

inline __device__ void persp_proj(
    // inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float fx, const float fy,
    const float cx, const float cy, const uint32_t width, const uint32_t height,
    // outputs
    glm::mat2 &cov2d, glm::vec2 &mean2d) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(fx * rz, 0.f,                  // 1st column
                                0.f, fy * rz,                  // 2nd column
                                -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );
    cov2d = J * cov3d * glm::transpose(J);
    mean2d = glm::vec2({fx * x * rz + cx, fy * y * rz + cy});
}

inline __device__ void persp_proj_vjp(
    // fwd inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float fx, const float fy,
    const float cx, const float cy, const uint32_t width, const uint32_t height,
    // grad outputs
    const glm::mat2 v_cov2d, const glm::vec2 v_mean2d,
    // grad inputs
    glm::vec3 &v_mean3d, glm::mat3 &v_cov3d) {
    float x = mean3d[0], y = mean3d[1], z = mean3d[2];

    float tan_fovx = 0.5f * width / fx;
    float tan_fovy = 0.5f * height / fy;
    float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    float lim_y_neg = cy / fy + 0.3f * tan_fovy;

    float rz = 1.f / z;
    float rz2 = rz * rz;
    float tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    float ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(fx * rz, 0.f,                  // 1st column
                                0.f, fy * rz,                  // 2nd column
                                -fx * tx * rz2, -fy * ty * rz2 // 3rd column
    );

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    // df/dx = fx * rz * df/dpixx
    // df/dy = fy * rz * df/dpixy
    // df/dz = - fx * mean.x * rz2 * df/dpixx - fy * mean.y * rz2 * df/dpixy
    v_mean3d += glm::vec3(fx * rz * v_mean2d[0], fy * rz * v_mean2d[1],
                          -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2);

    // df/dx = -fx * rz2 * df/dJ_02
    // df/dy = -fy * rz2 * df/dJ_12
    // df/dz = -fx * rz2 * df/dJ_00 - fy * rz2 * df/dJ_11
    //         + 2 * fx * tx * rz3 * df/dJ_02 + 2 * fy * ty * rz3
    float rz3 = rz2 * rz;
    glm::mat3x2 v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    // fov clipping
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean3d.x += -fx * rz2 * v_J[2][0];
    } else {
        v_mean3d.z += -fx * rz3 * v_J[2][0] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean3d.y += -fy * rz2 * v_J[2][1];
    } else {
        v_mean3d.z += -fy * rz3 * v_J[2][1] * ty;
    }
    v_mean3d.z += -fx * rz2 * v_J[0][0] - fy * rz2 * v_J[1][1] +
                  2.f * fx * tx * rz3 * v_J[2][0] + 2.f * fy * ty * rz3 * v_J[2][1];
}

inline __device__ void lidar_proj(
    const glm::vec3& __restrict__ mean3d,
    const glm::mat3& __restrict__ cov3d,
    const float eps2d,
    glm::vec2 &mean2d,
    glm::mat2 &cov2d,
    glm::vec2 &depth_compensation,
    glm::mat3 &jacobian
) {
    const float x2 = mean3d.x * mean3d.x;
    const float y2 = mean3d.y * mean3d.y;
    const float z2 = mean3d.z * mean3d.z;
    const float xz = mean3d.x * mean3d.z;
    const float yz = mean3d.y * mean3d.z;
    const float r2 = x2 + y2 + z2;
    const float rinv = rnorm3df(mean3d.x, mean3d.y, mean3d.z);
    const float sqrtx2y2 = hypotf(mean3d.x, mean3d.y);
    const float sqrtx2y2_inv = rhypotf(mean3d.x, mean3d.y);
    const float r2sqrtx2y2_inv = 1.f / (r2) * sqrtx2y2_inv;
    // column major
    // we only care about the top 2x2 submatrix
    jacobian = glm::mat3(
        -mean3d.y / (x2 + y2) * RAD_TO_DEG, -xz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.x * rinv, // 1st column
        mean3d.x / (x2 + y2)  * RAD_TO_DEG, -yz * r2sqrtx2y2_inv * RAD_TO_DEG, mean3d.y * rinv, // 2nd column
        0.f                               , sqrtx2y2 / r2  * RAD_TO_DEG      , mean3d.z * rinv  // 3rd column
    );

    glm::mat3 cov3d_spherical = jacobian * cov3d * glm::transpose(jacobian);

    cov2d = glm::mat2(cov3d_spherical[0][0], cov3d_spherical[0][1],
                      cov3d_spherical[1][0], cov3d_spherical[1][1]);

    mean2d = glm::vec2(
        glm::atan(mean3d.y, mean3d.x) * RAD_TO_DEG, glm::asin(mean3d.z * rinv) * RAD_TO_DEG
    );

    depth_compensation_from_cov3d(cov3d_spherical, eps2d, depth_compensation);
}

inline __device__ void lidar_proj_vjp(
    // fwd inputs
    const glm::vec3 mean3d, const glm::mat3 cov3d, const float eps2d,
    // grad outputs
    const glm::vec2 v_mean2d, const glm::mat2 v_cov2d, const glm::vec2 v_depth_compensation,
    // grad inputs
    glm::vec3 &v_mean3d, glm::mat3 &v_cov3d) {

    const float x2 = mean3d.x * mean3d.x;
    const float y2 = mean3d.y * mean3d.y;
    const float z2 = mean3d.z * mean3d.z;
    const float r2 = x2 + y2 + z2;
    const float rinv = rnorm3df(mean3d.x, mean3d.y, mean3d.z);
    const float sqrtx2y2 = hypotf(mean3d.x, mean3d.y);
    const float sqrtx2y2_inv = rhypotf(mean3d.x, mean3d.y);
    const float xy = mean3d.x * mean3d.y;
    const float xz = mean3d.x * mean3d.z;
    const float yz = mean3d.y * mean3d.z;
    const float r2sqrtx2y2_inv = 1 / (r2) * sqrtx2y2_inv;

    // column major, mat3x2 is 3 columns x 2 rows.
    glm::mat3x2 J = glm::mat3x2(
        -mean3d.y / (x2 + y2), -xz * r2sqrtx2y2_inv, // 1st column
        mean3d.x / (x2 + y2) , -yz * r2sqrtx2y2_inv, // 2nd column
        0.f                  , sqrtx2y2 / r2     // 3rd column
    ) * RAD_TO_DEG;

    glm::mat3 J3x3 = glm::mat3(
        J[0][0], J[0][1], mean3d.x * rinv,
        J[1][0], J[1][1], mean3d.y * rinv,
        J[2][0], J[2][1], mean3d.z * rinv
    );

    glm::mat3 cov3d_spherical = J3x3 * cov3d * glm::transpose(J3x3);
    glm::mat3 v_cov3d_spherical(0.f);
    depth_compensation_from_cov3d_vjp(cov3d_spherical, eps2d, v_depth_compensation, v_cov3d_spherical);
    v_cov3d += glm::transpose(J3x3) * v_cov3d_spherical * J3x3;

    // cov = J * V * Jt; G = df/dcov = v_cov
    // -> df/dV = Jt * G * J
    // -> df/dJ = G * J * Vt + Gt * J * V
    v_cov3d += glm::transpose(J) * v_cov2d * J;

    v_mean3d += v_mean2d * J;
    // glm::vec3(
    //     -mean3d.y/(x2y2)*v_mean2d.x - mean3d.x*mean3d.z/r2sqrtx2y2*v_mean2d.y,
    //     mean3d.x/x2y2 * v_mean2d.x - mean3d.y*mean3d.z/r2sqrtx2y2*v_mean2d.y,
    //     sqrtx2y2/r2*v_mean2d.y
    // );

    const float x2plusy2 = x2 + y2;
    const float x2plusy2squared = x2plusy2 * x2plusy2;
    const float x2plusy2sqrt3andahalf = sqrtx2y2 * x2plusy2;
    const float x4 = x2 * x2;
    const float y4 = y2 * y2;
    const float r4 = r2 * r2;
    const float xyz = mean3d.x * mean3d.y * mean3d.z;
    const float denom1 = 1.f / (x2plusy2sqrt3andahalf * r4);
    const float denom2 = 1.f / r4 * sqrtx2y2_inv;
    const float r3_inv = 1.f / r2 * rinv;

    glm::mat3x2 v_J =
        v_cov2d * J * glm::transpose(cov3d) + glm::transpose(v_cov2d) * J * cov3d;

    glm::mat3 v_J3x3 = v_cov3d_spherical * J3x3 * glm::transpose(cov3d) + glm::transpose(v_cov3d_spherical) * J3x3 * cov3d;
    v_mean3d.x += (
        2.f * mean3d.x * mean3d.y / x2plusy2squared * (v_J[0][0] + v_J3x3[0][0])
        + (y2 - x2) / x2plusy2squared * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        + mean3d.z * (2.f*x4 + x2*y2 - y2*(y2 + z2)) * denom1 * (v_J[0][1] + v_J3x3[0][1])
        + xyz * (3.f*x2 + 3.f*y2 + z2) * denom1 * (v_J[1][1] + v_J3x3[1][1])
        - mean3d.x * (x2 + y2 - z2) * denom2 * (v_J[2][1]+v_J3x3[2][1])
        ) * RAD_TO_DEG;
        v_mean3d.x += (
            (y2 + z2) * r3_inv * v_J3x3[0][2] 
            - xy * r3_inv * v_J3x3[1][2] 
            - xz * r3_inv * v_J3x3[2][2]);

    v_mean3d.y += (
        (y2 - x2) / x2plusy2squared * (v_J[0][0] + v_J3x3[0][0])
        - 2.f * mean3d.x * mean3d.y / x2plusy2squared * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        + xyz * (3.f*x2 + 3.f*y2 + z2) * denom1 * (v_J[0][1] + v_J3x3[0][1])
        - mean3d.z * (x4 + x2*(z2-y2)-2.f*y4) * denom1 * (v_J[1][1] + v_J3x3[1][1])
        - mean3d.y * (x2 + y2 - z2) * denom2 * (v_J[2][1] + v_J3x3[2][1])
        ) * RAD_TO_DEG;
    v_mean3d.y += (
        - xy * r3_inv * v_J3x3[0][2]
        + (x2 + z2) * r3_inv * v_J3x3[1][2] 
        - yz * r3_inv * v_J3x3[2][2]
    );

    v_mean3d.z += (
        0.f * (v_J[0][0] + v_J3x3[0][0])
        + 0.f * (v_J[1][0] + v_J3x3[1][0])
        + 0.f * (v_J[2][0] + v_J3x3[2][0])
        - mean3d.x * (x2 + y2 -z2) * denom2 * (v_J[0][1] + v_J3x3[0][1])
        - mean3d.y * (x2 + y2 -z2) * denom2 * (v_J[1][1] + v_J3x3[1][1])
        - 2.f*mean3d.z * sqrtx2y2 / r4 * (v_J[2][1] + v_J3x3[2][1])
        ) * RAD_TO_DEG;
    v_mean3d.z += (
        - xz * r3_inv * v_J3x3[0][2]
        - yz * r3_inv * v_J3x3[1][2]
        + (x2 + y2) * r3_inv * v_J3x3[2][2]
    );

}

inline __device__ void vel_world_to_cam(
    // [R] is the world-to-camera rotation
    const glm::mat3 R, const glm::vec3 vel, glm::vec3 &vel_c) {
    vel_c = R * vel;
}

inline __device__ void vel_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::vec3 vel,
    // grad outputs
    const glm::vec3 v_vel_c,
    // grad inputs
    glm::mat3 &v_R, glm::vec3 &v_vel) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_vel_c, vel);
    v_vel += glm::transpose(R) * v_vel_c;
}

inline __device__ void pos_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const glm::mat3 R, const glm::vec3 t, const glm::vec3 p, glm::vec3 &p_c) {
    p_c = R * p + t;
}

inline __device__ void pos_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::vec3 t, const glm::vec3 p,
    // grad outputs
    const glm::vec3 v_p_c,
    // grad inputs
    glm::mat3 &v_R, glm::vec3 &v_t, glm::vec3 &v_p) {
    // for D = W * X, G = df/dD
    // df/dW = G * XT, df/dX = WT * G
    v_R += glm::outerProduct(v_p_c, p);
    v_t += v_p_c;
    v_p += glm::transpose(R) * v_p_c;
}

inline __device__ void covar_world_to_cam(
    // [R, t] is the world-to-camera transformation
    const glm::mat3 R, const glm::mat3 covar, glm::mat3 &covar_c) {
    covar_c = R * covar * glm::transpose(R);
}

inline __device__ void covar_world_to_cam_vjp(
    // fwd inputs
    const glm::mat3 R, const glm::mat3 covar,
    // grad outputs
    const glm::mat3 v_covar_c,
    // grad inputs
    glm::mat3 &v_R, glm::mat3 &v_covar) {
    // for D = W * X * WT, G = df/dD
    // df/dX = WT * G * W
    // df/dW
    // = G * (X * WT)T + ((W * X)T * G)T
    // = G * W * XT + (XT * WT * G)T
    // = G * W * XT + GT * W * X
    v_R +=
        v_covar_c * R * glm::transpose(covar) + glm::transpose(v_covar_c) * R * covar;
    v_covar += glm::transpose(R) * v_covar_c * R;
}

inline __device__ float inverse(const glm::mat2 M, glm::mat2 &Minv) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (det <= 0.f) {
        return det;
    }
    float invDet = 1.f / det;
    Minv[0][0] = M[1][1] * invDet;
    Minv[0][1] = -M[0][1] * invDet;
    Minv[1][0] = Minv[0][1];
    Minv[1][1] = M[0][0] * invDet;
    return det;
}

template <class T>
inline __device__ void inverse_vjp(const T Minv, const T v_Minv, T &v_M) {
    // P = M^-1
    // df/dM = -P * df/dP * P
    v_M += -Minv * v_Minv * Minv;
}

inline __device__ float add_blur(const float eps2d, glm::mat2 &covar,
                                 float &compensation) {
    float det_orig = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    covar[0][0] += eps2d;
    covar[1][1] += eps2d;
    float det_blur = covar[0][0] * covar[1][1] - covar[0][1] * covar[1][0];
    compensation = sqrt(max(0.f, det_orig / det_blur));
    return det_blur;
}

inline __device__ void add_blur_vjp(const float eps2d, const glm::mat2 conic_blur,
                                    const float compensation,
                                    const float v_compensation, glm::mat2 &v_covar) {
    // comp = sqrt(det(covar) / det(covar_blur))

    // d [det(M)] / d M = adj(M)
    // d [det(M + aI)] / d M  = adj(M + aI) = adj(M) + a * I
    // d [det(M) / det(M + aI)] / d M
    // = (det(M + aI) * adj(M) - det(M) * adj(M + aI)) / (det(M + aI))^2
    // = adj(M) / det(M + aI) - adj(M + aI) / det(M + aI) * comp^2
    // = (adj(M) - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M + aI) = adj(M) + a * I
    // = (adj(M + aI) - aI - adj(M + aI) * comp^2) / det(M + aI)
    // given that adj(M) / det(M) = inv(M)
    // = (1 - comp^2) * inv(M + aI) - aI / det(M + aI)
    // given det(inv(M)) = 1 / det(M)
    // = (1 - comp^2) * inv(M + aI) - aI * det(inv(M + aI))
    // = (1 - comp^2) * conic_blur - aI * det(conic_blur)

    float det_conic_blur =
        conic_blur[0][0] * conic_blur[1][1] - conic_blur[0][1] * conic_blur[1][0];
    float v_sqr_comp = v_compensation * 0.5 / (compensation + 1e-6);
    float one_minus_sqr_comp = 1 - compensation * compensation;
    v_covar[0][0] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][0] - eps2d * det_conic_blur);
    v_covar[0][1] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[0][1]);
    v_covar[1][0] += v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][0]);
    v_covar[1][1] +=
        v_sqr_comp * (one_minus_sqr_comp * conic_blur[1][1] - eps2d * det_conic_blur);
}

inline __device__ float angle_difference(const float angle1, const float angle2) {
    float diff = fmodf((angle1 - angle2 + 180.f), 360.f) - 180.f;
    return diff < -180.f ? diff + 360.f : diff;
}

#endif // GSPLAT_CUDA_UTILS_H