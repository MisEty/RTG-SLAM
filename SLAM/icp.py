import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SLAM.utils import *

def point2plane_loss(p_t0, p_t1, n_t0, reduce="mean"):
    loss = ((p_t1 - p_t0) * n_t0).sum(dim=-1)
    if reduce == "mean":
        loss = (loss * loss).mean()
    else:
        loss = (loss * loss).sum()
    return loss


class ICP(nn.Module):
    def __init__(
        self,
        max_iter=3,
        damping=1e-6,
        distance_threshold=0.2,
        normal_threshold=20,
        verbose=False,
    ):
        super(ICP, self).__init__()

        self.max_iterations = max_iter
        self.distance_threshold = distance_threshold
        self.normal_threshold = np.cos(np.deg2rad(normal_threshold))
        self.damping = damping
        self.verbose = verbose

    def icp(self, pose10,vertex_t0,vertex_t1,normal_t0,normal_t1, K):
        mask0 = (vertex_t0[..., -1] > 0.0)
        
        for idx in range(self.max_iterations):
            # compute residuals
            residuals, J_F_p, valid_mask = self.compute_residuals_jacobian(
                vertex_t0, vertex_t1, normal_t0, normal_t1, mask0, pose10, K,
                self.distance_threshold, self.normal_threshold
            )

            JtWJ = self.compute_jtj(J_F_p)  # [B, 6, 6]
            JtR = self.compute_jtr(J_F_p, residuals)
            pose10 = self.GN_solver(JtWJ, JtR, pose10, damping=self.damping)
        H,W = vertex_t0.shape[:2]
        valid_ratio = valid_mask.sum() / H / W
        return pose10, valid_ratio
    

    @staticmethod
    def compute_residuals_jacobian(vertex0, vertex1, normal0, normal1, mask0, pose10, K, 
                                   distance_threshold, normal_threshold):
        """
        :param vertex0: vertex map 0
        :param vertex1: vertex map 1
        :param normal0: normal map 0
        :param normal1: normal map 1
        :param mask0: valid mask of template depth image
        :param pose10: current estimate of pose10
        :param K: intrinsics
        :return: residuals and Jacobians
        """
        R = pose10[:3, :3]
        t = pose10[:3, 3]
        H, W, C = vertex0.shape

        rot_vertex0_to1 = (R @ vertex0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)
        vertex0_to1 = rot_vertex0_to1 + t[None, None, :]
        normal0_to1 = (R @ normal0.view(-1, 3).permute(1, 0)).permute(1, 0).view(H, W, 3)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_, y_, z_ = vertex0_to1[..., 0], vertex0_to1[..., 1], vertex0_to1[..., 2]  # [h, w]
        u_ = (x_ / z_) * fx + cx  # [h, w]
        v_ = (y_ / z_) * fy + cy  # [h, w]

        inviews = (u_ > 0) & (u_ < W-1) & (v_ > 0) & (v_ < H-1)
        # projective data association
        r_vertex1 = warp_features(vertex1, u_, v_)  # [h, w, 3]
        r_normal1 = warp_features(normal1, u_, v_)  # [h, w, 3]
        mask1 = r_vertex1[..., -1] > 0.
        diff = vertex0_to1 - r_vertex1  # [h, w, 3]

        normal_diff_mask = torch.sum(normal0_to1 * r_normal1, dim=-1) > normal_threshold
        
        # point-to-plane residuals
        res = (r_normal1 * diff).sum(dim=-1)  # [h, w]
        # point-to-plane jacobians
        J_trs = r_normal1.view(-1, 3)  # [hw, 3]
        J_rot = -torch.bmm(J_trs.unsqueeze(dim=1), batch_skew(vertex0_to1.view(-1, 3))).squeeze()   # [hw, 3]

        # compose jacobians
        J_F_p = torch.cat((J_rot, J_trs), dim=-1).view(H, W, 6)  # follow the order of [rot, trs]  [hw, 1, 6]

        # occlusion
        occ = ~inviews | (diff.norm(p=2, dim=-1) > distance_threshold) 
        invalid_mask = occ | ~mask0 | ~mask1 | ~normal_diff_mask
        J_F_p[invalid_mask] = 0.
        res[invalid_mask] = 0.

        res = res.view(-1, 1)  # [hw, 1]
        J_F_p = J_F_p.view(-1, 1, 6)  # [hw, 1, 6]

        return res, J_F_p, ~invalid_mask

    @staticmethod
    def compute_jtj(jac):
        # J in the dimension of (HW, C, 6)
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtj = torch.bmm(jacT, jac).sum(0)  # [6, 6]
        return jtj  # [6, 6]

    @staticmethod
    def compute_jtr(jac, res):
        # J in the dimension of (HW, C, 6)
        # res in the dimension of [HW, C]
        jacT = jac.transpose(-1, -2)  # [HW, 6, C]
        jtr = torch.bmm(jacT, res.unsqueeze(-1)).sum(0)  # [6, 1]
        return jtr  # [6, 1]

    @staticmethod
    def GN_solver(JtJ, JtR, pose0, damping=1e-6):
        # Add a small diagonal damping. Without it, the training becomes quite unstable
        # Do not see a clear difference by removing the damping in inference though
        Hessian = lev_mar_H(JtJ, damping)
        # Hessian = JtJ
        updated_pose = forward_update_pose(Hessian, JtR, pose0)

        return updated_pose


def warp_features(Feat, u, v, mode="nearest"):
    """
    Warp the feature map (F) w.r.t. the grid (u, v). This is the non-batch version
    """
    assert len(Feat.shape) == 3
    H, W, C = Feat.shape
    u_norm = u / ((W - 1) / 2) - 1  # [h, w]
    v_norm = v / ((H - 1) / 2) - 1  # [h, w]
    uv_grid = torch.cat((u_norm.view(1, H, W, 1), v_norm.view(1, H, W, 1)), dim=-1)
    Feat_warped = F.grid_sample(
        Feat.unsqueeze(0).permute(0, 3, 1, 2),
        uv_grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    ).squeeze()
    return Feat_warped.permute(1, 2, 0)


def compute_vertex(depth, K):
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    device = depth.device

    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H)
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().to(device)  # [h, w]
    j = j.t().to(device)  # [h, w]

    vertex = (
        torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1).to(device)
        * depth[..., None]
    )  # [h, w, 3]
    return vertex


def compute_normal(vertex_map):
    """Calculate the normal map from a depth map
    :param the input depth image
    -----------
    :return the normal map
    """
    H, W, C = vertex_map.shape
    img_dx, img_dy = feature_gradient(vertex_map, normalize_gradient=False)  # [h, w, 3]

    normal = torch.cross(img_dx.view(-1, 3), img_dy.view(-1, 3))
    normal = normal.view(H, W, 3)  # [h, w, 3]

    mag = torch.norm(normal, p=2, dim=-1, keepdim=True)
    normal = normal / (mag + 1e-8)

    # filter out invalid pixels
    depth = vertex_map[:, :, -1]
    # 0.5 and 5.
    invalid_mask = (depth <= depth.min()) | (depth >= depth.max())
    zero_normal = torch.zeros_like(normal)
    normal = torch.where(invalid_mask[..., None], zero_normal, normal)

    return normal


def feature_gradient(img, normalize_gradient=True):
    """Calculate the gradient on the feature space using Sobel operator
    :param the input image
    -----------
    :return the gradient of the image in x, y direction
    """
    H, W, C = img.shape
    # to filter the image equally in each channel
    wx = (
        torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .view(1, 1, 3, 3)
        .type_as(img)
    )
    wy = (
        torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .view(1, 1, 3, 3)
        .type_as(img)
    )

    img_permuted = img.permute(2, 0, 1).view(-1, 1, H, W)  # [c, 1, h, w]
    img_pad = F.pad(img_permuted, (1, 1, 1, 1), mode="replicate")
    img_dx = (
        F.conv2d(img_pad, wx, stride=1, padding=0).squeeze().permute(1, 2, 0)
    )  # [h, w, c]
    img_dy = (
        F.conv2d(img_pad, wy, stride=1, padding=0).squeeze().permute(1, 2, 0)
    )  # [h, w, c]

    if normalize_gradient:
        mag = torch.sqrt((img_dx**2) + (img_dy**2) + 1e-8)
        img_dx = img_dx / mag
        img_dy = img_dy / mag

    return img_dx, img_dy  # [h, w, c]


def batch_skew(w):
    """Generate a batch of skew-symmetric matrices.

        function tested in 'test_geometry.py'

    :input
    :param skew symmetric matrix entry Bx3
    ---------
    :return
    :param the skew-symmetric matrix Bx3x3
    """
    B, D = w.shape
    assert D == 3
    o = torch.zeros(B).type_as(w)
    w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
    return torch.stack((o, -w2, w1, w2, o, -w0, -w1, w0, o), 1).view(B, 3, 3)


def lev_mar_H(JtWJ, damping):
    # Add a small diagonal damping. Without it, the training becomes quite unstable
    # Do not see a clear difference by removing the damping in inference though
    diag_mask = torch.eye(6).to(JtWJ)
    diagJtJ = diag_mask * JtWJ
    traceJtJ = torch.sum(diagJtJ)
    epsilon = (traceJtJ * damping) * diag_mask
    Hessian = JtWJ + epsilon
    return Hessian


def forward_update_pose(H, Rhs, pose):
    """
    :param H:
    :param Rhs:
    :param pose:
    :return:
    """
    xi = least_square_solve(H, Rhs).squeeze()
    pose = exp_se3(xi) @ pose
    return pose


def exp_se3(xi):
    """
    :param x: Cartesian vector of Lie Algebra se(3)
    :return: exponential map of x
    """
    w = xi[:3].squeeze()  # rotation
    v = xi[3:6].squeeze()  # translation
    w_hat = torch.tensor(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
    ).to(xi)
    w_hat_second = torch.mm(w_hat, w_hat).to(xi)

    theta = torch.norm(w)
    theta_2 = theta**2
    theta_3 = theta**3
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    eye_3 = torch.eye(3).to(xi)

    eps = 1e-8

    if theta <= eps:
        e_w = eye_3
        j = eye_3
    else:
        e_w = (
            eye_3
            + w_hat * sin_theta / theta
            + w_hat_second * (1.0 - cos_theta) / theta_2
        )
        k1 = (1 - cos_theta) / theta_2
        k2 = (theta - sin_theta) / theta_3
        j = eye_3 + k1 * w_hat + k2 * w_hat_second

    T = torch.eye(4).to(xi)
    T[:3, :3] = e_w
    T[:3, 3] = torch.mv(j, v)
    # T[:3, 3] = v

    return T


def invH(H):
    """Generate (H+damp)^{-1}, with predicted damping values
    :param approximate Hessian matrix JtWJ
    -----------
    :return the inverse of Hessian
    """
    # GPU is much slower for matrix inverse when the size is small (compare to CPU)
    # works (50x faster) than inversing the dense matrix in GPU
    if H.is_cuda:
        invH = torch.inverse(H.cpu()).cuda()
    else:
        invH = torch.inverse(H)
    return invH


def least_square_solve(H, Rhs):
    """
    Solve for JTJ @ xi = -JTR
    """
    inv_H = invH(H)  # [B, 6, 6] square matrix
    xi = -inv_H @ Rhs
    return xi


class ImagePyramids(nn.Module):
    """ Construct the pyramids in the image / depth space
    """
    def __init__(self, scales, pool='avg'):
        super(ImagePyramids, self).__init__()
        if pool == 'avg':
            self.multiscales = [nn.AvgPool2d(1<<i, 1<<i) for i in scales]
        elif pool == 'max':
            self.multiscales = [nn.MaxPool2d(1<<i, 1<<i) for i in scales]
        else:
            raise NotImplementedError()

    def forward(self, x):
        if x.dtype == torch.bool:
            x = x.to(torch.float32)
            x_out = [f(x).to(torch.bool) for f in self.multiscales]
        else:
            x_out = [f(x) for f in self.multiscales]
        return x_out

class IcpTracker:
    def __init__(self, args):
        self.icp_trackers = []
        self.icp_downscales = args.icp_downscales
        self.icp_warmup_frames = args.icp_warmup_frames
        self.icp_use_model_depth = args.icp_use_model_depth
        for iters in args.icp_downscale_iters:
            self.icp_trackers.append(
                ICP(
                    iters,
                    distance_threshold=args.icp_distance_threshold,
                    normal_threshold=args.icp_normal_threshold,
                    damping=args.icp_damping,
                    verbose=args.verbose,
                )
            )
        
        self.depth_pyramid_builder = ImagePyramids(list(range(len(self.icp_downscales)-1,-1, -1)), "max")
        self.icp_sample_distance_threshold = args.icp_sample_distance_threshold
        self.icp_sample_normal_threshold = args.icp_sample_normal_threshold
        self.icp_fail_threshold = args.icp_fail_threshold

        self.normal_pyramid_t0 = None
        self.vertex_pyramid_t0 = None
        self.verbose = args.verbose

        self.K = None
        
    def update_curr_status(self, depth_t1, K):
        if self.K is None:
            self.K = K
        self.depth_t1 = depth_t1
        self.vertex_pyramid_t1 = build_vertex_pyramid(depth_t1, self.depth_pyramid_builder, self.K)
        self.normal_pyramid_t1 = build_normal_pyramid(self.vertex_pyramid_t1)

    def move_last_status(self):
        self.vertex_pyramid_t0 = self.vertex_pyramid_t1
        self.normal_pyramid_t0 = self.normal_pyramid_t1
        self.last_model_depth = self.depth_t1

    def update_last_status(
        self, frame, render_depth, frame_depth, render_normal, frame_normal
    ):

        intrinsic = frame.get_intrinsic
        normal_mask = (
            1 - F.cosine_similarity(render_normal, frame_normal, dim=-1)
        ) > self.icp_sample_normal_threshold
        depth_filling_mask = (
            (
                torch.abs(render_depth - frame_depth)
                > self.icp_sample_distance_threshold
            )[..., 0]
            | (render_depth == 0)[..., 0]
            | (normal_mask)
        ) & (frame_depth > 0)[..., 0]

        render_depth[depth_filling_mask] = frame_depth[depth_filling_mask]
        self.last_model_depth = render_depth

    def predict_pose(self, frame):
        K = frame["K"]
        frame_id = frame["frame_id"]
        if self.vertex_pyramid_t0 is None:
            pose_t1_t0 = np.eye(4)
            self.K = K
        else:
            if self.icp_use_model_depth and frame_id >= self.icp_warmup_frames:
                self.vertex_pyramid_t0 = build_vertex_pyramid(self.last_model_depth, self.depth_pyramid_builder, self.K)
                self.normal_pyramid_t0 = build_normal_pyramid(self.vertex_pyramid_t0)
                
            pose_t1_t0 = devF(torch.from_numpy(np.eye(4)))
            levels = len(self.icp_downscales)
            for level in range(levels):
                downscale = self.icp_downscales[level]
                K_downscale = K * downscale
                K_downscale[2,2] = 1.0
                vertex_t0 = self.vertex_pyramid_t0[level]
                vertex_t1 = self.vertex_pyramid_t1[level]
                normal_t0 = self.normal_pyramid_t0[level]
                normal_t1 = self.normal_pyramid_t1[level]
                pose_t1_t0, valid_ratio = self.icp_trackers[level].icp(
                    pose_t1_t0, vertex_t1, vertex_t0, normal_t1, normal_t0,
                    K_downscale 
                )
            pose_t1_t0 = pose_t1_t0.cpu().numpy()
        pose_t1_t0_pytorch = torch.tensor(pose_t1_t0).cuda().float()
        p2ploss = point2plane_loss(self.vertex_pyramid_t0[-1], 
                                   self.vertex_pyramid_t1[-1] @ pose_t1_t0_pytorch[:3,:3].T + pose_t1_t0_pytorch[:3, 3], 
                                   self.normal_pyramid_t0[-1], 
                                   )
        print(p2ploss, valid_ratio)
        tracking_success = True
        if p2ploss > self.icp_fail_threshold:
            tracking_success = False
        return pose_t1_t0, tracking_success
