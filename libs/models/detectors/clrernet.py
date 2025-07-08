from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
import torch.nn.functional as F
import torch
@DETECTORS.register_module()
class CLRerNet(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        """CLRerNet detector."""
        super(CLRerNet, self).__init__(
            backbone, neck, bbox_head, train_cfg, test_cfg, pretrained, init_cfg
        )

        self.num_timesteps = 100
    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        # t = torch.randint(0, 0, (len(img_metas),), device=self.bbox_head.device).long()+9
        t = torch.full((len(img_metas),), 9, device=self.bbox_head.device, dtype = torch.long)
        x_t = self.backbone(img.detach(), t)
        x_t = self.neck(x_t)
        
        # with torch.no_grad():
        #     x_100 = self.backbone(img.detach(), torch.full((len(img_metas),), 9, device=self.bbox_head.device, dtype = torch.long))
        #     x_100 = self.neck(x_100)
        x_100 = None
        losses = self.bbox_head.forward_train(x_100, x_t, t*11, img_metas)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """
        Single-image test without augmentation.
        Args:
            img (torch.Tensor): Input image tensor of shape (1, 3, height, width).
            img_metas (List[dict]): Meta dict containing image information.
        Returns:
            result_dict (List[dict]): Single-image result containing prediction outputs and
             img_metas as 'result' and 'metas' respectively.
        """
        assert (
            img.shape[0] == 1 and len(img_metas) == 1
        ), "Only single-image test is supported."
        img_metas[0]["batch_input_shape"] = tuple(img.size()[-2:])

        feat_list =  []
        tau = 1
        for i in range(0, 10, tau):
            t = torch.full((len(img_metas),), i, device=self.bbox_head.device, dtype = torch.long)
            x_t = self.backbone(img.detach(), t)
            x_t = self.neck(x_t)
            feat_list.append(x_t)
        output = self.bbox_head.simple_test(feat_list)
        result_dict = {
            "result": output,
            "meta": img_metas[0],
        }
        return [result_dict]  # assuming batch size is 1