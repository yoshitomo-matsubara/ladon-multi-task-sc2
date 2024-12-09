from collections import OrderedDict

import torch
from sc2bench.models.backbone import FeatureExtractionBackbone
from sc2bench.models.detection.rcnn import BaseRCNN, _process_torchvision_pretrained_weights, \
    create_faster_rcnn_fpn
from sc2bench.models.registry import load_classification_model
from sc2bench.models.segmentation.deeplabv3 import create_deeplabv3
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import load_ckpt

logger = def_logger.getChild(__name__)
MODEL_URL_DICT = {
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
    'deeplabv3_mobilenet_v3_large': 'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth'
}


class RuntimeEntropicEncoder(nn.Module):
    def __init__(self, bottleneck_layer):
        super().__init__()
        self.encoder = bottleneck_layer.encoder
        self.entropy_bottleneck = bottleneck_layer.entropy_bottleneck

    def forward(self, x):
        latent = self.encoder(x)
        latent_strings = self.entropy_bottleneck.compress(latent)
        return {'strings': [latent_strings], 'shape': latent.size()[-2:]}


class RuntimeEntropicSharedBody(nn.Module):
    def __init__(self, splittable_resnet_model):
        super().__init__()
        self.entropy_bottleneck = splittable_resnet_model.bottleneck_layer.entropy_bottleneck
        self.decoder = splittable_resnet_model.bottleneck_layer.decoder
        self.layer2 = splittable_resnet_model.layer2
        self.layer3 = splittable_resnet_model.layer3
        self.layer4 = splittable_resnet_model.layer4

    def forward(self, z):
        z = self.entropy_bottleneck.decompress(z['strings'][0], z['shape'])
        feature_dict = OrderedDict()
        z = self.decoder(z)
        feature_dict['0'] = z
        z = self.layer2(z)
        feature_dict['1'] = z
        z = self.layer3(z)
        feature_dict['2'] = z
        z = self.layer4(z)
        feature_dict['3'] = z
        return feature_dict


class LadonResNetClassificationHead(nn.Module):
    def __init__(self, splittable_resnet_model, pool_path='avgpool', fc_path='fc'):
        super().__init__()
        self.avgpool = getattr(splittable_resnet_model, pool_path)
        self.fc = getattr(splittable_resnet_model, fc_path)

    def forward(self, z):
        z = self.avgpool(z)
        z = torch.flatten(z, 1)
        z = self.fc(z)
        return z


class LadonRCNNDetectionHead(nn.Module):
    def __init__(self, splittable_resnet_detection_model):
        super().__init__()
        self.transform = splittable_resnet_detection_model.transform
        self.fpn = splittable_resnet_detection_model.backbone.fpn
        self.rpn = splittable_resnet_detection_model.rpn
        self.roi_heads = splittable_resnet_detection_model.roi_heads

    def forward(self, images, z, original_image_sizes):
        if isinstance(z, torch.Tensor):
            z = OrderedDict([('0', z)])
        z = self.fpn(z)
        proposals, _ = self.rpn(images, z, targets=None)
        detections, _ = self.roi_heads(z, proposals, images.image_sizes, targets=None)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections


class LadonDeepLabSegmentationHead(nn.Module):
    def __init__(self, splittable_resnet_segmentation_model):
        super().__init__()
        self.classifier = splittable_resnet_segmentation_model.classifier

    def forward(self, z, input_shape):
        z = self.classifier(z)
        z = functional.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)
        return z


class ThreeHeadedLadonResNet(nn.Module):
    def __init__(self, encoder, shared_body, classification_head, detection_head, segmentation_head):
        super().__init__()
        self.encoder = encoder
        self.shared_body = shared_body
        self.classification_head = classification_head
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

    def forward(self, images):
        # `images` is image tensor for object detection
        original_image_size_list = list()
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f'expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}',
            )
            original_image_size_list.append((val[0], val[1]))

        images, _ = self.detection_head.transform(images, None)
        input_shape = images.tensors.shape[-2:]
        z = self.encoder(images.tensors)
        z = self.shared_body(z)
        pred_dict = dict()
        last_layer_output = list(z.values())[-1]
        pred_dict['classification'] = self.classification_head(last_layer_output)
        pred_dict['detection'] = self.detection_head(images, z, original_image_size_list)
        pred_dict['segmentation'] = self.segmentation_head(last_layer_output, input_shape)
        return pred_dict


def faster_rcnn_model_with_backbone(backbone, pretrained=True, pretrained_backbone_name=None, progress=True,
                                    backbone_fpn_kwargs=None, analysis_config=None, num_classes=91,
                                    start_ckpt_file_path=None, **kwargs):
    if backbone_fpn_kwargs is None:
        backbone_fpn_kwargs = dict()

    if analysis_config is None:
        analysis_config = dict()

    faster_rcnn_model = create_faster_rcnn_fpn(backbone, num_classes=num_classes, **backbone_fpn_kwargs, **kwargs)
    model = BaseRCNN(faster_rcnn_model, analysis_config=analysis_config)
    if pretrained and pretrained_backbone_name in ('resnet50', 'mobilenet_v3_large_320', 'mobilenet_v3_large'):
        _process_torchvision_pretrained_weights(model, pretrained_backbone_name, progress)

    if start_ckpt_file_path is not None:
        load_ckpt(start_ckpt_file_path, model=model, strict=False)
    return model


def deeplabv3_model_with_backbone(backbone, pretrained=True, pretrained_backbone_name=None, progress=True,
                                  num_input_channels=2048, uses_aux=False, num_aux_channels=1024,
                                  return_layer_dict=None, analysis_config=None, analyzable_layer_key=None,
                                  num_classes=21, start_ckpt_file_path=None, **kwargs):
    if analysis_config is None:
        analysis_config = dict()

    if return_layer_dict is None:
        return_layer_dict = {'layer4': 'out'}
        if uses_aux:
            return_layer_dict['layer3'] = 'aux'

    backbone_model = \
        FeatureExtractionBackbone(backbone, return_layer_dict, analysis_config.get('analyzer_configs', list()),
                                  analysis_config.get('analyzes_after_compress', False),
                                  analyzable_layer_key=analyzable_layer_key)
    model = create_deeplabv3(backbone_model, num_input_channels=num_input_channels,
                             uses_aux=uses_aux, num_aux_channels=num_aux_channels, num_classes=num_classes)
    if pretrained and pretrained_backbone_name in ('resnet50', 'resnet101'):
        state_dict = \
            load_state_dict_from_url(MODEL_URL_DICT['deeplabv3_{}_coco'.format(pretrained_backbone_name)],
                                     progress=progress)
        model.load_state_dict(state_dict, strict=False)

    if start_ckpt_file_path is not None:
        load_ckpt(start_ckpt_file_path, model=model, strict=False)
    return model


def _ladon_splittable_resnet(splittable_resnet_model, splittable_resnet_detection_model,
                             splittable_resnet_segmentation_model, pool_path='avgpool', fc_path='fc'):
    encoder = RuntimeEntropicEncoder(splittable_resnet_model.bottleneck_layer)
    shared_body = RuntimeEntropicSharedBody(splittable_resnet_model)
    classification_head = LadonResNetClassificationHead(splittable_resnet_model, pool_path=pool_path, fc_path=fc_path)
    detection_head = LadonRCNNDetectionHead(splittable_resnet_detection_model)
    segmentation_head = LadonDeepLabSegmentationHead(splittable_resnet_segmentation_model)
    ladon_resnet_model = \
        ThreeHeadedLadonResNet(encoder, shared_body, classification_head, detection_head, segmentation_head)
    return ladon_resnet_model


def ladon_splittable_resnet(model_config, device):
    splittable_resnet_model = load_classification_model(model_config['classification'], device, False)
    splittable_resnet_model.update()
    splittable_resnet_detection_model = faster_rcnn_model_with_backbone(splittable_resnet_model,
                                                                        **model_config['detection']['kwargs'])
    splittable_resnet_segmentation_model = deeplabv3_model_with_backbone(splittable_resnet_model,
                                                                         **model_config['segmentation']['kwargs'])
    model = _ladon_splittable_resnet(splittable_resnet_model, splittable_resnet_detection_model,
                                     splittable_resnet_segmentation_model)
    return model
