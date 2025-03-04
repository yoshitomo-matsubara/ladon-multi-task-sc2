# A Multi-task Supervised Compression Model for Split Computing

This is the official repository for our WACV 2025 paper, ***"A Multi-task Supervised Compression Model for Split Computing"***.

Split computing (/= split learning) is a promising approach to deep learning models for resource-constrained edge 
computing systems, where weak sensor (mobile) devices are wirelessly connected to stronger edge servers through channels 
with limited communication capacity. State-of-the-art work on split computing presents methods for single tasks such as 
image classification, object detection, or semantic segmentation. The application of existing methods to multi-task 
problems degrades model accuracy and/or significantly increase runtime latency. In this study, we propose Ladon, 
the first multi-task-head supervised compression model for multi-task split computing. Experimental results show that 
the multi-task supervised compression model either outperformed or rivaled strong lightweight baseline models 
in terms of predictive performance for ILSVRC 2012, COCO 2017, and PASCAL VOC 2012 datasets while learning compressed 
representations at its early layers. Furthermore, our models reduced end-to-end latency (by up to 95.4%) and 
energy consumption of mobile devices (by up to 88.2%) in multi-task split computing scenarios.

## Ladon: the first multi-task supervised compression model for split computing
![Entropic Student vs. Ladon](imgs/ladon_model-comparison.png)

In multi-task split computing scenarios, it is critical to optimize learnable parameters of task-specific modules on
a unified image processing pipeline instead of task-specific pipelines so that we can
1. reduce encoding cost and energy consumption on a weak local device,
2. save offloading data size by transferring only one compressed representation from the weak local device to a(n) cloud/edge server, and
3. reduce end-to-end latency and local device energy consumption,

while outperforming or rivaling predictive performance of lightweight models.

Note that in split computing, models are trained offline (e.g., on a single machine) and the distributed inference 
like the above figure occurs only at runtime.

Refer to our previous work for [supervised compression](https://github.com/yoshitomo-matsubara/supervised-compression) 
and [SC2 benchmark (supervised compression for split computing)](https://github.com/yoshitomo-matsubara/sc2-benchmark)

## Citation
[[Paper](https://openaccess.thecvf.com/content/WACV2025/html/Matsubara_A_Multi-Task_Supervised_Compression_Model_for_Split_Computing_WACV_2025_paper.html)][[Preprint](https://arxiv.org/abs/2501.01420)]
```bibtex
@inproceedings{matsubara2025multi,
  title={{A Multi-Task Supervised Compression Model for Split Computing}},
  author={Matsubara, Yoshitomo and Mendula, Matteo and Levorato, Marco},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={4913-4922},
  month={February},
  year={2025}
}
```

## Checkpoints
You can download our checkpoints including trained model weights [here](https://github.com/yoshitomo-matsubara/ladon-multi-task-sc2/releases/tag/wacv2025).  
Unzip the downloaded zip files under `./`, then there will be `./resource/ckpt/`.

## Legacy code
This study was done prior to the release of sc2bench v0.1.0, which introduces a lot of breaking changes and updates the structure of configuration files.
To reuse the code and configurations at that time, check the files in [./legacy/](./legacy/).  
If you want to use more recent versions of the required packages, refer to [Updated code](#updated-code)).

## Updated code

### Requirements
- Python >= 3.9
- sc2bench >= 0.1.0
- numpy

```shell
pipenv install
```

### Datasets
- Image classification: [ILSVRC 2012 (ImageNet)](https://www.image-net.org/challenges/LSVRC/2012/)
- Object detection: [COCO 2017](https://cocodataset.org/#detection-2017)
- Semantic segmentation: [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

Follow the instructions in [my sc2-benchmark repository](https://github.com/yoshitomo-matsubara/sc2-benchmark/tree/main/script#datasets)


### Evaluation with checkpoints

Download our checkpoints including trained model weights [here](https://github.com/yoshitomo-matsubara/ladon-multi-task-sc2/releases/tag/wacv2025).  
Unzip the downloaded zip files under `./`, then there will be `./resource/ckpt/`.

#### 1. ImageNet (ILSVRC 2012): Image Classification

##### Ladon (ResNet-50) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/image_classification.py -test_only -student_only \
    --config legacy/configs/ilsvrc2012/ladon/splittable_resnet50-fp-beta${beta}_from_resnet50.yaml \
    --run_log logs/ilsvrc2012/ladon/splittable_resnet50-fp-beta${beta}_from_resnet50.txt
done
```

##### Ladon (ResNeSt-269e) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/image_classification.py -test_only -student_only \
    --config configs/ilsvrc2012/ladon/splittable_resnest269e-fp-beta${beta}_from_resnest269e.yaml \
    --run_log logs/ilsvrc2012/ladon/splittable_resnest269e-fp-beta${beta}_from_resnest269e.txt
done
```

#### 2. COCO 2017: Object Detection
##### Ladon (Faster R-CNN with ResNet-50 and FPN) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/object_detection.py -test_only \
    --config configs/coco2017/ladon/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn.yaml \
    --run_log logs/coco2017/ladon/faster_rcnn_splittable_resnet50-fp-beta${beta}_fpn.txt
done
```

##### Ladon (Faster R-CNN with ResNeSt-269e and FPN) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/object_detection.py -test_only \
    --config configs/coco2017/ladon/faster_rcnn_splittable_resnest269e-fp-beta${beta}_fpn.yaml \
    --run_log logs/coco2017/ladon/faster_rcnn_splittable_resnest269e-fp-beta${beta}_fpn.txt
done
```

#### 3. PASCAL VOC 2012: Semantic Segmentation
##### Ladon (DeepLabv3 with ResNet-50) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/semantic_segmentation.py -test_only \
    --config configs/pascal_voc2012/ladon/deeplabv3_splittable_resnet50-fp-beta${beta}_two_stages.yaml \
    --run_log logs/pascal_voc2012/ladon/deeplabv3_splittable_resnet50-fp-beta${beta}_two_stages.txt
done
```

##### Ladon (DeepLabv3 with ResNeSt-269e) 
```shell
for beta in 0.32 1.28 5.12 10.24 20.48; do
  python scripts/semantic_segmentation.py -test_only \
    --config configs/pascal_voc2012/ladon/deeplabv3_splittable_resnest269e-fp-beta${beta}_two_stages.yaml \
    --run_log logs/pascal_voc2012/ladon/deeplabv3_splittable_resnest269e-fp-beta${beta}_two_stages.txt
done
```

#### Training

Use the same command as above but without `-test_only`.
