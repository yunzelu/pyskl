stream = 'b'
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        in_channels=3,
        num_person=1,
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=9, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'data/radar_v4/pyskl/radar_v4_yolo26xpose_clip60.pkl'
coco_left = [1, 3, 5, 7, 9, 11, 13, 15]
coco_right = [2, 4, 6, 8, 10, 12, 14, 16]
class_prob = [2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
train_pipeline = [
    dict(
        type='Flip',
        flip_ratio=0.5,
        direction='horizontal',
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file='data/radar_v4/pyskl/radar_v4_yolo26xpose_clip60.pkl',
        pipeline=[
            dict(
                type='Flip',
                flip_ratio=0.5,
                direction='horizontal',
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(type='PreNormalize2D', mode='auto'),
            dict(type='GenSkeFeat', dataset='coco', feats=['b']),
            dict(type='UniformSample', clip_len=60),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='train',
        class_prob=[2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]),
    val=dict(
        type='PoseDataset',
        ann_file='data/radar_v4/pyskl/radar_v4_yolo26xpose_clip60.pkl',
        pipeline=[
            dict(type='PreNormalize2D', mode='auto'),
            dict(type='GenSkeFeat', dataset='coco', feats=['b']),
            dict(type='UniformSample', clip_len=60, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/radar_v4/pyskl/radar_v4_yolo26xpose_clip60.pkl',
        pipeline=[
            dict(type='PreNormalize2D', mode='auto'),
            dict(type='GenSkeFeat', dataset='coco', feats=['b']),
            dict(type='UniformSample', clip_len=60, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'))
optimizer = dict(
    type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 20
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/radar_v4_yolo26xpose_clip60/b'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 4)
