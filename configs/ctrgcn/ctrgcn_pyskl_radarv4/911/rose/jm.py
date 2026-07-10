# ============================================================
# CTR-GCN on RADAR v4 YOLO26x-pose skeletons, LOSO fold
# Stream options: 'j', 'b', 'jm', 'bm'
# ============================================================

stream = 'jm'  # change to 'b', 'jm', or 'bm' for other streams
pkl = 'radarv4_yolo26xpose_clip60_val_mia_test_rose'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        in_channels=3,      # x, y, score. If no keypoint_score exists, change to 2.
        num_person=1,
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=9,
        in_channels=256
    )
)

dataset_type = 'PoseDataset'
ann_file = f'data/radar_v4/pyskl/{pkl}.pkl'

# COCO-17 left/right keypoint ids
coco_left = [1, 3, 5, 7, 9, 11, 13, 15]
coco_right = [2, 4, 6, 8, 10, 12, 14, 16]

# Current LOSO fold train counts:
# Falling: 582
# Lying-Stationary: 1141
# Sit-Stationary: 595
# Transition-LayBed-to-Sit: 567
# Transition-LayFloor-to-Stand: 572
# Transition-Sit-to-LayBed: 568
# Transition-Sit-to-Stand: 1138
# Transition-Stand-to-Sit: 1142
# Walking: 1881
class_prob = [2.00, 1.00, 2.00, 2.00, 2.00, 2.00, 1.00, 1.00, 1.00]

train_pipeline = [
    # Flip must be before PreNormalize2D because Flip uses raw pixel x coordinate.
    dict(
        type='Flip',
        flip_ratio=0.5,
        direction='horizontal',
        left_kp=coco_left,
        right_kp=coco_right
    ),

    # Normalize 2D keypoints. Make sure img_shape exists correctly in the PKL.
    dict(type='PreNormalize2D', mode='auto'),

    # stream = 'j', 'b', 'jm', or 'bm'
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),

    # Some samples are already 60 frames; some stationary/long transition samples may be longer.
    dict(type='UniformSample', clip_len=60),

    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
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
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split='train',
        class_prob=class_prob
    ),

    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val'
    ),

    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='test'
    )
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)

total_epochs = 20

checkpoint_config = dict(interval=1)

evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1'],
    save_best='macro_f1',
    rule='greater'
)

log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook')]
)

# runtime settings
log_level = 'INFO'

work_dir = f'./work_dirs/ctrgcn/{pkl}/{stream}'