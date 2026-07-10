# ============================================================
# ST-GCN++ on RADAR v4 YOLO26x-pose skeletons, LOSO fold
# Stream options: 'j', 'b', 'jm', 'bm'
# ============================================================

stream = 'b'
pkl = 'radarv4_yolo26xpose_clip60_val_mia_test_jiadi'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        in_channels=3,
        num_person=1,
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=9,
        in_channels=256
    )
)

dataset_type = 'PoseDataset'
ann_file = f'data/radar_v4/pyskl/911/{pkl}.pkl'

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
    dict(
        type='Flip',
        flip_ratio=0.5,
        direction='horizontal',
        left_kp=coco_left,
        right_kp=coco_right
    ),
    dict(type='PreNormalize2D', mode='auto'),
    dict(type='GenSkeFeat', dataset='coco', feats=[stream]),
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

optimizer = dict(
    type='SGD',
    lr=0.05,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 20
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy', 'macro_f1'],
    save_best='macro_f1',
    rule='greater'
)
test_evaluation = dict(
    metrics=[
        'top_k_accuracy', 'mean_class_accuracy', 'macro_f1',
        'per_class_f1', 'confusion_matrix'
    ],
    topk=(1, 5)
)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook')]
)
log_level = 'INFO'
work_dir = f'./work_dirs/stgcn++/{pkl}/{stream}'
