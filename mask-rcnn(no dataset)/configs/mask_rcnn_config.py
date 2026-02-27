# configs/mask_rcnn_config.py

# ✅ 必须加上 'mmdet::' 前缀，告诉系统去已安装的 mmdet 包里找基础配置
_base_ = [
    'mmdet::_base_/models/mask-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]


# 1. 模型配置：修改类别数为 1 (树冠)
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

# 2. 数据集配置
# 因为 train.json 在项目根目录，所以 data_root 设为 './'
data_root = './' 

metainfo = {
    'classes': ('tree', ),
    'palette': [
        (220, 20, 60), # 红色，用于可视化
    ]
}

# 训练集
train_dataloader = dict(
    batch_size=2,      # 如果显存够大 (>=8G)，可以改为 4
    num_workers=2,     # 数据加载线程数
    dataset=dict(
        data_root=data_root,
        ann_file='train.json',       # 根目录下的训练标注
        data_prefix=dict(img='images_jpg/'), # 根目录下的图片文件夹
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True) # 过滤掉没有标注的图片
    )
)

# 验证集
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        ann_file='val.json',         # 根目录下的验证标注
        data_prefix=dict(img='images_jpg/'),
        metainfo=metainfo,
    )
)

# 测试集 (通常复用验证集配置)
test_dataloader = val_dataloader

# 3. 训练策略
# 为了快速跑通，我们只训练 6 个 epoch
train_cfg = dict(max_epochs=6, val_interval=2) 

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# 学习率调整
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=6, by_epoch=True, milestones=[4, 5], gamma=0.1)
]

# 评估指标
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox', 'segm'],
    format_only=False)
test_evaluator = val_evaluator