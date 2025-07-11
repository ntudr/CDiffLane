_base_ = [
    "../base_CDiffLane.py",
    "dataset_culane_CDiffLane.py",
    "../../_base_/default_runtime.py",
]

# custom imports
custom_imports = dict(
    imports=[
        "libs.models",
        "libs.datasets",
        "libs.core.bbox",
        "libs.core.anchor",
        "libs.core.hook",
    ],
    allow_failed_imports=False,
)

cfg_name = "CDiffLane_culane_dla34.py"

model = dict(test_cfg=dict(conf_threshold=0.41))

total_epochs = 15
evaluation = dict(interval=16)
checkpoint_config = dict(interval=total_epochs)

data = dict(samples_per_gpu=24)  # single GPU setting

# optimizer
optimizer = dict(type="AdamW", lr=5e-4)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy="CosineAnnealing", min_lr=1e-6, by_epoch=False)

log_config = dict(
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHookEpoch"),
    ]
)
