from box import Box

config = {
    "num_devices": 1,
    "batch_size": 16,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 250,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_t',
        "checkpoint": "weights/mobile_sam_add_text.pt",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "datasets/val2017",
            "annotation_file": "datasets/annotations/instances_val2017.json"
        },
        "val": {
            "root_dir": "datasets/val2017",
            "annotation_file": "datasets/annotations/instances_val2017.json"
        }
    }
}

cfg = Box(config)
