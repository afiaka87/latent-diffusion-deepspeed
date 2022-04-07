def deepspeed_config_from_args(args):
    return {
        "zero_optimization": {
            "stage": 1,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": args.min_lr,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "total_num_steps": args.max_steps,
            }
        },
        'train_batch_size': args.batch_size,
        # 'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': 1.0,
        'fp16': {
            'enabled': args.use_fp16,
            'initial_scale_power': 28, # the default, often it's better to start lower around 16-24
        },
        "tensorboard": {
            "enabled": True,
            "output_path": "finetune-ldm-logs/",
            "job_name": "finetune-ldm"
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False,
    }



def distributed_setup(model, optimizer, data, distr_backend, args, use_webdataset):
    training_data = None
    if not use_webdataset: # esoteric bug in deepspeed
        training_data = data
        
    deepspeed_config = deepspeed_config_from_args(args)
    (distributed_model, distributed_optimizer, distributed_dataloader, distributed_scheduler) = distr_backend.distribute(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=[x for x in model.parameters() if x.requires_grad],
        training_data=training_data,
        lr_scheduler=None, # TODO: allow for pytorch scheduler
        config_params=deepspeed_config,
    )
    return distributed_model, distributed_optimizer, distributed_dataloader, distributed_scheduler