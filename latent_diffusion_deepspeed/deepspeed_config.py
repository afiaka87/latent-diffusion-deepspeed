def deepspeed_config_from_args(args):
    deepspeed_config = {
        "zero_optimization": { 
            "stage": 1,
            "round_robin_gradients": True,
        },
        'train_micro_batch_size_per_gpu': args.batch_size,
        'gradient_accumulation_steps': args.ga_steps,
        'gradient_clipping': 1.0,
        'fp16': {
            'enabled': args.use_fp16,
            'initial_scale_power': 24, # the default, often it's better to start lower around 16-24
        },
        "tensorboard": {
            "enabled": True,
            "output_path": f"tensorboard_logs/{args.wandb_project}",
            "job_name": f"{args.wandb_project}",
        },
        "steps_per_print": args.log_interval,
        "wall_clock_breakdown": False,
        "zero_allow_untested_optimizer": True
    }
    return deepspeed_config


