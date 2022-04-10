import random
import pathlib
import torch
import shutil
import glob
import os

DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'
KEEP_N_CHECKPOINTS = 10

### Convenience functions

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, pathlib.Path):
        cp_path = pathlib.Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = pathlib.Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


### Checkpointing
@torch.no_grad()
def save_model(model, path: str, is_root: bool, epoch=0, using_deepspeed=False, opt=None):
    save_obj = {'epoch': epoch, }
    if using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')
        if KEEP_N_CHECKPOINTS is not None and is_root:
            checkpoints = sorted(glob.glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
            for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                shutil.rmtree(checkpoint)

        model.save_checkpoint(cp_dir, client_state=save_obj)
        if not is_root: return
        # Save a nonsense value that directs the user to convert the checkpoint to a normal 32-bit pytorch model.
        save_obj = {
            **save_obj,
            'weights': (
                'To get a working standard checkpoint, '
                'look into consolidating DeepSpeed checkpoints.'
            ),
        }
        torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
    if not is_root: return
    save_obj = { **save_obj, 'weights': model.state_dict(), }
    if opt is not None:
        save_obj = { **save_obj, 'opt_state': opt.state_dict(), }
    torch.save(save_obj, path)


# dataloader = create_dataloader(
#     distr_backend,
#     data_dir,
#     batch_size,
#     image_size,
#     # dataset_length=max_steps*batch_size,
#     dataset_length=max_steps, #TODO
#     random_crop=random_crop,
#     random_flip=random_flip,
#     use_webdataset=use_webdataset,
#     num_workers=num_workers,
# )
@torch.no_grad()
def ldm_encode_data_gn(dataloader, encoder, bert, device, use_fp16):
    with torch.cuda.amp.autocast(enabled=use_fp16):
        for text, batch in dataloader:
            text_emb = bert.encode(list(text)).to(device)
            text_blank = bert.encode(['']*batch.shape[0]).to(device)
            for i in range(batch.shape[0]):
                if random.randint(0, 100) < 20:
                    text_emb[i] = text_blank[i]

            # model_kwargs["context"] = text_emb
            model_kwargs = {"context": text_emb}
            batch = batch.to(device)
            emb = encoder.encode(batch).sample()
            emb *= 0.18215
            yield emb, model_kwargs


