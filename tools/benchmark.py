import argparse
import time
import torch
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


# from mmdet3d.models import build_fusion_model
# from torchpack.utils.config import configs
# from mmdet3d.utils import recursive_eval

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--samples", default=2000, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # if args.cfg_options is not None:
    #     cfg.merge_from_dict(args.cfg_options)

    # configs.load(args.config, recursive=True)
    # cfg = Config(recursive_eval(configs), filename=args.config)

    cfg = Config.fromfile(args.config)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # cfg = compat_cfg(cfg)



    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # model = build_fusion_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.fp16:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % int(args.log_interval) == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f"Done image [{i + 1:<3}/ {args.samples}], "
                    f"fps: {fps:.1f} img / s"
                )

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f"Overall fps: {fps:.1f} img / s")
            break


if __name__ == "__main__":
    main()
