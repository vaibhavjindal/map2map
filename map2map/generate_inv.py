import os
import sys
import warnings
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GenerateFieldDataset
from .data import norms
from . import models
from .models import narrow_cast
from .models.gen_lin_field import gen_lin_field
from .utils import import_attr, load_model_state_dict
from .utils.timer import Timer

def generate(args):

    if torch.cuda.is_available():
        #
        # Change this to run v2v and d2d on separate gpus?
        #
        if torch.cuda.device_count() > 1:
            warnings.warn('Not parallelized but given more than 1 GPUs')

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda', 0)

        torch.backends.cudnn.benchmark = True
    else:  # CPU multithreading
        device = torch.device('cpu')

        if args.num_threads is None:
            args.num_threads = int(os.environ['SLURM_CPUS_ON_NODE'])

        torch.set_num_threads(args.num_threads)

    if args.verbose :
        print('pytorch {}'.format(torch.__version__))
        print()
        pprint(vars(args))
        print()
        sys.stdout.flush()

    timer = Timer(len("Computing displacements 0000 of 0000... "))

    try :
        os.makedirs(args.out_dir)
    except OSError as e :
        if type(e) is not FileExistsError :
            raise e

    in_filename = "./" + args.out_dir + "/dis.npy"

    

    generate_dataset = GenerateFieldDataset(
        style_pattern=args.gen_style_pattern,
        in_patterns=[in_filename],
        in_norms = args.in_norms,
        callback_at=args.callback_at,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )

    generate_loader = DataLoader(
        generate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = generate_dataset.style_size
    chan = generate_dataset.in_chan
    num_batches = len(generate_loader)
    log10batch = int(np.floor(np.log10(num_batches)))
    print_factor = max(1, int(np.round(num_batches / 10)))

    # timer.printDone()

    if not args.no_dis :
        d2d_model = import_attr("d2d.StyledVNet", models, callback_at=args.callback_at)
        d2d_model = d2d_model(style_size, sum(chan), sum(chan),
                      scale_factor=args.scale_factor, **args.misc_kwargs)
        d2d_model.to(device)
        state = torch.load(os.path.dirname(__file__) + "/model_parameters/state_128.pt", map_location=device)
        load_model_state_dict(d2d_model, state['model'])
        d2d_model.eval()
        with torch.no_grad():
            for i, data in enumerate(generate_loader):
                if i % print_factor == 0 :
                    timer.printStart("Computing lin field %s of %d..." % (str(i + 1).zfill(log10batch + 1), num_batches))
                style, input = data['style'], data['input']
                style = style.to(device, non_blocking=True)
                input = input.to(device, non_blocking=True)
                output = narrow_cast(d2d_model(input, style))[0]
                # norms.cosmology.dis(output, undo=False, **args.misc_kwargs)
                generate_dataset.assemble('lin_inv_model', chan, output, [[args.out_dir + "/"]])
                if (i + 1) % print_factor == 0 or i == num_batches - 1 :
                    timer.printDone()

