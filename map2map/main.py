from .args import get_args
from . import train
from . import test
from . import generate
from . import generate_inv
from . import infer_disp


def main():

    args = get_args()

    if args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)
    elif args.mode == 'generate':
        generate.generate(args)
    elif args.mode == 'generate_inv':
        generate_inv.generate(args)
    elif args.mode == 'infer_disp':
        infer_disp.generate(args)

if __name__ == '__main__':
    main()
