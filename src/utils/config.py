import argparse
import configparser
import os.path


def dump_configs(fp,cfgs):
    if not fp.endswith('.ini'):
        fp = fp + '.ini'
    cfgs = vars(cfgs)
    cfgs.pop('config')
    with open(fp, 'w') as fio:
        fio.write('[DEFAULT]\n')
        for arg in sorted(cfgs):
            attr = cfgs[arg]
            fio.write('{} = {}\n'.format(arg, attr))

def get_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    # ray casting and sampling settings
    parser.add_argument("--train_rays", type=int, default=4000,
                        help='number of sampling rays during training(decrease if running out of memory)')
    parser.add_argument("--test_rays", type=int, default=1024*14,
                        help='number of rays processed during test(decrease if running out of memory)')
    parser.add_argument("--N_samples", type=int, default=32,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=32,
                        help='number of additional fine samples per ray')
    parser.add_argument("--stratified", action='store_true', default=False,
                        help='stratified sampling')
    
    # training strategy
    parser.add_argument("--lrate", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=50000,
                        help='exponential learning rate decay')
    parser.add_argument("--scale", type=int, default=2, 
                        help="downsample scale of input image")
    parser.add_argument("--optimize_smpl", action='store_true', 
                        help="jointly optimize smpl parameters")
    
    # model options
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="dimension of latent code")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="dimension of query embedding ")
    parser.add_argument("--use_bkgd", action='store_true', 
                        help="use static nerf to model background")
    parser.add_argument("--use_direction", action='store_true', 
                        help="use relative direction features")

    # dataset options
    parser.add_argument('--output_dir', type=str, default='./logs',
                        help='where to store the checkpoints and output')
    parser.add_argument("--data_dir", type=str, default='./data/doublefusion',
                        help='input data directory')
    parser.add_argument("--test_dir", type=str, default='./data/doublefusion',
                        help='test data directory')
    parser.add_argument("--smpl_dir", type=str, default='./data/smpl',
                        help="smpl_path")

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=2000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path of pretrain model")

    #1. get the default
    cfgs = parser.parse_args()

    #2. overwrite with file
    if os.path.isfile(cfgs.config):
        cfg = configparser.ConfigParser()
        cfg.optionxform = lambda option: option
        with open(cfgs.config, 'r') as f:
            cfg.read_file(f)
        for k, v in cfg['DEFAULT'].items():
            if hasattr(cfgs,k):
                setattr(cfgs,k,type(getattr(cfgs,k))(v))
    
    #3. everthing else
    cfgs = parser.parse_args(namespace=cfgs)

    return cfgs
