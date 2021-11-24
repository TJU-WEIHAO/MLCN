import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    # base options
    parser.add_argument('--name', type=str, default='TEST')
    parser.add_argument("--stage", type=str, default='inf',
                        help='seg: stage_1, gmm: stage_2, tom: stage_3, inf: inference')
    parser.add_argument('--load_network', type=int, default=1, help='0 is retrain, 1 is load network')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./train_results/none/')
    parser.add_argument('--test_dir', type=str, default='./test_results/results/')

    # base options
    parser.add_argument("--gpu_ids", type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--istrain', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=192)
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')

    # train options
    parser.add_argument('--save_step', type=int, default=10000)
    parser.add_argument('--print_step', type=int, default=500)
    parser.add_argument('--display_count', type=int, default=500)
    parser.add_argument("--keep_step", type=int, default=200000)
    parser.add_argument("--decay_step", type=int, default=200000)
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard', help='save tensorboard infos')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test/paired_list.txt')

    # loading checkpoint
    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--tom_checkpoint', type=str, default='tom_final.pth')

    # common
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'],
                        default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt
