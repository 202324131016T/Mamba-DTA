from model.model_MambaDTA import net
# my_experiment_net
'''
model.model_CoVAE
model_MambaDTA
model_MambaDTA_none
model_MambaDTA_mamba
model_MambaDTA_ISF
'''

import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument( # epoch
        '--num_epoch',
        type=int,
        default=100, # 100
        help='Number of epochs to train.'
    )
    parser.add_argument(  # dataset path
        '--dataset',
        type=str,
        default='davis',  # davis  kiba
        help='Directory for input data.'
    )
    parser.add_argument( # data setting
        '--problem_type',
        type=int,
        default=2,  # 2 for drug setting and 3 for target setting
        help='Type of the prediction problem (2-3)'
    )
    parser.add_argument(  # only test (no train)
        '--only_test',
        type=bool,
        default=False,
    )
    parser.add_argument(  # epoch
        '--lr',
        type=float,
        default=0.0004,  # 0.001 0.0004
        help='lr'
    )
    parser.add_argument(  # batch_size
        '--batch_size',
        type=int,
        default=128,  # 256  128
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument( # proteins max len
        '--max_seq_len',
        type=int,
        default=0, # 1200 1000
        help='Length of input sequences.'
    )
    parser.add_argument( # drug max len
        '--max_smi_len',
        type=int,
        default=0, # 85 100
        help='Length of input sequences.'
    )
    parser.add_argument( # dataset path
        '--dataset_path',
        type=str,
        default='./data/',
        help='Directory for input data.'
    )
    parser.add_argument( # checkpoint path
        '--checkpoint_path',
        type=str,
        default='./checkpoints/',
        help='Path to write checkpoint file.'
    )
    parser.add_argument( # log path
        '--log_dir',
        type=str,
        default='./logs/',
        help='Directory for log data.'
    )
    parser.add_argument( # result path
        '--result_path',
        type=str,
        default='./result/',
        help='Path to write result file.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def logging(msg, FLAGS):
    # print("logging: ", msg)
    fpath = FLAGS.log_dir
    with open(fpath, "a" ) as fw:
      fw.write("%s\n" % msg)