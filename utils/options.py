import argparse


def get_args():
    parser = argparse.ArgumentParser(description="TransTextReID")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    # parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_dir", default="/data1/ccq/multimodality")
    # parser.add_argument("--gpu_id", default="0", help="select gpu to run")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.07, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')
    parser.add_argument("--nlp_aug", default=False, action='store_true')
    # parser.add_argument("--embed_dim", type=int, default=512, help="the final visual and textual feature dim")

    ## cross transfomer setting
    parser.add_argument("--num_colors", type=int, default=60, help="num colors of Mask Color Modeling labels")
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mcm task, 1.0 indicates mask every color in a caption")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--use_imageid", default=False, action='store_true', help="whether to use image_id info to build soft label.")
    parser.add_argument("--MCQ", default=False, action='store_true', help="whether to use Multiple Choice Questions dataset")
    parser.add_argument("--MCM", default=False, action='store_true', help="whether to use Mask Color Modeling dataset")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")
    parser.add_argument("--MSM", default=False, action='store_true', help="whether to use Mask Subsequence Matching dataset")
    parser.add_argument("--MCQMLM", default=False, action='store_true', help="whether to use MCQMLM dataset")
    parser.add_argument("--MSMMLM", default=False, action='store_true', help="whether to use MSMMLM dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='itc', help="which loss to use ['tcmpm','mcm', 'mcq', 'mlm', 'msm', 'id', 'itc', 'sdm']")
    parser.add_argument("--cmm_loss_weight", type=float, default=1.0, help="cross modal matching loss (tcmpm, cmpm, infonce...) weight")
    parser.add_argument("--mcm_loss_weight", type=float, default=1.0, help="mcm loss weight")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--mcq_loss_weight", type=float, default=1.0, help="mcq loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--learnable_loss_weight", default=False)
    parser.add_argument("--label_mix", default=False, action='store_true', help="whether mix pid and imagid label")
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=1e-8)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, F30K, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from type idtentity and random")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", type=str, default="/data0/data_ccq/CUHK-PEDES/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false') # whether in training mode
    parser.add_argument("--test_setting", type=int, default=0)

    ######################## multi-modality model settings ########################
    parser.add_argument("--fusion_way", default='add', help="[add, weight add, cross attention]") # whether use text and sketch fusion method
    parser.add_argument("--only_sketch", default=False, action='store_true', help="whether training with only sketch")
    parser.add_argument("--only_text", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--pa", type=float, default=0.1, help="parameter add for fusion")
    parser.add_argument("--only_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--four_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss2", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focal_three_fusion_loss3", default=False, action='store_true', help="sketch label kl")
    parser.add_argument("--focal_three_fusion_loss4", default=False, action='store_true', help=" text label kl")
    parser.add_argument("--focal_three_fusion_loss5", default=False, action='store_true', help=" text label two kl")
    parser.add_argument("--focal_three_fusion_loss6", default=False, action='store_true', help=" text label two kl")
    parser.add_argument("--focalthree_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--focalthree_four_fusion_loss", default=False, action='store_true', help="whether training with only text")
    parser.add_argument("--al", type=float, default=1.0, help="parameter add for fusion")
    parser.add_argument("--ga", type=float, default=2.0, help="parameter add for fusion")
    parser.add_argument("--klp", type=float, default=1.0, help="parameter add for fusion")

    args = parser.parse_args()

    return args