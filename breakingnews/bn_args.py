import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='BreakingNews')
    parser.add_argument('--data_path', default='dataset')
    parser.add_argument('--data_to_use', default='breaking-news', type=str)
    parser.add_argument('--results_path', default = 'breakingnews/results')
    parser.add_argument('--emb_dim', default = 1024, type=int)
    parser.add_argument('--txt_dim', default=3072, type=int)
    parser.add_argument('--img_dim', default=512, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--resume', default='-', type=str)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--path_results', default='experiments/logs', type=str)
    parser.add_argument('--freeze_text', default=False, type=bool)
    parser.add_argument('--freeze_image', default=False, type=bool)
    parser.add_argument('--tensorboard', default=False, type=bool)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float) 
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--evaluation_results', default='experiments/evaluation', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    
    parser.add_argument('--model_name', default='reg_v_clip', type=str, help='reg_v_clip, reg_t_2bert, reg_mm_clip_loc_scene')
    parser.add_argument('--check_point',  default ='epoch_20.pth.tar'  )
    return parser
