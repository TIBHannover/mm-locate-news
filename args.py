import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='MM-Locate-News')

    parser.add_argument('--data_to_use', default='mm-locate-news', type=str)
    parser.add_argument('--data_path', default='/data/1/mm-locate-news/dataset')
    parser.add_argument('--model_name', default='v_loc_obj', type=str, help=' \
                        [v_clip, v_loc_obj, v_scene_obj, v_loc_scene, v_loc_scene_obj, v_clip_loc, v_clip_scene, v_loc_scene_obj, v_clip_loc_scene    , v_loc, v_scene, v_obj ], \
                        [t_body, t_entity ,t_2bert], \
                        [m_2bert_clip, m_2bert_clip_loc, m_2bert_clip_scene, m_2bert_loc_scene, m_2bert_clip_loc_scene ] ')

    parser.add_argument('--txt_dim', default=3072, type=int)
    parser.add_argument('--img_dim', default=2048, type=int)
    parser.add_argument('--clip_dim', default=512, type=int)
    parser.add_argument('--path_results', default='experiments/logs', type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--evaluation_results', default='experiments/evaluation', type=str)
    parser.add_argument('--emb_dim', default=1024, type=int)
    parser.add_argument('--n_classes', default=389, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--test_check_point', default='-', type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)  
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='-', type=str)
    parser.add_argument('--freeze_text', default=False, type=bool)
    parser.add_argument('--freeze_image', default=False, type=bool)
    parser.add_argument('--test_best_model', default=False, type=bool)
    parser.add_argument('--tensorboard', default=False, type=bool)

    return parser

