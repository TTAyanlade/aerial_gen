# --------------------------------------------------------
# Based on the timm and MAE-priv code base
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/BUPT-PRIV/MAE-priv
# --------------------------------------------------------
import io
import os
from pathlib import Path

import torch

from .dist import save_on_master
from .model import get_state_dict
from utils.pos_embed import interpolate_pos_embed_multimae
import pdb
import pandas as pd
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, loss_balancer=None, model_ema=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args
            }

            if loss_balancer is not None:
                to_save['loss_balancer'] = loss_balancer.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, device = 'cpu'):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)
        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu')
            else:
                
                checkpoint = torch.load(args.resume, map_location='cpu')
                checkpoint_model = checkpoint['model']

                # Remove keys for semantic segmentation
                for k in list(checkpoint_model.keys()):
                    if "semseg" in k:
                        del checkpoint_model[k]

                # for k in list(checkpoint_model.keys()):
                #     if "depth" in k:
                #         del checkpoint_model[k]

                # Interpolate position embedding
                interpolate_pos_embed_multimae(model, checkpoint_model)
                
                #CODE TO LOAD KEYS MANUALLY FOR NEW MODALITIES 
                # making a list of keys (parameters) in the model that need to be updated manually
                # if 'UAV_RGB' in args.in_domains:
                #     list_uav_rgb = [f for f in list(model_without_ddp.state_dict().keys()) if 'UAV_RGB' in f]
                # if 'Sat_RGB' in args.in_domains:
                #     list_sat_rgb = [f for f in list(model_without_ddp.state_dict().keys()) if 'UAV_RGB' in f]
                # making a list of keys (parameters) in the checkpoint that we want to use for our new modality
                list_rgb = [f for f in checkpoint_model.keys() if 'rgb' in f]
                list_depth = [f for f in checkpoint_model.keys() if 'depth' in f]
                #loop through the model parameter list and update the weight values
                #model.state_dict()['input_adapters.UAV.pos_emb'].detach().cpu() ==checkpoint_model['input_adapters.rgb.pos_emb']
                #list2[0].replace("rgb", "UAV" )
                # Load pre-trained model
            #pdb.set_trace()
            msg = model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
            #print(msg)
            new_model_without_ddp_state_dict = model_without_ddp.state_dict()
            #new_model_without_ddp_state_dict = model_without_ddp.copy()
            #output_adapters.norm_UAV.decoder_transformer.0.mlp.fc1.bias
            #checkpoint_model.to(device)
            cnt = 0
            if 'UAV_RGB' in args.in_domains:
                for k in list_rgb:        
                    if "rgb" in k and ("depth" not in k):
                        k_uav = k.replace("rgb", "UAV_RGB")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_uav} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_uav] = checkpoint_model[k]
                        cnt +=1
            keys=[]
            if 'Sat_RGB' in args.in_domains:
                for k in list_rgb:        
                    if "rgb" in k and ("depth" not in k):
                        keys.append(k)
                        k_sat = k.replace("rgb", "Sat_RGB")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_sat} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_sat] = checkpoint_model[k]
                        cnt +=1
            # df = pd.DataFrame(keys, columns = ['RGB_Keys'])
            # df.to_csv('/work/mech-ai/aapowadi/MultiMAE/RGB_KEYS.csv')

            if 'nir' in args.in_domains:
                for k in list_depth:        
                    if "depth" in k and ("rgb" not in k):
                        k_sat_mul = k.replace("depth", "nir")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_sat_mul} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_sat_mul] = checkpoint_model[k]
                        cnt +=1
            if 'red_e' in args.in_domains:
                for k in list_depth:        
                    if "depth" in k and ("rgb" not in k):
                        k_sat_mul = k.replace("depth", "red_e")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_sat_mul} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_sat_mul] = checkpoint_model[k]
                        cnt +=1
            if 'd_blue' in args.in_domains:
                for k in list_depth:        
                    if "depth" in k and ("rgb" not in k):
                        k_sat_mul = k.replace("depth", "d_blue")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_sat_mul} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_sat_mul] = checkpoint_model[k]
                        cnt +=1
            if 'Sat_multispec' in args.in_domains:
                for k in list_rgb:        
                    if "rgb" in k and ("depth" not in k):
                        k_sat_mul = k.replace("rgb", "Sat_multispec")
                        #list2[cnt].replace("rgb", "UAV" )
                        print(f"replacing the values for {k_sat_mul} of created model from the {k} of the checpoint model ")
                        new_model_without_ddp_state_dict[k_sat_mul] = checkpoint_model[k]
                        cnt +=1

            

            #model_without_ddp.to(device)
            #model_without_ddp.state_dict()['output_adapters.norm_UAV.decoder_transformer.0.mlp.fc1.bias'] == checkpoint_model['output_adapters.norm_rgb.decoder_transformer.0.mlp.fc1.bias']
            msg1 = model_without_ddp.load_state_dict(new_model_without_ddp_state_dict, strict=False)
            print(msg1)
            #pdb.set_trace()
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])