'''
Define the function that trains the model.
It is in a separated file because it is probably a long function.
'''
from utils import *
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss


def train_model(device, train_loader, val_loader, prepro_mod, encoder, name2branch, objs, train_args, branch2params, evals):

    epochs = train_args['epochs']
    lr = train_args['lr']
    eval_every = train_args['eval_every']
    save_every = train_args['save_every']
    eval_samps = train_args['eval_samps']
    save_name = train_args['save_name']

    mods = list(name2branch.values()) + [encoder]
    optimizer = torch.optim.Adam([param for mod in mods for param in mod.parameters()], lr = lr)

    # ====== Does a single module has the .train() function?
    prepro_mod.train()
    encoder.train()

    # iterate through epochs and batches
    for epoch_idx in range(epochs):
        print ('\n===== EPOCH %d =====' % epoch_idx)

        if not epoch_idx % eval_every:
            eval_model(val_loader, prepro_mod, encoder, evals, eval_samps)

        if not epoch_idx % save_every:
            save_model(save_name + '_epoch%d' % epoch_idx, prepro_mod, encoder)

        # obj2loss saves a tuple recording (summation of losses, summation of weighted losses) for each learning objective.
        obj2sum_losses = {obj_dict['name']: {'sum_loss': 0.0, 'sum_weighted_loss': 0.0} for obj_dict in objs}
        sum_ttl_loss = 0.0
        samp_cnt = 0

        for b_idx, batch_dict in enumerate(train_loader):
            print ('\tbatch_idx: %d / %d' % (b_idx, len(train_loader)))
            samp = batch_dict['samp'].to(device)
            samp_cnt += len(samp)

            # Forward propagation for encoder and branches.
            emb_tmp = prepro_mod(samp)
            emb = encoder(emb_tmp)

            branch2out = dict()
            for name, branch_mod in name2branch.items():
                branch_mod.train()
                branch2out[name] = branch_mod(emb)


            # calculate losses from each learning objective and add to total loss
            ttl_loss = 0.0

            for obj_dict in objs:
                obj_name = obj_dict['name']
                obj_type = obj_dict['type']
                loss_fn = obj_dict['loss_fn']
                branch_name = obj_dict['branch']
                start_idx, end_idx = obj_dict['indices']
                alpha = obj_dict['alpha']
                branch_type = branch2params[branch_name]['type']
                
                out = branch2out[branch_name]

                # Get prediction values from the desired branch and slicing indices
                if branch_type == 'decoder':
                    # Slice along the channel dimension
                    pred = out[:, start_idx: end_idx, :, :, :]
                elif branch_type == 'predictor':
                    pred = out[:, start_idx: end_idx]
                else:
                    raise Exception ('Invalid branch type found.')


                # Calculate the loss terms of each training objective according to objective type and loss function
                if obj_type in ['reconstruction', 'pred_stack']:
                    
                    if obj_type == 'reconstruction':
                        label = samp
                    elif obj_type == 'pred_stack':
                        label = batch_dict[obj_name].to(device)

                    if loss_fn == 'mse':
                        loss = mse_loss(pred, label)
                    
                    elif loss_fn == 'BCE_w_logits':
                        flat_label_tmp = torch.moveaxis(label, 1, -1)
                        flat_label = flat_label_tmp.reshape((-1, flat_label_tmp.shape[-1]))

                        flat_pred_tmp = torch.moveaxis(pred, 1, -1)
                        flat_pred = flat_pred_tmp.reshape((-1, flat_pred_tmp.shape[-1]))

                        loss = binary_cross_entropy_with_logits(flat_pred, flat_label)
                    
                    else:
                        raise Exception ('Invalid loss_fn found.')

                elif obj_type in ['curr_value', 'pred_value']:
                    label = batch_dict[obj_name].to(device)
                    if loss_fn == 'mse':
                        loss = mse_loss(pred, label)
                    else:
                        raise Exception ('Invalid loss_fn found.')
                
                else:
                    raise Exception ('Invalid obj_type found.')

                obj_loss = alpha * loss
                ttl_loss += obj_loss

                obj2sum_losses[obj_name]['sum_loss'] += loss.detach().cpu()
                obj2sum_losses[obj_name]['sum_weighted_loss'] += obj_loss.detach().cpu()

            sum_ttl_loss += ttl_loss.detach().cpu()

            # Train the model
            optimizer.zero_grad()
            ttl_loss.backward()
            optimizer.step()


            ####### For Debugging #######
            # if b_idx == 3:
            #   break

        print_obj2sum_losses(obj2sum_losses, sum_ttl_loss, samp_cnt)



