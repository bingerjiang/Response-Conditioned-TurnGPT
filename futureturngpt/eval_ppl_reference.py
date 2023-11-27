elif args.eval_mode == 'compare':
    import time
    start_time = time.time()
    ## then the thresholds have been established.
    ## no need to search for best threshold
    with torch.no_grad():
        from original_turngpt.model import originalTurnGPT, originalTurnGPTWandbCallbacks
        if model.trp_eval_type == 'trp_probs':
            threshold_1 = 0.35
            threshold_2 = 0.39
        else:
            threshold_1 = 0.39
            threshold_2 = 0.38
        print('trp eval type: ', model.trp_eval_type)
        print('thresholds: ', threshold_1, threshold_2)
        correct_turns_1 = 0
        correct_turns_2 = 0
        num_both_correct = 0
        num_both_wrong = 0
        num_turngpt_better = 0
        num_cond_better = 0
        num_cond_below_threshold = 0
        num_turngpt_below_threshold = 0
        num_cond_pos_wrong = 0
        num_turngpt_pos_wrong =0
        num_turngpt_threshold_better = 0
        num_turngpt_pos_better = 0
        n_processed = 0
        
        model2 = originalTurnGPT.load_from_checkpoint(
            join(args.checkpoint2_path))
        model2.eval()
        for batch_idx, batch in enumerate(dm.val_dataloader()):
            #while batch_idx <468:
            print(batch_idx/len(dm.val_dataloader()))
            input_ids = batch['input_ids']
            ts_idx = [(sublist == 50257).nonzero(as_tuple=True)[0] for sublist in input_ids]

            ## no need for labels
            ## labels are only for loss, we don't need loss
            out, _ = model(
                batch['input_ids'].to(model.device), 
                speaker_ids=batch["speaker_ids"].to(model.device),
            )
            out2 = model2(
                batch['input_ids'].to(model.device), 
                speaker_ids=batch["speaker_ids"].to(model.device),
            )
            #pdb.set_trace()

            if not model.no_lm and not model.negs_only:
                out["probs"] = out["logits"].softmax(dim=-1)
                out["trp_probs"] = model.get_trp(out["probs"])
                out2["probs"] = out2["logits"].softmax(dim=-1)
                out2["trp_probs"] = model2.get_trp(out["probs"])
            if "mc_logits" in out:
                out["trp_proj"] = out["mc_logits"].sigmoid()
                out2["trp_proj"] = out2["mc_logits"].sigmoid()  
            if model.trp_eval_type not in ["trp_proj", "trp_probs"]:
                print("wrong trp_eval_type")
        #else:
            out_trp_1 = out[model.trp_eval_type].cpu()
            out_trp_2 = out2[model.trp_eval_type].cpu()
            assert(out_trp_1.shape == out_trp_2.shape)
            #n_turns = 0
            ##!!! ts index is not the same for the two models because of the insersion of next_utt_embedding!!!!
            for b in range(out_trp_1.shape[0]):
                dialog_ts = ts_idx[b]
                #n_turns += len(dialog_ts)-1
                
                ith_turn = 0              
                while ith_turn < len(dialog_ts)-1:
                    turnlevel_1 = False
                    turnlevel_2 = False
                    if ith_turn == 0:
                        start_idx = 0
                        end_idx = dialog_ts[ith_turn]+1
                    else:
                        start_idx = dialog_ts[ith_turn-1]+1
                        end_idx = dialog_ts[ith_turn]+1
                    trp_proj_1 = out_trp_1[b].cpu()
                    trp_proj_2 = out_trp_2[b].cpu()
                    current_utt_1 = trp_proj_1[start_idx:end_idx]
                    current_utt_2 = trp_proj_2[start_idx:end_idx-1] # because turngpt doesn't need to shift by 1
                    idx_above_threshold_1 = [i for i in range(len(current_utt_1)) if current_utt_1[i]>threshold_1]
                    idx_above_threshold_2 = [i for i in range(len(current_utt_2)) if current_utt_2[i]>threshold_2]
                    if len(idx_above_threshold_1)==1:
                        if idx_above_threshold_1[0] == len(current_utt_1)-1:
                    ## if only one satisfies and it is the last position where it is the ts
                    ## this is the only case it satisfies. add 1
                            correct_turns_1+=1
                            turnlevel_1 = True
                        else: # wrong position
                            num_cond_pos_wrong +=1
                            conversation = batch['input_ids'][b].cpu()
                            utt = batch['input_ids'][b][start_idx:end_idx-1].cpu() #-1 because input_ids are not shifted, as opposed to trp_prob
                            #print('cond wrong pos')
                            #pdb.set_trace()
                            with open('cond_wrong_pos_'+args.trp_eval_type+'_'+args.write_turngpt_better+'.txt','a') as f:
                                f.writelines("conversation: ")
                                f.writelines(model.tokenizer.decode(conversation))
                                f.writelines('\n')                                    
                                f.writelines("utt: ")
                                f.writelines(model.tokenizer.decode(utt))
                                f.writelines('\n')
                                f.writelines("pos: ")
                                #f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]+1]))
                                f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]])) 
                                # No need to +1
                                # trp_idx already is the +1 version
                                f.writelines('\n')
                    else:
                        max_trp_idx = torch.max(current_utt_1,0).indices
                        max_trp = torch.max(current_utt_1,0).values
                        if len(idx_above_threshold_1)<1: # max < threshold
                            assert(max_trp<threshold_1)
                            num_cond_below_threshold +=1
                        else: # wrong position
                            num_cond_pos_wrong +=1
                            conversation = batch['input_ids'][b].cpu()
                            utt = batch['input_ids'][b][start_idx:end_idx-1].cpu() #-1 because input_ids are not shifted, as opposed to trp_prob
                            #print('cond wrong pos')
                            #pdb.set_trace()
                            with open('cond_wrong_pos_'+args.trp_eval_type+'_'+args.write_turngpt_better+'.txt','a') as f:
                                f.writelines("conversation: ")
                                f.writelines(model.tokenizer.decode(conversation))
                                f.writelines('\n')                                    
                                f.writelines("utt: ")
                                f.writelines(model.tokenizer.decode(utt))
                                f.writelines('\n')
                                f.writelines("pos: ")
                                #f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]+1]))
                                f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]])) 
                                f.writelines('\n')
                    
                    
                    if len(idx_above_threshold_2)==1:
                        if idx_above_threshold_2[0] == len(current_utt_2)-1:
                    ## if only one satisfies and it is the last position where it is the ts
                    ## this is the only case it satisfies. add 1
                            correct_turns_2+=1
                            turnlevel_2 = True
                        else: # wrong position
                            num_turngpt_pos_wrong +=1
                            conversation = batch['input_ids'][b].cpu()
                            utt = batch['input_ids'][b][start_idx:end_idx-1].cpu()
                            #print('turngpt wrong pos')
                            #pdb.set_trace()
                            with open('turngpt_wrong_pos_'+args.trp_eval_type+'_'+args.write_turngpt_better+'.txt','a') as f:
                                f.writelines("conversation: ")
                                f.writelines(model2.tokenizer.decode(conversation))
                                f.writelines('\n')
                                f.writelines("utt: ")
                                f.writelines(model2.tokenizer.decode(utt))
                                f.writelines('\n')
                                f.writelines("pos: ")
                                f.writelines(model2.tokenizer.decode(utt[:idx_above_threshold_2[0]+1]))
                                f.writelines('\n')
                    else:
                        max_trp_idx = torch.max(current_utt_2,0).indices
                        max_trp = torch.max(current_utt_2,0).values
                        if len(idx_above_threshold_2)<1: # max < threshold
                            assert(max_trp<threshold_2)
                            num_turngpt_below_threshold +=1
                        else: # wrong position
                            num_turngpt_pos_wrong +=1
                            conversation = batch['input_ids'][b].cpu()
                            utt = batch['input_ids'][b][start_idx:end_idx-1].cpu()
                            #print('turngpt wrong pos')
                            #pdb.set_trace()
                            with open('turngpt_wrong_pos_'+args.trp_eval_type+'_'+args.write_turngpt_better+'.txt','a') as f:
                                f.writelines("conversation: ")
                                f.writelines(model2.tokenizer.decode(conversation))
                                f.writelines('\n')
                                f.writelines("utt: ")
                                f.writelines(model2.tokenizer.decode(utt))
                                f.writelines('\n')
                                f.writelines("pos: ")
                                f.writelines(model2.tokenizer.decode(utt[:idx_above_threshold_2[0]+1]))
                                f.writelines('\n')
                    if turnlevel_1:
                        if turnlevel_2:
                            num_both_correct +=1
                        else:
                            num_cond_better +=1
                    else:
                        if turnlevel_2:
                            num_turngpt_better +=1
                            if idx_above_threshold_1 ==[]:
                                num_turngpt_threshold_better +=1
                            else:
                                num_turngpt_pos_better +=1
                                if args.write_turngpt_better:
                                    with open('turngpt_better_'+args.trp_eval_type+'_'+args.write_turngpt_better+'.txt','a') as f:
                                        # utt = batch['input_ids'][b].cpu()
                                        # f.writelines('entire dialog:')
                                        # f.writelines(model.tokenizer.decode(utt))
                                        # f.writelines('\n')
                                        # f.writelines('utt:')
                                        # f.writelines(model.tokenizer.decode(utt[start_idx:end_idx]))
                                        # f.writelines('\n')
                                        conversation = batch['input_ids'][b].cpu()
                                        utt = batch['input_ids'][b][start_idx:end_idx-1].cpu()
                                        f.writelines("conversation: ")
                                        f.writelines(model.tokenizer.decode(conversation))
                                        f.writelines('\n')                                    
                                        f.writelines("utt: ")
                                        f.writelines(model.tokenizer.decode(utt))
                                        f.writelines('\n')
                                        f.writelines("pos: ")
                                        #f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]+1]))
                                        
                                        f.writelines(model.tokenizer.decode(utt[:idx_above_threshold_1[0]])) 
                                        
                                        f.writelines('\n')                            
                        else:
                            num_both_wrong +=1
                    
                    
                    
                    ith_turn += 1
        #self.n_turns += n_turns            
        # if 50259 not in input_ids and 50258 not in input_ids:
        #     i = 0
        #     while i < len(input_ids):
        #         if input_ids[i][-1] in [50256, 50257]: # if <endoftext> or <ts>
        #             # remove last sentence
        #             try:
        #                 second_last_ts_idx = ts_idx_long[i][-2]                   
        #                 pad_len = len(input_ids[i][second_last_ts_idx+1:])
        #                 pad_tensor = torch.Tensor([50256]).to(torch.int64).repeat(pad_len)
        #                 input_ids[i][second_last_ts_idx+1:]= pad_tensor
        #             except IndexError:
        #                 print('ts_idx[i]', ts_idx_long[i])
        #         else: # end with word token, so last sentence is truncated
        #             try:
        #                 last_ts_idx = ts_idx_long[i][-1]
        #                 pad_len = len(input_ids[i][last_ts_idx+1:])
        #                 pad_tensor = torch.Tensor([50256]).to(torch.int64).repeat(pad_len)
        #                 input_ids[i][last_ts_idx+1:] = pad_tensor
        #             except IndexError:
        #                 print('ts_idx[i]', ts_idx_long[i])

        #         i+=1
            n_processed += out_trp_1.shape[0]
    print("number of hours: ",(time.time()-start_time)/3600)
    pdb.set_trace()