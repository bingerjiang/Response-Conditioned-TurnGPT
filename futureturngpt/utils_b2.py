def calculate_main_attention (attns, percent = 0.9):
    '''
    attention: one per each converstation, only last one permutated
    [batch_size, n_head, len_dialog, len_dialog]
    '''
    furthest_indices_all = []
    rel_furthest_indices_all = []
    for attn in attns:
        
    return furthest_indices_all, rel_furthest_indices_all