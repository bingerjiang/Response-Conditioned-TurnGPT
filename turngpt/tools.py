def remove_nonutt_tokens (dialogs):
    '''
    dialogs: list of lists
    '''
    cleaned_dialogs = []
    for dialog in dialogs:
        cleaned_dialog = []
        for utt in dialog:
            if utt != '' and utt[0]!='<':
                cleaned_dialog.append(utt)
        cleaned_dialogs.append(cleaned_dialog)
    return cleaned_dialogs