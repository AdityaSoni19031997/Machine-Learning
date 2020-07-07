#!/usr/bin/env python
# coding: utf-8

import string
from requests import requests
from emojis import EMOTICONS, chat_words_str
from nltk.tokenize.treebank import TreebankWordTokenizer

white_list = string.ascii_letters + string.digits + ' '
white_list += "'"
symbols_to_delete = '\t'
symbols_to_isolate = '@=[.!,-:?&;/()#_*$]`~+\\¯¿½|}{%^‚¡´'

isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}

def handle_chat_words(chat_words_str=chat_words_str):

    chat_words_map_dict = {}
    chat_words_list = []
    for line in chat_words_str.split("\n"):
        if line != "":
            cw = line.split("=")[0]
            cw_expanded = line.split("=")[1]
            chat_words_list.append(cw)
            chat_words_map_dict[cw] = cw_expanded
    chat_words_list = set(chat_words_list)
    return chat_words_list

chat_words_list = handle_chat_words(chat_words_str)

def download_file(url, loc):

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(loc, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=258384)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return loc

def download_word_embeddings(save_path):

    if not os.path.isfile(save_path):
        print("Embedding vectors not found. Downloading")
        download_file(word_embeddings_url, loc=save_path)
    else:
        print("Embedding vectors exist. Skipping.")

def convert_emoticons(text):

    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', " ".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def handle_punctuation(x):

    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x

def chat_words_conversion(text):

    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def handle_contractions(x, tokenizer = TreebankWordTokenizer()):
    x = tokenizer.tokenize(x)
    x = ' '.join(x)
    return x

def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x

def meta_nlp_feats(df,col):
    
    df[col] = df[col].fillna("None")
    df['length'] = df[col].apply(lambda x : len(x))
    df['capitals'] = df[col].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['length']),axis=1)
    df['num_exclamation_marks'] = df[col].apply(lambda comment: comment.count('!'))
    df['num_question_marks'] = df[col].apply(lambda comment: comment.count('?'))
    df['num_punctuation'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
    df['num_symbols'] = df[col].apply(lambda comment: sum(comment.count(w) for w in '*&$%#'))
    df['num_words'] = df[col].apply(lambda comment: len(comment.split()))
    df['num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    df['num_smilies'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (":‑)",":)",":-]",":]",":-3",":3",":->",":>","8-)",
        "8)",":-}",":}",":o)",":c)",":^)","=]","=)",":‑D",":D","8‑D","8D","x‑D","xD","X‑D","XD","=D",
        "=3","B^D",":-))",";‑)",";)","*-)","*)",";‑]",";]",";^)",":‑,",";D",":‑P",":P","X‑P","XP", 
        "x‑p","xp",":‑p",":p",":‑Þ",":Þ",":‑þ",":þ",":‑b",":b","d:","=p",">:P", ":'‑)", ":')",  ":-*", ":*", ":×")))
    df['num_sad'] = df[col].apply(lambda comment: sum(comment.count(w) for w in (':-<', ':()', ';-()', ';(', ":‑(",":(",":‑c",":c",":‑<",
        ":<",":‑[",":[",":-||",">:[",":{",":@",">:(","D‑':","D:<","D:","D8","D;","D=","DX",":‑/",
        ":/",":‑.",'>:\\', ">:/", ":\\", "=/" ,"=\\", ":L", "=L",":S",":‑|",":|","|‑O","<:‑|")))
    return df
