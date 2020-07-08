#fleet 1

lgb_param = {
    'min_data_in_leaf': 106, 
    'num_leaves': 500, 
    'learning_rate': 0.005,
    'min_child_weight': 0.03454472573214212,
    'bagging_fraction': 0.4181193142567742, 
    'feature_fraction': 0.3797454081646243,
    'max_depth': -1, 
    'objective': 'binary',
    'seed': SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
    'boosting_type': 'gbdt',
    'verbose': 1,
    'metric':'auc',
}
import json
with open('lgbm_model_1.txt', 'w') as json_file:
    json.dump(lgb_param, json_file)

#fleet 2

lgb_param = {
    'min_data_in_leaf': 106, 
    'num_leaves': 500, 
    'learning_rate': 0.009,
    'min_child_weight': 0.03454472573214212,
    'bagging_fraction': 0.4181193142567742, 
    'feature_fraction': 0.3797454081646243,
    #'reg_lambda': 0.6485237330340494,
    #'reg_alpha': 0.3899927210061127,
    'max_depth': 10,
    'objective': 'binary',
    'seed': SEED,
    'feature_fraction_seed': SEED,
    'bagging_seed': SEED,
    'drop_seed': SEED,
    'data_random_seed': SEED,
    'boosting_type': 'gbdt',
    'verbose': 1,
    'metric':'auc',
}
import json
with open('lgbm_model_1.txt', 'w') as json_file:
    json.dump(lgb_param, json_file)


def get_sentences(input_file_pointer):
    while True:
        line = input_file_pointer.readline()
        if not line:
            break
        yield line
import re
def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r’[^a-z0-9\s]’, '’, sentence)
    return re.sub(r’\s{2,}’, ' ', sentence)

def generate_candidate_phrases(text, stopwords):
    """ generate phrases using phrase boundary markers """
 
    # generate approximate phrases with punctation
    coarse_candidates = char_splitter.split(text.lower())
 
    candidate_phrases = []
 
    for coarse_phrase\
            in coarse_candidates:
 
        words = re.split("\\s+", coarse_phrase)
        previous_stop = False
 
        # examine each word to determine if it is a phrase boundary marker or part of a phrase or lone ranger
        for w in words:
 
            if w in stopwords and not previous_stop:
                # phrase boundary encountered, so put a hard indicator
                candidate_phrases.append(";")
                previous_stop = True
            elif w not in stopwords and len(w) > 3:
                # keep adding words to list until a phrase boundary is detected
                candidate_phrases.append(w.strip())
                previous_stop = False
 
    # get a list of candidate phrases without boundary demarcation
    phrases = re.split(";+", ' '.join(candidate_phrases))
 
    return phrases

def generate_and_tag_phrases(text_rdd,min_phrase_count=50):
    """Find top phrases, tag corpora with those top phrases"""
 
    # load stop words for phrase boundary marking
    logging.info ("Loading stop words...")
    stopwords = load_stop_words ()
 
    # get top phrases with counts > min_phrase_count
    logging.info ("Generating and collecting top phrases...")
    top_phrases_rdd = \
        text_rdd.map(lambda txt: remove_special_characters(txt))\
        .map(lambda txt: generate_candidate_phrases(txt, stopwords)) \
        .flatMap(lambda phrases: phrase_to_counts(phrases)) \
        .reduceByKey(add) \
        .sortBy(lambda phrases: phrases[1], ascending=False) \
        .filter(lambda phrases: phrases[1] >= min_phrase_count) \
        .sortBy(lambda phrases: phrases[0], ascending=True) \
        .map(lambda phrases: (phrases[0], phrases[0].replace(" ", "_")))
 
    shortlisted_phrases = top_phrases_rdd.collectAsMap()
    logging.info("Done with phrase generation...")
 
    # write phrases to file which you can use down the road to tag your text
    logging.info("Saving top phrases to {0}".format(phrases_file))
    with open(os.path.join(abspath, phrases_file), "w") as f:
        for phrase in shortlisted_phrases:
            f.write(phrase)
            f.write("\n")
 
    # tag corpora and save as new corpora
    logging.info("Tagging corpora with phrases...this will take a while")
    tagged_text_rdd = text_rdd.map(
            lambda txt: tag_data(
                txt,
                shortlisted_phrases))
 
    return tagged_text_rdd