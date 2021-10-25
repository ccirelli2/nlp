import os


def save_load_gensim_objs(g_dict: dict, operation:str,
    filename_dict: str, filename_corpus: str, dir_output: str):
    """
    Function to save and load gensim dict and corpus objects.

    Parameters
    ::g_dict: dict,
    ::g_corpus: dict,
    ::operation:str,
    ::filename_dict: str,
    ::filename_corpus: str
    ::dir_output: str,

    """
    if operation.upper() == 'SAVE':
        g_dict.save(os.path.join(filename_dict))
        g_corpus.corpora.MmCorpus(os.path.join(dir_output, filename_corpus))
    else:
        g_dict_loaded = corpora.Dictionary.load(os.path.join(filename_dict))
        return g_dict_loaded
