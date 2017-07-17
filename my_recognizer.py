import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
        both lists are ordered by the test set word_id
        probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
        guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # dictionary of (X, lengths) tuple, where X is a numpy array of feature lists and lengths is
    # a list of lengths of sequences within X
    # {'FRANK': (array([[ 87, 225],[ 87, 225], ...  [ 87, 225,  62, 127], [ 87, 225,  65, 128]]), [14, 18]),
    #             ...}
    for X, lengths in test_set.get_all_Xlengths().values():
        # each key a word and value is Log Liklihood
        probability = {}

        for word, model in models.items():
            # The hmmlearn library may not be able to train or score all models.
            try:
                probability[word] = model.score(X, lengths)
            except:
                probability[word] = float("-inf")

        word_with_max_probability =\
            max(list(probability.items()), key=lambda x: x[1])[0]
        guesses.append(word_with_max_probability)
        probabilities.append(probability)

    return probabilities, guesses
