import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_bic = float("inf")
        best_model = None

        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                # model.score computes log probability
                # https://github.com/hmmlearn/hmmlearn/blob/7dc439708a8c102fbb77daeb784ca87637308bc4/hmmlearn/base.py#L219-L252
                logL = model.score(self.X, self.lengths)
                # p is calculated as per Dana's suggestion here:
                # https://ai-nd.slack.com/files/ylu/F4S90AJFR/number_of_parameters_in_bic.txt
                #  p = n*(n-1) + (n-1) + 2*d*n = n^2 + 2*d*n - 1
                p = n_components**2 + 2 * len(self.X[0]) * n_components - 1
                N = np.sum(self.lengths)
                bic = -2 * logL + p * np.log(N)

                if bic < min_bic:
                    min_bic = bic
                    best_model = model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_dic = float("-inf")
        best_model = None

        # Following Andrew Parsons's comment:
        # https://ai-nd.slack.com/archives/C4GQUB39T/p1490874184037691
        # We want to calculate the largest difference between the logL
        # for the target word and the average logL for all other words.
        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
            except:
                model = self.base_model(1)
                logL = model.score(self.X, self.lengths)

            other_words = self.words.copy()
            del other_words[self.this_word]

            logL_other_words_total = 0
            for word in other_words:
                other_x, other_length = self.hwords[word]
                try:
                    logL_other_words_total += model.score(other_x, other_length)
                except:
                    pass

            logL_other_words_average = logL_other_words_total/len(other_words)
            dic = logL - logL_other_words_average
            if dic > max_dic:
                max_dic = dic
                best_model = model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = float("-inf")
        best_n_components = 1

        for n_components in range(self.min_n_components,
                                  self.max_n_components + 1):

            # We can maximize the total (instead of avg) to keep it simple
            total_score = 0
            split_method = KFold()

            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    cv_train_x,
                    cv_train_length = combine_sequences(cv_train_idx,
                                                        self.sequences)
                    _,
                    cv_test_length = combine_sequences(cv_test_idx,
                                                       self.sequences)
                    model = GaussianHMM(n_components=n_components,
                                        covariance_type="diag",
                                        n_iter=1000,
                                        random_state=self.random_state,
                                        verbose=False).fit(cv_train_x,
                                                           cv_train_length)
                    total_score += model.score(cv_test_idx, cv_test_length)
            except:
                pass

            if total_score > max_score:
                max_score = total_score
                best_n_components = n_components

        return self.base_model(best_n_components)
