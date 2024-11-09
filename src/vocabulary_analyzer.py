import nltk

modal_verbs = ['can', 'may', 'must', 'shall', 'will', 'could', 'might', 'should', 'would']

def identifying_vocabulary_set(tokens):
    filtered_tokens = [token for token in tokens if token.isalpha()]
    vocabulary_set = set(filtered_tokens)
    vocabulary_size = len(vocabulary_set)
    return vocabulary_set, vocabulary_size

def identifying_vocabulary_tokens(tokens): 
    filtered_tokens = [token for token in tokens if token.isalpha()]
    total_tokens = len(filtered_tokens)
    return total_tokens

def identifying_vocabulary_adjectives(tagged_tokens):
    adjectives = [word for word, pos in tagged_tokens if pos.startswith('JJ')]
    total_adjectives = len(adjectives)
    return total_adjectives

def identifying_vocabulary_verbs(tagged_tokens):
    verbs = [word for word, pos in tagged_tokens if pos.startswith('VB')]
    total_verbs = len(verbs)
    return total_verbs

def identifying_vocabulary_nouns(tagged_tokens):
    nouns = [word for word, pos in tagged_tokens if pos.startswith('NN')]
    total_nouns = len(nouns)
    return total_nouns

def calculate_cumulative_vocabulary(tokens):
    """
    Calculate cumulative vocabulary size and cumulative count of modal verbs as tokens accumulate.
    """
    vocab_set = set()
    cumulative_vocab_sizes = [] 
    cumulative_tokens = []
    cumulative_modal_count = []
    filtered_tokens = [token for token in tokens if token.isalpha()]
    
    modal_count = 0
    
    for i, token in enumerate(filtered_tokens, start=1):
        if token in modal_verbs:
            modal_count += 1
        cumulative_modal_count.append(modal_count)
        
        vocab_set.add(token)
        
        cumulative_tokens.append(i)
        cumulative_vocab_sizes.append(len(vocab_set))
    
    return cumulative_tokens, cumulative_vocab_sizes, cumulative_modal_count

def remove_stopwords(tokens):
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in Stopwords]
    return filtered_tokens
