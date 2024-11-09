import re
import string
import pandas as pd
import nltk
import numpy as np
import ternary
import matplotlib.pyplot as plt
from collections import Counter
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import t
from empath import Empath
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import vocabulary_analyzer as v

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('omw-1.4')

def load_text():
    with open(f"../data/raw/dolls_house_play.txt.", "r", encoding='utf-8') as f:
        text = f.read()
    return text

def save_modified_text(modified_text):
    with open("../data/modified/modified_play.txt", 'w', encoding='utf-8') as f:
        f.write(modified_text)

def parse_sentences(corpus):
    lines = corpus.strip().split('\n')
    return lines

def parse_data_frames(corpus):
    """This function gathers the text associated with each character (dialogue), 
    excluding the short text under square brackets concerning the state of the target, 
    as a single dataframe with standardized character names."""
    
    dialogues = []
    
    modified_text = re.sub(r'\_*\[.*?\]\_*', "", corpus, flags=re.DOTALL) # Remove all stage directions
    cleaned_text = re.sub(r'^\.\s*', '', modified_text, flags=re.MULTILINE) # remove unnecessary dots
    cleaned_text = re.sub(r'(?<=\S) {2,}(?=\S)', ' ', cleaned_text) # remove double spaces
    
    # Split text into lines
    lines = parse_sentences(cleaned_text)
    
    # Pattern to detect character names as the start of a line
    speech_prefix_pattern = re.compile(r'^([A-Z]+(?: [A-Z]+)?)\.', re.MULTILINE)
    
    character_map = {
        'DOCTOR RANK': 'RANK',
        'CHILDREN': 'THE CHILDREN'
    }
    
    current_character = None
    current_dialogue = ""
    
    for line in lines:
        if "*** END OF THE PROJECT GUTENBERG EBOOK A DOLL'S HOUSE : A PLAY ***" in line:
            break  # Stop there (because there is no dialogue after this)
        
        if speech_prefix_pattern.match(line):
            if current_character:
                dialogues.append({'Character': current_character, 'Dialogue': current_dialogue.strip()})
            
            # Extract character name and dialogue
            current_character = speech_prefix_pattern.match(line).group(1).strip()
            current_character = character_map.get(current_character, current_character)
            dialogue = speech_prefix_pattern.sub('', line).strip()
            current_dialogue = dialogue  # Start new dialogue
        elif current_character:
            # Compine dialogue for the current character
            current_dialogue += " " + line.strip()
    
    if current_character:
        dialogues.append({'Character': current_character, 'Dialogue': current_dialogue.strip()})
    
    df = pd.DataFrame(dialogues)
    return df

def calculate_characters_occurrence(corpus):
    """This function records the number of occurrences of each character in the whole corpus."""

    # DRAMATIS PERSONAE
    characters_alternatives = [
        "Helmer", # Torvald Helmer
        "Torvald",
        "Torvald Helmer",


        "Nora", # Nora Helmer
        "Nora Helmer",
        "Mrs Helmer",

        "Rank", # Doctor Rank

        "Linde", # Christine Linde
        "Christine",
        "Christine Linde",

        "Krogstad", # Nils Krogstad
        "Nils", 
        "Nils Krogstad",

        "Nurse", 
        "Anne",

        "Maid",
        "Helen",

        "Porter",
        "Children"
    ]


    character_counts = {character: 0 for character in characters_alternatives}

    for character in characters_alternatives:
        # Create a pattern to match the character's name, including possessive form
        pattern = r'\b' + re.escape(character) + r'\b|\b' + re.escape(character) + r'\'s\b'
        count = len(re.findall(pattern, corpus, re.IGNORECASE))  # Find all matches

        character_counts[character] = count

    character_counts["Helmer"] += character_counts["Torvald"] - character_counts["Torvald Helmer"] - character_counts["Mrs Helmer"] - character_counts["Nora Helmer"] 
    character_counts["Nora"] += character_counts["Mrs Helmer"]
    character_counts["Linde"] += character_counts["Christine"] - character_counts["Christine Linde"]
    character_counts["Krogstad"] += character_counts["Nils"] - character_counts["Nils Krogstad"]
    character_counts["Nurse"] += character_counts["Anne"]
    character_counts["Maid"] += character_counts["Helen"]

    del character_counts["Torvald"]
    del character_counts["Torvald Helmer"]
    del character_counts["Nora Helmer"]
    del character_counts["Nils"]
    del character_counts["Christine"]
    del character_counts["Christine Linde"]
    del character_counts["Nils Krogstad"]
    del character_counts["Anne"]
    del character_counts["Helen"]
    del character_counts["Mrs Helmer"]

    return character_counts


def calculate_citation_count(df):
    """This function records the number of citations of each character in whole corpus. In other words,
    how many times one person mentions another. We assume stage directions do not constitute mentions by other 
    characters, so they should not be counted in the citations metric"""

    # DRAMATIS PERSONAE
    characters_alternatives = [
        "Helmer", # Torvald Helmer
        "Torvald",
        "Torvald Helmer",

        "Nora", # Nora Helmer
        "Nora Helmer",
        "Mrs Helmer",

        "Rank", # Doctor Rank

        "Linde", # Christine Linde
        "Christine",
        "Christine Linde",

        "Krogstad", # Nils Krogstad
        "Nils", 
        "Nils Krogstad",

        "Nurse", 
        "Anne",

        "Maid",
        "Helen",

        "Porter",
        "Children"
    ]


    all_dialogues = df['Dialogue'].to_string(index=False)

    citation_counts = {citation: 0 for citation in characters_alternatives}

    pattern = r'\b(?:' + '|'.join(re.escape(citation) + r'(?:\'s)?' for citation in characters_alternatives) + r')\b'

    matches = re.findall(pattern, all_dialogues, re.IGNORECASE)

    # Initialize counts for each citation
    citation_counts = {citation: 0 for citation in characters_alternatives}

    # Count occurrences of each citation in the matches
    for match in matches:
        # Normalize case
        normalized_match = match.lower()
        for citation in characters_alternatives:
            if normalized_match.startswith(citation.lower()):
                citation_counts[citation] += 1
                break

    citation_counts["Helmer"] += citation_counts["Torvald"] - citation_counts["Torvald Helmer"] - citation_counts["Mrs Helmer"]
    citation_counts["Nora"] += citation_counts["Mrs Helmer"]
    citation_counts["Linde"] += citation_counts["Christine"] - citation_counts["Christine Linde"]
    citation_counts["Krogstad"] += citation_counts["Nils"] - citation_counts["Nils Krogstad"]
    citation_counts["Nurse"] += citation_counts["Anne"]
    citation_counts["Maid"] += citation_counts["Helen"]

    del citation_counts["Torvald"]
    del citation_counts["Torvald Helmer"]
    del citation_counts["Nora Helmer"]
    del citation_counts["Nils"]
    del citation_counts["Christine"]
    del citation_counts["Christine Linde"]
    del citation_counts["Nils Krogstad"]
    del citation_counts["Anne"]
    del citation_counts["Helen"]
    del citation_counts["Mrs Helmer"]

    return citation_counts


def analyze_vocabulary(df_combined):

    summary_data = []

    for i, row in enumerate(df_combined.itertuples(), start=1):
        character = row.Character
        dialogue = row.Dialogue
    
        tokens = word_tokenize(dialogue.lower())
        tagged_tokens = pos_tag(tokens)

        # vocabulary analysis for this character
        vocabulary_set, vocabulary_size = v.identifying_vocabulary_set(tokens)
        total_tokens = v.identifying_vocabulary_tokens(tokens)
        total_adjectives = v.identifying_vocabulary_adjectives(tagged_tokens)
        total_verbs = v.identifying_vocabulary_verbs(tagged_tokens)
        total_nouns = v.identifying_vocabulary_nouns(tagged_tokens)

        summary_data.append({
            'Character': character,
            'Vocabulary Set' : vocabulary_set,
            'Vocabulary Size': vocabulary_size,
            'Total Tokens': total_tokens,
            'Total Adjectives': total_adjectives,
            'Total Verbs': total_verbs,
            'Total Nouns': total_nouns
        })

    return summary_data

def plot_vocabulary_summary(df_summary):
    # Plot the characteristics for each character
    plt.figure(figsize=(12, 8))

    # vocabulary size
    plt.subplot(2, 2, 1)
    plt.bar(df_summary['Character'], df_summary['Vocabulary Size'], color='skyblue')
    plt.title('Vocabulary Size per Character')
    plt.xticks(rotation=45)

    # total adjectives
    plt.subplot(2, 2, 2)
    plt.bar(df_summary['Character'], df_summary['Total Adjectives'], color='salmon')
    plt.title('Total Adjectives per Character')
    plt.xticks(rotation=45)

    # total verbs
    plt.subplot(2, 2, 3)
    plt.bar(df_summary['Character'], df_summary['Total Verbs'], color='lightgreen')
    plt.title('Total Verbs per Character')
    plt.xticks(rotation=45)

    # total nouns
    plt.subplot(2, 2, 4)
    plt.bar(df_summary['Character'], df_summary['Total Nouns'], color='orchid')
    plt.title('Total Nouns per Character')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def plot_character_and_citation_counts(total_occurences, citation_count):

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # total occurences
    characters = list(total_occurences.keys())
    counts = list(total_occurences.values())
    ax[0].bar(characters, counts, color='skyblue')
    ax[0].set_xlabel('Characters')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Character Occurrence Counts')
    ax[0].tick_params(axis='x', rotation=45)

    # citation occurences
    documents = list(citation_count.keys())
    citations = list(citation_count.values())
    ax[1].bar(documents, citations, color='orange')
    ax[1].set_xlabel('Documents')
    ax[1].set_ylabel('Citation Count')
    ax[1].set_title('Citation Counts per Document')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def fit_heaps_law(cumulative_tokens, cumulative_vocab_sizes):
    """
    Fit the vocabulary growth to Heap's law and fit curve.
    """

    # Using Log-transform and linear fitting (log-log scale)
    log_tokens = np.log(cumulative_tokens).reshape(-1, 1)
    log_vocab_sizes = np.log(cumulative_vocab_sizes)
    model = LinearRegression()
    model.fit(log_tokens, log_vocab_sizes)
    
    # Extract Heap's law parameters
    beta = model.coef_[0]
    k = np.exp(model.intercept_)

    vocab_predicted = model.predict(log_tokens)
    
    # R-squared and adjusted R-squared
    r_squared = r2_score(log_vocab_sizes, vocab_predicted)
    n = len(cumulative_tokens)
    if n < 2:
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)
    else:
        adjusted_r_squared = r_squared

    return {
        'k': k,
        'beta': beta,
        'R-squared': r_squared,
        'Adjusted R-squared': adjusted_r_squared
    }

def vocabulary_evolution(df_combined, fitting_results):
    """
    Plot vocabulary size evolution for each character and print Heap's law fitting results.
    """
    plt.figure(figsize=(12, 8))
    
    for i, row in enumerate(df_combined.itertuples(), start=1):
        character = row.Character
        dialogue = row.Dialogue    

        # Calculate cumulative growth
        tokens = word_tokenize(dialogue.lower())
        cumulative_tokens, cumulative_vocab_sizes, cumulative_modal_verbs = v.calculate_cumulative_vocabulary(tokens)
        
        # Plot vocabulary size evolution
        plt.plot(cumulative_tokens, cumulative_vocab_sizes, label=character)
  
        # Fit Heap's law
        fit_results = fit_heaps_law(cumulative_tokens, cumulative_vocab_sizes)
        fitting_results[character] = fit_results
    
    plt.xlabel("Number of Tokens")
    plt.ylabel("Vocabulary Size")
    plt.title("Evolution of Vocabulary Size with Number of Tokens")
    plt.legend()
    plt.show()

    print("Evolution of Vocabulary Size\n")
    for character, results in fitting_results.items():
        print(f"Character: {character}")
        print(f"  Heap's Law coefficient (k): {results['k']:.4f}")
        print(f"  Heap's Law exponent (beta): {results['beta']:.4f}")
        print(f"  R-squared: {results['R-squared']:.4f}")
        print(f"  Adjusted R-squared: {results['Adjusted R-squared']:.4f}\n")


def modal_verbs_evolution(df_combined):
    """
    calculate and plot modal verbs evolution for each character with respect to the number of tokens.
    """
    plt.figure(figsize=(12, 8))
    
    for i, row in enumerate(df_combined.itertuples(), start=1):
        character = row.Character
        dialogue = row.Dialogue    

        tokens = word_tokenize(dialogue.lower())

        cumulative_tokens, cumulative_vocab_sizes, cumulative_modal_verbs = v.calculate_cumulative_vocabulary(tokens)
 
        # Plot original modal verbs evolution and store the line color
        original_line, = plt.plot(cumulative_tokens, cumulative_modal_verbs, label=character, alpha=0.7)
        color = original_line.get_color()  # Retrieve color of the original plot line

        # now fit polynomial curve
        degree = 3
        coeffs = np.polyfit(cumulative_tokens, cumulative_modal_verbs, degree)
        poly_func = np.poly1d(coeffs)
        fit_values = poly_func(cumulative_tokens)
        
        # Plot the fitted polynomial
        plt.plot(cumulative_tokens, fit_values, linestyle='--', color=color, label=f'{character} (fit)')

        # Calculate residuals and standard deviation of the error
        residuals = cumulative_modal_verbs - fit_values
        std_error = np.std(residuals)
        
        # 90% confidence level
        t_value = t.ppf(0.95, len(cumulative_tokens) - degree - 1)
        margin_of_error = t_value * std_error

        # upper and lower bounds for the confidence interval
        lower_bound = fit_values - margin_of_error
        upper_bound = fit_values + margin_of_error
        
        plt.fill_between(cumulative_tokens, lower_bound, upper_bound, color=color, alpha=0.3, label=f'{character} 90% CI')

    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Modal Verbs', fontsize=12)
    plt.title('Modal Verb Usage vs Number of Tokens', fontsize=14)
    plt.legend()
    plt.show()

def calculate_most_frequent_words(df_combined, n = 5):

    character_vectors = {}

    print("Most common words of each character: \n")

    for i, row in enumerate(df_combined.itertuples(), start=1):
        character = row.Character
        dialogue = row.Dialogue    

        tokens = word_tokenize(dialogue.lower())
        stopword_removed_tokens = v.remove_stopwords(tokens)
        token_counts = Counter(stopword_removed_tokens)

        # Identify the most common tokens
        most_common_tokens = [word for word, freq in token_counts.most_common(n)]
       
        print("Character: ", character)
        print(most_common_tokens)
        print()
               
        # Create a vector for the character with the frequency of the most common tokens
        character_vector = [token_counts.get(token, 0) for token in most_common_tokens]
        
        # Store the vector in the dictionary
        character_vectors[character] = character_vector

    return character_vectors

def calculate_words_cosine_similarity(character_vectors):

    characters = list(character_vectors.keys())
    similarity_df = pd.DataFrame(index=characters, columns=characters)
    # Calculate cosine similarity for each pair of characters
    for i, char1 in enumerate(characters):
        for j, char2 in enumerate(characters):
            if i == j:
                similarity_df.loc[char1, char2] = 1.0
            elif pd.isna(similarity_df.loc[char1, char2]):  
                vector1 = character_vectors[char1]
                vector2 = character_vectors[char2]
                if len(vector1) != len(vector2):
                    continue
                similarity = 1 - cosine(vector1, vector2)
                similarity_df.loc[char1, char2] = similarity
                similarity_df.loc[char2, char1] = similarity

    similarity_df = similarity_df.apply(pd.to_numeric)

    return similarity_df

def doc2vec_embedding(df_combined):
    #  tokens + character = tag
    documents = [TaggedDocument(row.Dialogue.split(), [row.Character]) for row in df_combined.itertuples()]
    
    # Doc2Vec model
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=20)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Generate embeddings for each ccharacter
    character_embeddings = {}
    for row in df_combined.itertuples():
        character = row.Character
        dialogue = row.Dialogue
        tokens = dialogue.split()
        character_embeddings[character] = model.infer_vector(tokens)
    
    return character_embeddings

def calculate_cosine_similarity(embeddings):
    characters = list(embeddings.keys())
    similarity_matrix = pd.DataFrame(index=characters, columns=characters)
    
    for char1 in characters:
        for char2 in characters:
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[char1]], [embeddings[char2]])[0][0]
            similarity_matrix.loc[char1, char2] = similarity
    
    return similarity_matrix

def empath_embedding(df_combined):
    lexicon = Empath()
    character_embeddings = {}
    
    # iterate through the dataframe rows and extract Empath category embeddings
    for row in df_combined.itertuples():
        character = row.Character
        dialogue = row.Dialogue 
        embedding = lexicon.analyze(dialogue, normalize=True)
        embedding_vector = np.array(list(embedding.values()))     
        character_embeddings[character] = embedding_vector
    
    return character_embeddings


def doc2vec_similarity(df_combined):
    # Generate embeddings for each character
    character_embeddings = doc2vec_embedding(df_combined)
    # Calculate cosine similarity between embeddings
    similarity_matrix = calculate_cosine_similarity(character_embeddings)
    
    return similarity_matrix

def empath_similarity(df_combined):
    # Generate embeddings for each character using Empath
    character_embeddings = empath_embedding(df_combined)
    # Calculate cosine similarity between embeddings
    similarity_matrix = calculate_cosine_similarity(character_embeddings)
    
    return similarity_matrix


def plot_pos_tag_histogram(character_name, pos_tags):
    # Count the frequency of each PoS tag
    tag_freq = Counter(pos_tags)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(tag_freq.keys(), tag_freq.values(), color='skyblue')
    plt.xlabel('PoS Tags')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of PoS Tags in {character_name}\'s Dialogue')
    plt.xticks(rotation=45)
    plt.show()


def get_pos_tags(df_dialogue):
    all_pos_tags = []
    
    # Iterate through each dialogue entry to extract PoS tags
    for row in df_dialogue.itertuples():
        dialogue = row.Dialogue
        tokens = word_tokenize(dialogue)
        filtered_tokens = [token for token in tokens if token.isalpha()]
        pos_tags = pos_tag(filtered_tokens)
        all_pos_tags.extend(tag for word, tag in pos_tags)

    return all_pos_tags

def plot_pos_analysis(df_combined):
    grouped = df_combined.groupby('Character')
    # Loop through each character and generate histograms and dispersion plots
    for character, df_dialogue in grouped:
        pos_tags = get_pos_tags(df_dialogue)
        plot_pos_tag_histogram(character, pos_tags)
        dispersion_plot(pos_tags)


def dispersion_plot(pos_tags):
    # Find the most frequent PoS tag
    most_common_tag, _ = Counter(pos_tags).most_common(1)[0]
    
    # Identify positions where the most frequent tag occurs
    tag_positions = [i for i, tag in enumerate(pos_tags) if tag == most_common_tag]
    
    # Plot the dispersion for the most frequent tag
    plt.figure(figsize=(10, 6))
    plt.plot(tag_positions, [most_common_tag] * len(tag_positions), '|', color='blue', label=most_common_tag)
    
    plt.xlabel('Word Position')
    plt.ylabel('PoS Tag')
    plt.title(f'Dispersion Plot of Most Frequent PoS Tag: {most_common_tag}')
    plt.legend(loc='upper right')
    plt.show()


def get_sentiment_score(word):
    word = word.lower().strip(string.punctuation)
    
    # each word's synsets in SentiWordNet
    synsets = list(swn.senti_synsets(word))
    
    if not synsets:
        return 0  # Neutral if there is no synset
    
    # Average positive and negative scores
    pos_score = sum(s.pos_score() for s in synsets) / len(synsets)
    neg_score = sum(s.neg_score() for s in synsets) / len(synsets)
    
    # Return the difference
    return pos_score - neg_score

def calculate_dialogue_sentiment(dialogue):
    tokens = word_tokenize(dialogue)
    sentiment_scores = [get_sentiment_score(word) for word in tokens if word.isalpha()]  # Only consider alphabetic words
    if sentiment_scores:
        return sum(sentiment_scores) / len(sentiment_scores)  # Average sentiment score
    return 0  # Return 0 for neutral


def normalize_to_ternary(value):
    point = []
    # Calculate the components of the ternary TODO fix this (there may be an error)
    x = value
    y = (1 - value) / 2
    z = (1 - value) / 2

    point.append((x, y, z))

    return point


def plot_sentiments(df_sentiments):
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    tax.set_title("Sentiment Plot", fontsize=15)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=5, color="blue")

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']

    for idx, row in enumerate(df_sentiments.itertuples()):
        sentiment_score = row.Sentiment
        character_name = row.Character
        points = normalize_to_ternary(sentiment_score)
        
        color = colors[idx % len(colors)]
        
        tax.scatter(points, marker='s', color=color, label=f'{character_name}')
    
    tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=5)

    tax.show()

def compare_emojis(generated_emoji, reference_emoji):
    return generated_emoji == reference_emoji

def calculate_matching_score(texts, emoji_data):
    correct_matches = 0
    total_messages = len(texts)
    
    for text in texts:
        generated_emoji = generate_emoji(text)
        
        reference_emoji = emoji_data.get(text)
        
        if compare_emojis(generated_emoji, reference_emoji):
            correct_matches += 1
    
    # Calculate accuracy
    accuracy = correct_matches / total_messages
    return accuracy

def generate_emoji():
    return 0

def main():

    try:
        corpus = load_text()
    except FileNotFoundError:
        print("Error: The specified file was not found.")
        return None

    df = parse_data_frames(corpus)

    # Task 1: importance of individual characters
    total_occurences = calculate_characters_occurrence(corpus)
    print("Individual Counts:", total_occurences)

    citation_count = calculate_citation_count(df)
    print("Citation count:", citation_count)

    plot_character_and_citation_counts(total_occurences, citation_count)


    # Task 2: identifying characters vocabulary
    df_combined = df.groupby('Character', as_index=False).agg({'Dialogue': ' '.join})

    summary_data = []
    summary_data = analyze_vocabulary(df_combined)

    df_summary = pd.DataFrame(summary_data)
    print("Summary of characters vocabulary: \n", df_summary)
   
    plot_vocabulary_summary(df_summary)
    
    # Task 3: vocabulary evolution and calculate Heap's law fit for each character 

    fitting_results = {}
    vocabulary_evolution(df_combined, fitting_results) 

    # Task 4: tracks the occurrence of modal verbs in each dataframe
    fitting_results2 = {} 
    modal_verbs_evolution(df_combined)

    # Task 5 five most frequent tokens of each character
    character_vectors = calculate_most_frequent_words(df_combined)
    print("a five dimension vector for each character: \n", character_vectors)
    
    words_cosine_similarity = calculate_words_cosine_similarity(character_vectors)

    print("Words cosine similarity: \n",words_cosine_similarity)

    # Task 6  generates embedding vector for each dataframe doc2vec embedding and compare the
    # similarity between the characters in terms of the corresponding cosine
    # similarity score between the corresponding embedding vectors

    similarity_table = doc2vec_similarity(df_combined)
    print("doc2vec_similarity: \n", similarity_table, end="\n")

    # Task 7 same as 6 but empath category-based embedding employed

    similarity_table2 = empath_similarity(df_combined)
    print("empath_similarity: \n", similarity_table2)

    # Task 8  PoS tag analysis

    plot_pos_analysis(df_combined)

    # Task 9 sentimen analysis
    df_combined['Sentiment'] = df_combined['Dialogue'].apply(calculate_dialogue_sentiment)

    print()
    plot_sentiments(df_combined[['Character', 'Sentiment']])
    print("Sentiment analysis: ", df_combined[['Character', 'Sentiment']])

    
    # Task 10 Stage direction analysis with emojis () TODO

    """    emoji_data = {}
    with open('../data/modified/final_emoji_like.txt', 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines or lines that don't contain ' : '
            if ' : ' not in line:
                continue
            
            try:
                emotion, emoji = line.split(' : ', 1)  # Ensure to split only on the first occurrence of ' : '
                emoji_data[emotion.strip()] = emoji.strip()
            except ValueError:
                # Handle any unexpected issues, e.g., lines that still don't split correctly
                print(f"Skipping line: {line.strip()}")

    print(emoji_data)

    emoji_index = generate_emoji()
    print(emoji_index)

    accuracy = calculate_matching_score(texts, emoji_data) """


if __name__ == "__main__":
    main()
