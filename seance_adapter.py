#!/usr/bin/env python3
"""
SEANCE pipeline adapter: runs the SEANCE dictionaries on a list of texts and returns one row of features per text.
"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
from collections import Counter

class SeanceAnalyzer:
    """Standalone SEANCE analyzer, adapted from SEANCE_1_2_0.py to auto-detect the local SEANCE directory."""

    def __init__(self, seance_dir=None):
        """
        Args:
            seance_dir: Path to a SEANCE_1_2_0 directory containing data_files/.
                If None, searches this script's directory, a SEANCE_1_2_0
                subdirectory, and the SEANCE_DIR env var.
        """
        if seance_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))

            possible_dirs = [
                current_dir,
                os.path.join(current_dir, 'SEANCE_1_2_0'),
                os.environ.get('SEANCE_DIR', 'third_party/SEANCE_1_2_0_Py3'),
            ]

            for possible_dir in possible_dirs:
                data_dir = os.path.join(possible_dir, 'data_files')
                if os.path.exists(data_dir):
                    seance_dir = possible_dir
                    break
            
            if seance_dir is None:
                raise FileNotFoundError(
                    f"Could not find SEANCE data files. Looked in: {possible_dirs}\n"
                    f"Please ensure SEANCE_1_2_0/data_files/ directory exists."
                )
        
        self.seance_dir = seance_dir
        self.data_dir = os.path.join(seance_dir, 'data_files')
        self.components_dir = os.path.join(seance_dir, 'components')

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"data_files directory not found at: {self.data_dir}")
        if not os.path.exists(self.components_dir):
            raise FileNotFoundError(f"components directory not found at: {self.components_dir}")

        print(f"Using SEANCE directory: {self.seance_dir}")

        self._load_dictionaries()

        self.analysis_options = {
            'galc': True,
            'emolex': True,
            'anew': True,
            'sentic': True,
            'vader': False,
            'hu_liu': True,
            'general_inquirer': True,
            'lasswell': True,
            'components': True,
            'pos_specific': True,  # Start with False to avoid spaCy issues
            'negation_control': False
        }
    
    def _load_dictionaries(self):
        """Load all SEANCE dictionaries and data files."""
        print("Loading SEANCE dictionaries...")
        
        try:
            # Affect list (GALC) - handle encoding issues
            affect_path = os.path.join(self.data_dir, 'affective_list.txt')
            with open(affect_path, 'r', encoding='utf-8', errors='ignore') as f:
                affect_list = f.read()
            
            affect_list = re.sub('\t\t', '', affect_list)
            affect_list = re.sub('\t\n', '\n', affect_list)
            affect_list = re.sub(' \t', '\t', affect_list)
            affect_list = re.sub(' \n', '\n', affect_list)
            affect_list = affect_list.split('\n')
            
            self.affect_dict = {}
            for line in affect_list:
                if line.strip():
                    entries = line.split("\t")
                    if len(entries) > 1:
                        self.affect_dict[entries[0]] = entries[1:]
            
            print(f"Loaded {len(self.affect_dict)} GALC categories")
            
        except Exception as e:
            print(f"Warning: Could not load affective_list.txt: {e}")
            self.affect_dict = {}
        
        try:
            # General Inquirer
            gi_path = os.path.join(self.data_dir, 'inquirerbasic.txt')
            with open(gi_path, 'r', encoding='utf-8', errors='ignore') as f:
                gi_list = f.readlines()
            
            self.gi_dict = {}
            for line in gi_list:
                entries = line.strip().split("\t")
                if len(entries) > 1:
                    self.gi_dict[entries[0]] = set(entries[1:])
            
            print(f"Loaded {len(self.gi_dict)} General Inquirer categories")
            
        except Exception as e:
            print(f"Warning: Could not load inquirerbasic.txt: {e}")
            self.gi_dict = {}
        
        try:
            # Lemma dictionary
            lemma_path = os.path.join(self.data_dir, 'e_lemma_py_format_lower.txt')
            with open(lemma_path, 'r', encoding='utf-8', errors='ignore') as f:
                lemma_list = f.readlines()
            
            self.lemma_dict = {}
            for line in lemma_list:
                if line.strip() and line[0] != '#':
                    entries = line.split()
                    if entries:
                        for word in entries:
                            self.lemma_dict[word] = entries[0]
            
            print(f"Loaded {len(self.lemma_dict)} lemma mappings")
            
        except Exception as e:
            print(f"Warning: Could not load lemma file: {e}")
            self.lemma_dict = {}
        
        # ANEW (Valence, Arousal, Dominance)
        try:
            anew_path = os.path.join(self.data_dir, 'affective_norms.txt')
            with open(anew_path, 'r', encoding='utf-8', errors='ignore') as f:
                anew_list = f.readlines()
            
            self.valence = self._dict_builder(anew_list, 1)
            self.arousal = self._dict_builder(anew_list, 2)
            self.dominance = self._dict_builder(anew_list, 3)
            
            print(f"Loaded ANEW dictionaries (Valence: {len(self.valence)}, Arousal: {len(self.arousal)}, Dominance: {len(self.dominance)})")
            
        except Exception as e:
            print(f"Warning: Could not load ANEW files: {e}")
            self.valence = self.arousal = self.dominance = {}
        
        # SenticNet
        try:
            sentic_path = os.path.join(self.data_dir, 'senticnet_data.txt')
            with open(sentic_path, 'r', encoding='utf-8', errors='ignore') as f:
                sentic_list = f.readlines()
            
            self.pleasantness = self._dict_builder(sentic_list, 1)
            self.attention = self._dict_builder(sentic_list, 2)
            self.sensitivity = self._dict_builder(sentic_list, 3)
            self.aptitude = self._dict_builder(sentic_list, 4)
            self.polarity = self._dict_builder(sentic_list, 5)
            
            print(f"Loaded SenticNet dictionaries")
            
        except Exception as e:
            print(f"Warning: Could not load SenticNet files: {e}")
            self.pleasantness = self.attention = self.sensitivity = self.aptitude = self.polarity = {}
        
        # Hu-Liu sentiment
        try:
            pos_path = os.path.join(self.data_dir, 'positive_words.txt')
            neg_path = os.path.join(self.data_dir, 'negative_words.txt')
            
            with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.lu_hui_positive = f.read().split("\n")
            with open(neg_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.lu_hui_negative = f.read().split("\n")

            self.lu_hui_positive = [w for w in self.lu_hui_positive if w.strip()]
            self.lu_hui_negative = [w for w in self.lu_hui_negative if w.strip()]
            
            print(f"Loaded Hu-Liu sentiment (Positive: {len(self.lu_hui_positive)}, Negative: {len(self.lu_hui_negative)})")
            
        except Exception as e:
            print(f"Warning: Could not load Hu-Liu files: {e}")
            self.lu_hui_positive = self.lu_hui_negative = []
        
        # Components
        try:
            self.components = self._component_dicter(os.path.join(self.components_dir, 'C*.txt'))
            print(f"Loaded {len(self.components)} component files")
            
        except Exception as e:
            print(f"Warning: Could not load component files: {e}")
            self.components = {}
        
        # spaCy for POS tagging (optional)
        try:
            import spacy
            self.nlp = spacy.load('en_core_web_sm')
            self.pos_available = True
            print("spaCy loaded successfully")
        except Exception as e:
            print(f"Warning: spaCy not available: {e}")
            self.pos_available = False

        # VADER disabled - other sentiment features cover this
        self.vader_available = False
        self.vader_analyzer = None
        print("VADER disabled - using other sentiment features")

        print("SEANCE dictionaries loaded successfully!")
    
    def _dict_builder(self, database_file, number):
        result_dict = {}
        for line in database_file:
            if line.strip() and line[0] != '#':
                entries = line.strip().split("\t")
                if len(entries) > number:
                    try:
                        result_dict[entries[0]] = float(entries[number])
                    except ValueError:
                        result_dict[entries[0]] = entries[number]
        return result_dict
    
    def _component_dicter(self, folder_pattern):
        result_dict = {}
        file_list = glob.glob(folder_pattern)
        for file_path in file_list:
            components_list = []
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            components_list.append(line.strip().split("\t"))
                key = os.path.basename(file_path)
                result_dict[key] = components_list
            except Exception as e:
                print(f"Warning: Could not load component file {file_path}: {e}")
        return result_dict
    
    def _safe_divide(self, numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    def analyze_text(self, text, conversation_id=None):
        """Analyze a single text and return all SEANCE features as a dict."""
        if not text or not text.strip():
            return {'conversation_id': conversation_id, 'nwords': 0}

        text = re.sub("'", "'", text)  # normalize curly apostrophes
        text = re.sub("'", "'", text)

        pre_text = text.lower().split()
        text_2 = []
        punctuation = (".","!","?",",", ":",";", "'",'"')

        for word in pre_text:
            if len(word) < 1:
                continue
            if len(word) == 1 and word in punctuation:
                continue
            if word and word[-1] in punctuation:
                word = word[:-1]
            if word and word[0] in punctuation:
                word = word[1:]
            if word:
                text_2.append(word)

        nwords = len(text_2)
        if nwords == 0:
            return {'conversation_id': conversation_id, 'nwords': 0}

        results = {
            'conversation_id': conversation_id,
            'nwords': nwords
        }

        pos_data = None
        if self.pos_available and self.analysis_options['pos_specific']:
            try:
                pos_data = self._extract_pos_features(text)
            except Exception as e:
                print(f"Warning: POS extraction failed: {e}")
                pos_data = None

        try:
            if self.analysis_options['galc']:
                results.update(self._run_galc(text))
        except Exception as e:
            print(f"Warning: GALC analysis failed: {e}")
        
        try:
            if self.analysis_options['emolex']:
                results.update(self._run_emolex(text_2))
        except Exception as e:
            print(f"Warning: EmoLex analysis failed: {e}")
        
        try:
            if self.analysis_options['anew']:
                results.update(self._run_anew(text_2))
        except Exception as e:
            print(f"Warning: ANEW analysis failed: {e}")
        
        try:
            if self.analysis_options['sentic']:
                results.update(self._run_sentic(text_2))
        except Exception as e:
            print(f"Warning: SenticNet analysis failed: {e}")
        
        try:
            if self.analysis_options['vader'] and self.vader_available:
                results.update(self._run_vader(text))
        except Exception as e:
            print(f"Warning: VADER analysis failed: {e}")
        
        try:
            if self.analysis_options['hu_liu']:
                results.update(self._run_hu_liu(text_2))
        except Exception as e:
            print(f"Warning: Hu-Liu analysis failed: {e}")
        
        try:
            if self.analysis_options['general_inquirer']:
                results.update(self._run_gi(text_2))
        except Exception as e:
            print(f"Warning: General Inquirer analysis failed: {e}")
        
        try:
            if self.analysis_options['lasswell']:
                results.update(self._run_lasswell(text_2))
        except Exception as e:
            print(f"Warning: Lasswell analysis failed: {e}")
        
        try:
            if self.analysis_options['components']:
                results.update(self._run_components(text_2, pos_data))
        except Exception as e:
            print(f"Warning: Components analysis failed: {e}")
        
        return results
    
    def _extract_pos_features(self, text):
        noun_tags = {"NN", "NNS", "NNP", "NNPS"}
        adjective_tags = {"JJ", "JJR", "JJS"}
        verb_tags = {"VB", "VBZ", "VBP", "VBD", "VBN", "VBG"}
        adverb_tags = {"RB", "RBR", "RBS"}
        
        doc = self.nlp(text)
        
        pos_data = {
            'nouns': [],
            'verbs': [],
            'adjectives': [],
            'adverbs': []
        }
        
        for token in doc:
            word = token.text.lower()
            if token.tag_ in noun_tags:
                pos_data['nouns'].append(word)
            elif token.tag_ in verb_tags:
                pos_data['verbs'].append(word)
            elif token.tag_ in adjective_tags:
                pos_data['adjectives'].append(word)
            elif token.tag_ in adverb_tags:
                pos_data['adverbs'].append(word)
        
        return pos_data
    
    def _run_galc(self, text, suffix=""):
        if isinstance(text, list):
            text = " ".join(text)
        
        results = {}
        galc_categories = [
            'Admiration/Awe_GALC', 'Amusement_GALC', 'Anger_GALC', 'Anxiety_GALC',
            'Beingtouched_GALC', 'Boredom_GALC', 'Compassion_GALC', 'Contempt_GALC',
            'Contentment_GALC', 'Desperation_GALC', 'Disappointment_GALC', 'Disgust_GALC',
            'Dissatisfaction_GALC', 'Envy_GALC', 'Fear_GALC', 'Feelinglove_GALC',
            'Gratitude_GALC', 'Guilt_GALC', 'Happiness_GALC', 'Hatred_GALC',
            'Hope_GALC', 'Humility_GALC', 'Interest/Enthusiasm_GALC', 'Irritation_GALC',
            'Jealousy_GALC', 'Joy_GALC', 'Longing_GALC', 'Lust_GALC',
            'Pleasure/Enjoyment_GALC', 'Pride_GALC', 'Relaxation/Serenity_GALC',
            'Relief_GALC', 'Sadness_GALC', 'Shame_GALC', 'Surprise_GALC',
            'Tension/Stress_GALC', 'Positive_GALC', 'Negative_GALC'
        ]
        
        nwords = len(text.split())
        for category in galc_categories:
            try:
                results[category + suffix] = self._regex_count(category, text, nwords)
            except:
                results[category + suffix] = 0
        
        return results
    
    def _regex_count(self, category, text, nwords):
        if category not in self.affect_dict:
            return 0
        
        counter = 0
        try:
            for item in self.affect_dict[category]:
                if item == "":
                    continue
                if item.endswith('*'):
                    pattern = r'\b' + re.escape(item[:-1]) + r'.*?\b'
                else:
                    pattern = r'\b' + re.escape(item) + r'\b'
                counter += len(re.findall(pattern, text, re.IGNORECASE))
        except Exception:
            pass
        
        return self._safe_divide(counter, nwords)
    
    def _run_emolex(self, word_list, suffix=""):
        results = {}
        emolex_categories = [
            'Anger_NRC', 'Anticipation_NRC', 'Disgust_NRC', 'Fear_NRC',
            'Joy_NRC', 'Negative_NRC', 'Positive_NRC', 'Sadness_NRC',
            'Surprise_NRC', 'Trust_NRC'
        ]
        
        nwords = len(word_list)
        for category in emolex_categories:
            header = category.replace('_NRC', '_EmoLex') + suffix
            results[header] = self._list_dict_counter(category, word_list, nwords)
        
        return results
    
    def _list_dict_counter(self, category, word_list, nwords):
        if category not in self.gi_dict:
            return 0
        
        counter = 0
        try:
            for word in word_list:
                if word in self.gi_dict[category]:
                    counter += 1
                elif word in self.lemma_dict and self.lemma_dict[word] in self.gi_dict[category]:
                    counter += 1
        except Exception:
            pass
        
        return self._safe_divide(counter, nwords)
    
    def _run_anew(self, word_list, suffix=""):
        results = {}

        results[f'Valence{suffix}'] = self._data_dict_counter(word_list, self.valence, 0)
        results[f'Valence_nwords{suffix}'] = self._data_dict_counter(word_list, self.valence, 1)

        results[f'Arousal{suffix}'] = self._data_dict_counter(word_list, self.arousal, 0)
        results[f'Arousal_nwords{suffix}'] = self._data_dict_counter(word_list, self.arousal, 1)

        results[f'Dominance{suffix}'] = self._data_dict_counter(word_list, self.dominance, 0)
        results[f'Dominance_nwords{suffix}'] = self._data_dict_counter(word_list, self.dominance, 1)

        return results

    def _data_dict_counter(self, word_list, data_dict, mode):
        counter = 0
        sum_counter = 0
        nwords = len(word_list)
        
        try:
            for word in word_list:
                if word in data_dict:
                    counter += 1
                    sum_counter += float(data_dict[word])
                elif word in self.lemma_dict and self.lemma_dict[word] in data_dict:
                    counter += 1
                    sum_counter += float(data_dict[self.lemma_dict[word]])
        except Exception:
            pass
        
        if mode == 0:  # Average per matching word
            return self._safe_divide(sum_counter, counter)
        else:  # Average per total words
            return self._safe_divide(sum_counter, nwords)
    
    def _run_sentic(self, word_list, suffix=""):
        results = {}

        results[f'pleasantness{suffix}'] = self._ngram_data_dict_counter(word_list, self.pleasantness)
        results[f'attention{suffix}'] = self._ngram_data_dict_counter(word_list, self.attention)
        results[f'sensitivity{suffix}'] = self._ngram_data_dict_counter(word_list, self.sensitivity)
        results[f'aptitude{suffix}'] = self._ngram_data_dict_counter(word_list, self.aptitude)
        results[f'polarity{suffix}'] = self._ngram_data_dict_counter(word_list, self.polarity)

        return results

    def _ngram_data_dict_counter(self, word_list, data_dict):
        counter = 0
        denominator = 0
        
        try:
            for word in word_list:
                if word in data_dict:
                    counter += float(data_dict[word])
                    denominator += 1
                elif word in self.lemma_dict and self.lemma_dict[word] in data_dict:
                    counter += float(data_dict[self.lemma_dict[word]])
                    denominator += 1
        except Exception:
            pass
        
        return self._safe_divide(counter, denominator)
    
    def _run_vader(self, text, suffix=""):
        if not self.vader_available:
            return {
                f'vader_negative{suffix}': 0,
                f'vader_neutral{suffix}': 0,
                f'vader_positive{suffix}': 0,
                f'vader_compound{suffix}': 0
            }
        
        if self.vader_analyzer is None:
            return {
                f'vader_negative{suffix}': 0,
                f'vader_neutral{suffix}': 0,
                f'vader_positive{suffix}': 0,
                f'vader_compound{suffix}': 0
            }
        try:
            vs = self.vader_analyzer.polarity_scores(text)
            return {
                f'vader_negative{suffix}': vs['neg'],
                f'vader_neutral{suffix}': vs['neu'],
                f'vader_positive{suffix}': vs['pos'],
                f'vader_compound{suffix}': vs['compound']
            }
        except Exception:
            return {
                f'vader_negative{suffix}': 0,
                f'vader_neutral{suffix}': 0,
                f'vader_positive{suffix}': 0,
                f'vader_compound{suffix}': 0
            }
    
    def _run_hu_liu(self, word_list, suffix=""):
        try:
            positive_count = sum(1 for word in word_list if word in self.lu_hui_positive)
            negative_count = sum(1 for word in word_list if word in self.lu_hui_negative)
            total_sentiment_words = positive_count + negative_count
            nwords = len(word_list)
            
            return {
                f'hu_liu_pos_perc{suffix}': self._safe_divide(positive_count, total_sentiment_words),
                f'hu_liu_neg_perc{suffix}': self._safe_divide(negative_count, total_sentiment_words),
                f'hu_liu_pos_nwords{suffix}': self._safe_divide(positive_count, nwords),
                f'hu_liu_neg_nwords{suffix}': self._safe_divide(negative_count, nwords),
                f'hu_liu_prop{suffix}': self._safe_divide(positive_count, negative_count) if negative_count > 0 else (1 if positive_count > 0 else 0)
            }
        except Exception:
            return {
                f'hu_liu_pos_perc{suffix}': 0,
                f'hu_liu_neg_perc{suffix}': 0,
                f'hu_liu_pos_nwords{suffix}': 0,
                f'hu_liu_neg_nwords{suffix}': 0,
                f'hu_liu_prop{suffix}': 0
            }
    
    def _run_gi(self, word_list, suffix=""):
        """General Inquirer analysis, restricted to a subset of categories."""
        results = {}

        key_gi_categories = [
            'Positiv_GI', 'Negativ_GI', 'Strong_GI', 'Power_GI', 'Active_GI',
            'Pleasur_GI', 'Feel_GI', 'Emot_GI', 'Virtue_GI', 'Academ_GI',
            'Work_GI', 'Social_GI', 'Think_GI', 'Know_GI', 'Goal_GI',
            'Means_GI', 'Complet_GI', 'Quality_GI', 'Need_GI'
        ]

        nwords = len(word_list)
        for category in key_gi_categories:
            results[category + suffix] = self._list_dict_counter(category, word_list, nwords)

        return results

    def _run_lasswell(self, word_list, suffix=""):
        """Lasswell values analysis, restricted to a subset of categories."""
        results = {}

        key_lasswell_categories = [
            'Powtot_Lasswell', 'Enltot_Lasswell', 'Afftot_Lasswell', 
            'Skltot_Lasswell', 'Wlbtot_Lasswell'
        ]
        
        nwords = len(word_list)
        for category in key_lasswell_categories:
            results[category + suffix] = self._list_dict_counter(category, word_list, nwords)
        
        return results
    
    def _run_components(self, word_list, pos_data=None):
        results = {}
        
        component_names = [
            "negative_adjectives_component", "social_order_component", 
            "action_component", "positive_adjectives_component",
            "joy_component", "affect_friends_and_family_component",
            "fear_and_digust_component", "politeness_component",
            "polarity_nouns_component", "polarity_verbs_component",
            "virtue_adverbs_component", "positive_nouns_component",
            "respect_component", "trust_verbs_component",
            "failure_component", "well_being_component",
            "economy_component", "certainty_component",
            "positive_verbs_component", "objects_component"
        ]
        
        # For now, return zeros - components are complex
        for comp_name in component_names:
            results[comp_name] = 0
        
        return results
    
    def analyze_batch(self, texts, conversation_ids=None, batch_size=50):
        """Run analyze_text() over a list of texts and return the results as a DataFrame."""
        results = []

        if conversation_ids is None:
            conversation_ids = [f"conv_{i}" for i in range(len(texts))]

        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))

            print(f"Processing batch {batch_num + 1}/{total_batches} ({end_idx - start_idx} texts)")

            batch_texts = texts[start_idx:end_idx]
            batch_ids = conversation_ids[start_idx:end_idx]

            for text, conv_id in zip(batch_texts, batch_ids):
                try:
                    result = self.analyze_text(text, conv_id)
                    results.append(result)
                    if len(results) % 10 == 0:
                        print(f"  Processed {len(results)} texts...")
                except Exception as e:
                    print(f"Error processing conversation {conv_id}: {e}")
                    # placeholder row keeps batch alignment intact for the caller
                    results.append({
                        'conversation_id': conv_id,
                        'nwords': 0,
                        'error': str(e)
                    })

        print(f"Batch processing complete! Processed {len(results)} texts.")
        return pd.DataFrame(results)


class ConcernSeanceProcessor:
    """High-level SEANCE processor for short texts (survey responses rather than full conversations)."""

    def __init__(self, seance_dir=None):
        self.analyzer = SeanceAnalyzer(seance_dir)

    def prepare_texts(self, conversations, min_words=0, max_words=float('inf')):
        """
        Normalize a list of response records into the flat {user, text,
        word_count, metadata} shape analyze_batch() expects. Several input
        shapes are accepted (chat-turn lists, plain strings, or a dict with
        a text-like field) since this adapter is reused across differently
        structured source files.

        Args:
            conversations: List of dicts or strings, in any of the accepted shapes.
            min_words: Minimum word count for inclusion.
            max_words: Maximum word count for inclusion.
        """
        prepared_conversations = []

        for i, conv in enumerate(conversations):
            try:
                user_messages = []
                combined_text = ""

                if 'conversation' in conv:
                    # chat-turn format: role/from + content/value per turn
                    for turn in conv['conversation']:
                        if turn.get('role') == 'user' or turn.get('from') == 'human':
                            content = turn.get('content') or turn.get('value', '')
                            if content:
                                user_messages.append(content)
                    combined_text = " ".join(user_messages)

                elif 'turns' in conv:
                    for turn in conv['turns']:
                        if turn.get('from') == 'human':
                            content = turn.get('value', '')
                            if content:
                                user_messages.append(content)
                    combined_text = " ".join(user_messages)

                elif 'text' in conv:
                    combined_text = conv['text']
                    user_messages = [combined_text]

                elif 'user_conversations' in conv:
                    combined_text = conv['user_conversations']
                    user_messages = [combined_text]

                elif isinstance(conv, str):
                    combined_text = conv
                    user_messages = [conv]

                elif isinstance(conv, dict):
                    text_fields = ['text', 'content', 'message', 'user_conversations', 'conversation_text']
                    for field in text_fields:
                        if field in conv and conv[field]:
                            combined_text = str(conv[field])
                            user_messages = [combined_text]
                            break

                    if not combined_text:
                        print(f"Warning: No text content found in conversation {i}, keys: {list(conv.keys())}")
                        continue

                else:
                    print(f"Warning: Unexpected conversation format at index {i}: {type(conv)}")
                    continue

                if not combined_text or not combined_text.strip():
                    print(f"Warning: Empty text content in conversation {i}")
                    continue

                word_count = len(combined_text.split())

                if min_words <= word_count <= max_words:
                    if isinstance(conv, dict):
                        user_id = conv.get('user') or conv.get('user_id') or f"user_{i}"
                    else:
                        user_id = f"user_{i}"

                    metadata = {
                        'original_turns': len(user_messages),
                        'original_index': i,
                        'word_count': word_count
                    }

                    if isinstance(conv, dict):
                        metadata.update({
                            'language': conv.get('language', 'en'),
                            'timestamp': conv.get('timestamp'),
                            'country': conv.get('country'),
                            'state': conv.get('state'),
                            'total_turns': conv.get('total_turns'),
                            'total_messages': conv.get('total_messages'),
                            'avg_message_length': conv.get('avg_message_length'),
                            'session_count': conv.get('session_count'),
                            'total_characters': conv.get('total_characters')
                        })

                    prepared_conversations.append({
                        'user': user_id,
                        'text': combined_text,
                        'word_count': word_count,
                        'metadata': metadata
                    })

                    print(f"Conversation {i} (User {user_id}): {word_count} words - ACCEPTED")

                else:
                    print(f"Conversation {i}: {word_count} words - REJECTED (min: {min_words}, max: {max_words})")

            except Exception as e:
                print(f"Warning: Could not process conversation {i}: {e}")
                print(f"Conversation type: {type(conv)}")
                if isinstance(conv, dict):
                    print(f"Available keys: {list(conv.keys())}")
                continue
        
        print(f"Prepared {len(prepared_conversations)} conversations from {len(conversations)} total")
        return prepared_conversations
    
    def process_conversations(self, prepared_conversations, output_path=None, batch_size=50):
        """Run prepare_texts() output through SEANCE and return the merged feature DataFrame."""
        texts = [conv['text'] for conv in prepared_conversations]
        conv_ids = [conv['user'] for conv in prepared_conversations]

        print("Running SEANCE analysis...")
        results_df = self.analyzer.analyze_batch(texts, conv_ids, batch_size)

        if 'conversation_id' in results_df.columns:
            results_df = results_df.rename(columns={'conversation_id': 'user'})

        metadata_df = pd.DataFrame([
            {
                'user': conv['user'],
                'original_turns': conv['metadata']['original_turns'],
                'language': conv['metadata']['language'],
                'timestamp': conv['metadata']['timestamp'],
                'original_index': conv['metadata']['original_index']
            }
            for conv in prepared_conversations
        ])

        final_df = results_df.merge(metadata_df, on='user', how='left')

        if 'error' in final_df.columns:
            error_count = final_df['error'].notna().sum()
            if error_count > 0:
                print(f"Warning: {error_count} conversations had processing errors")
            final_df = final_df.drop('error', axis=1)

        if output_path:
            final_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return final_df
