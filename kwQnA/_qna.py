import json
import re
import numpy as np
import spacy

from kwQnA._complex import ComplexFunc
from kwQnA._getentitypair import GetEntity


class QuestionAnswer:
    """An improved Question-Answer system that uses word embeddings for semantic matching with lower thresholds."""

    def __init__(self):
        super(QuestionAnswer, self).__init__()
        self.complex = ComplexFunc()
        
        # Load the medium-sized spaCy model with word vectors
        # Note: You need to download this model: python -m spacy download en_core_web_md
        try:
            self.nlp = spacy.load('en_core_web_md')
            self.has_vectors = True
            print("Using spaCy model with word vectors for semantic matching")
        except:
            # Fall back to the smaller model if the medium one isn't available
            print("Warning: en_core_web_md model not found. Using en_core_web_sm instead.")
            print("For better semantic matching, install the medium model: python -m spacy download en_core_web_md")
            self.nlp = spacy.load('en_core_web_sm')
            self.has_vectors = False
        
        # Similarity thresholds - LOWER THRESHOLDS for better matching
        self.VERB_SIMILARITY_THRESHOLD = 0.55  # Reduced threshold for considering verbs similar
        self.OBJECT_SIMILARITY_THRESHOLD = 0.65  # Threshold for object matching
        self.SUBJECT_SIMILARITY_THRESHOLD = 0.65  # Threshold for subject matching
        self.PLACE_SIMILARITY_THRESHOLD = 0.60  # Threshold for place matching
        
        # Dictionary to cache similarity scores for improved performance
        self.similarity_cache = {}
        
        # Map for common verb pairs that should always match (even if vectors don't catch them)
        self.guaranteed_matches = {
            'find': ['discover', 'locate', 'uncover', 'detect', 'identify'],
            'discover': ['find', 'locate', 'uncover', 'detect', 'identify'],
            'located': ['situated', 'found', 'positioned', 'placed'],
            'situated': ['located', 'found', 'positioned', 'placed'],
            'develop': ['create', 'build', 'design', 'formulate', 'construct'],
            'invent': ['create', 'design', 'develop', 'originate', 'conceive']
        }
        # Make the guaranteed matches bidirectional
        self._make_guaranteed_matches_bidirectional()
        
    def _make_guaranteed_matches_bidirectional(self):
        """Ensure guaranteed matches are bidirectional."""
        new_matches = {}
        for word, synonyms in self.guaranteed_matches.items():
            for synonym in synonyms:
                if synonym not in self.guaranteed_matches:
                    new_matches[synonym] = [word]
                elif word not in self.guaranteed_matches[synonym]:
                    new_matches[synonym] = self.guaranteed_matches[synonym] + [word]
        
        # Update the dictionary with new bidirectional entries
        for word, synonyms in new_matches.items():
            if word in self.guaranteed_matches:
                self.guaranteed_matches[word].extend(synonyms)
            else:
                self.guaranteed_matches[word] = synonyms

    # Simple plural to singular conversion for common English plural forms
    def simple_singular(self, word):
        """Simple rule-based conversion from plural to singular"""
        word = word.lower().strip()
        
        # Skip if already singular or too short
        if len(word) <= 3:
            return word
            
        # Common plural endings
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'  # flies -> fly
        elif word.endswith('es') and len(word) > 3:
            if word[-3] in 'sxzo' or word[-4:-2] == 'ch' or word[-4:-2] == 'sh':
                return word[:-2]  # boxes -> box, glasses -> glass
            else:
                return word[:-1]  # notes -> note
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
            return word[:-1]  # dogs -> dog, but not pass -> pas
            
        # Return original if no rule applies
        return word

    def get_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts using word embeddings"""
        # For empty texts or exact matches, handle specially
        if not text1 or not text2:
            return 0
        
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Check cache for this pair
        cache_key = f"{text1}|{text2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Check for guaranteed matches for single words
        if ' ' not in text1 and ' ' not in text2:
            if text1 in self.guaranteed_matches and text2 in self.guaranteed_matches[text1]:
                self.similarity_cache[cache_key] = 0.9  # High similarity for guaranteed matches
                return 0.9
        
        # Process the texts with spaCy
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # If either document is empty (after processing), return 0
        if len(doc1) == 0 or len(doc2) == 0:
            self.similarity_cache[cache_key] = 0
            return 0
            
        # Return the semantic similarity (0-1 scale)
        if self.has_vectors:
            try:
                # Try to use the built-in similarity method
                similarity = doc1.similarity(doc2)
                self.similarity_cache[cache_key] = similarity
                return similarity
            except:
                # If similarity fails, fall back to word overlap
                similarity = self._fallback_similarity(text1, text2)
                self.similarity_cache[cache_key] = similarity
                return similarity
        else:
            # No vectors available, use fallback method
            similarity = self._fallback_similarity(text1, text2)
            self.similarity_cache[cache_key] = similarity
            return similarity
    
    def _fallback_similarity(self, text1, text2):
        """Fallback similarity calculation when word vectors aren't available"""
        # Word overlap method
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        overlap_score = overlap / total if total > 0 else 0
        
        # Check for guaranteed matches
        if overlap_score < 0.5:  # Only check guaranteed matches if overlap is low
            for word1 in words1:
                for word2 in words2:
                    # Check if words are in guaranteed matches
                    if word1 in self.guaranteed_matches and word2 in self.guaranteed_matches[word1]:
                        overlap_score += 0.4  # Higher boost score for guaranteed match
        
        return min(overlap_score, 1.0)  # Cap at 1.0

    def are_relations_similar(self, relation1, relation2):
        """Check if two relations are semantically similar using word embeddings."""
        # Direct match
        if relation1 == relation2:
            return True, 1.0
            
        # Contains match
        if relation1 in relation2 or relation2 in relation1:
            return True, 0.9
        
        # Use semantic similarity with embeddings
        similarity = self.get_semantic_similarity(relation1, relation2)
        if similarity >= self.VERB_SIMILARITY_THRESHOLD:
            return True, similarity
            
        # As a fallback, check lemmatized forms
        rel1_lemmas = [token.lemma_ for token in self.nlp(relation1)]
        rel2_lemmas = [token.lemma_ for token in self.nlp(relation2)]
        
        # Check for overlap in lemmatized tokens
        common_lemmas = set(rel1_lemmas).intersection(set(rel2_lemmas))
        if common_lemmas:
            return True, 0.8
            
        # Final check: verb-level similarity
        # Extract all verbs from both relations
        rel1_verbs = [token.lemma_ for token in self.nlp(relation1) if token.pos_ == 'VERB']
        rel2_verbs = [token.lemma_ for token in self.nlp(relation2) if token.pos_ == 'VERB']
        
        # Check similarity between each pair of verbs
        max_verb_sim = 0
        for v1 in rel1_verbs:
            for v2 in rel2_verbs:
                verb_sim = self.get_semantic_similarity(v1, v2)
                max_verb_sim = max(max_verb_sim, verb_sim)
                if verb_sim >= self.VERB_SIMILARITY_THRESHOLD:
                    return True, verb_sim
        
        return False, max_verb_sim

    def extract_verbs_from_text(self, text):
        """Extract all verbs from text for semantic matching."""
        # Process with spaCy
        doc = self.nlp(text.lower())
        
        # Extract all verb lemmas
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        return verbs

    def debug_print_similarity(self, word1, word2):
        """Print similarity between two words for debugging."""
        similarity = self.get_semantic_similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
        if word1 in self.guaranteed_matches and word2 in self.guaranteed_matches[word1]:
            print(f"  ** Guaranteed match from dictionary")
        return similarity

    def findanswer(self, question, c):
        """Find answer to the given question using the knowledge graph and semantic matching."""
        p = self.complex.question_pairs(question)

        if p is None or p == []:
            return "Not Applicable"

        pair = p[0]
        
        # Load the knowledge graph database
        try:
            with open("extra/database.json", "r", encoding="utf8") as f:
                listData = f.readlines()
                loaded = json.loads(listData[0])
        except Exception as e:
            print(f"Error loading database: {e}")
            return "Database Error"

        # Process the relation from the question
        relationQ = self.nlp(pair[1])
        relQ = []
        for i in relationQ:
            relationQ = i.lemma_
            relQ.append(relationQ)
        relationQ = " ".join(relQ).lower()
        
        # Get question components
        subjectQ = str(pair[0]).lower()
        objectQ = str(pair[3]).lower()
        timeQ = str(pair[4]).lower()
        placeQ = str(pair[5]).lower()
        
        # Extract verbs from the question for additional matching
        question_verbs = self.extract_verbs_from_text(question)
        
        # Print debug info about how relation verbs are matched
        if question_verbs:
            print(f"Question verbs: {question_verbs}")
        
        # For storing potential matches
        subList = []
        
        # WHO questions
        if pair[0] in ('who'):
            # Track best match and score
            best_match = None
            best_score = 0
            
            for i in loaded:
                # Get data from database
                relationS = loaded[str(i)]["relation"].lower()
                objectS = loaded[str(i)]["target"].lower()
                timeS = loaded[str(i)]["time"].lower()
                placeS = loaded[str(i)]["place"].lower()
                sourceS = loaded[str(i)]["source"].lower()
                
                # Debug output for verbs
                source_verbs = self.extract_verbs_from_text(relationS)
                if source_verbs and question_verbs:
                    print(f"Comparing question verbs {question_verbs} with source verbs {source_verbs}")
                    for qv in question_verbs:
                        for sv in source_verbs:
                            sim = self.debug_print_similarity(qv, sv)
                
                # Clean up data
                objectS = re.sub('-', ' ', objectS)
                objectQ = re.sub('-', ' ', objectQ)
                
                # Using simple singular conversion
                objectS_singular = self.simple_singular(objectS)
                objectQ_singular = self.simple_singular(objectQ)
                
                # Calculate object similarity score
                object_similarity = self.get_semantic_similarity(objectS, objectQ)
                object_match = (
                    objectS == objectQ or 
                    objectS in objectQ or 
                    objectQ in objectS or
                    objectS_singular == objectQ_singular or 
                    objectS_singular in objectQ_singular or
                    objectQ_singular in objectS_singular or
                    object_similarity >= self.OBJECT_SIMILARITY_THRESHOLD
                )
                
                # Check if relations are similar using embedding-based matching
                relation_match, similarity = self.are_relations_similar(relationS, relationQ)
                
                # If relation match failed, try verb-level matching
                if not relation_match and question_verbs:
                    source_verbs = self.extract_verbs_from_text(relationS)
                    
                    for qverb in question_verbs:
                        for sverb in source_verbs:
                            verb_sim = self.get_semantic_similarity(qverb, sverb)
                            print(f"Verb similarity between '{qverb}' and '{sverb}': {verb_sim:.4f}")
                            if verb_sim >= self.VERB_SIMILARITY_THRESHOLD:
                                relation_match = True
                                similarity = verb_sim
                                print(f"✓ Verb match found: '{qverb}' ~ '{sverb}' ({verb_sim:.4f})")
                                break
                        if relation_match:
                            break
                
                if relation_match:
                    print(f"Relation match with score {similarity:.4f}")
                    # Calculate match score (0-3)
                    score = 0
                    score += similarity * 2  # Weighted by similarity
                    
                    # Object match (most important)
                    if object_match:
                        score += 2
                        print(f"Object match: {objectQ} ~ {objectS}")
                    elif object_similarity > 0.5:
                        score += object_similarity * 2
                        print(f"Partial object match: {objectQ} ~ {objectS} ({object_similarity:.4f})")
                    
                    # Time match if specified
                    if timeQ and (timeQ in timeS or timeS in timeQ):
                        score += 1
                        print(f"Time match: {timeQ} ~ {timeS}")
                    
                    # Place match if specified
                    if placeQ and (placeQ in placeS or placeS in placeQ):
                        score += 1
                        print(f"Place match: {placeQ} ~ {placeS}")
                    
                    print(f"Total score: {score:.4f}")
                    
                    # If scored higher than previous matches
                    if score > best_score:
                        best_score = score
                        best_match = sourceS
                        print(f"New best match: {sourceS} with score {score:.4f}")
                        
                    # If this is a complete match with high score
                    if score >= 2:
                        subList.append(sourceS)
                        print(f"Added to matches: {sourceS}")
            
            # If we found good matches through scoring
            if subList:
                answer_subj = ",".join(subList)
            # Otherwise use the best match we found
            elif best_match and best_score > 1:
                answer_subj = best_match
            else:
                return "None"
                
            return answer_subj

        # WHAT questions
        elif pair[3] in ['what']:
            subjectQ = pair[0]
            
            # Get all targets for the subject + relation
            for i in loaded:
                subjectS = loaded[str(i)]["source"].lower()
                
                # Calculate subject similarity
                subject_similarity = self.get_semantic_similarity(subjectQ, subjectS)
                
                # Check if subject matches
                subject_match = (
                    subjectQ == subjectS or 
                    subjectQ in subjectS or 
                    subjectS in subjectQ or 
                    subject_similarity >= self.SUBJECT_SIMILARITY_THRESHOLD
                )
                    
                if subject_match:
                    # Check relation match with semantic similarity
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    relation_match, similarity = self.are_relations_similar(relationS, relationQ)
                    
                    # If relation match failed, try verb-level matching
                    if not relation_match and question_verbs:
                        source_verbs = self.extract_verbs_from_text(relationS)
                        
                        for qverb in question_verbs:
                            for sverb in source_verbs:
                                verb_sim = self.get_semantic_similarity(qverb, sverb)
                                if verb_sim >= self.VERB_SIMILARITY_THRESHOLD:
                                    relation_match = True
                                    similarity = verb_sim
                                    break
                            if relation_match:
                                break
                    
                    if relation_match:
                        # Check place and time if specified
                        place_match = True
                        if str(pair[5]) != "":
                            placeS = loaded[str(i)]["place"].lower()
                            place_similarity = self.get_semantic_similarity(placeQ, placeS)
                            if not (placeQ in placeS or placeS in placeQ or place_similarity >= self.PLACE_SIMILARITY_THRESHOLD):
                                place_match = False
                                
                        time_match = True
                        if str(pair[4]) != "":
                            timeS = loaded[str(i)]["time"].lower()
                            if not (timeQ in timeS or timeS in timeQ):
                                time_match = False
                                
                        if place_match and time_match:
                            answer_obj = loaded[str(i)]["target"]
                            subList.append(answer_obj)
            
            if subList:
                answer_obj = ",".join(subList)
                return answer_obj
            return "None"

        # WHEN questions
        elif pair[4] in ['when']:
            subjectQ = pair[0]
            
            for i in loaded:
                subjectS = loaded[str(i)]["source"].lower()
                
                # Calculate subject similarity
                subject_similarity = self.get_semantic_similarity(subjectQ, subjectS)
                
                # Check if subject matches using semantic similarity
                subject_match = (
                    subjectQ == subjectS or 
                    subjectQ in subjectS or 
                    subjectS in subjectQ or 
                    subject_similarity >= self.SUBJECT_SIMILARITY_THRESHOLD
                )
                
                if subject_match:
                    # Check relation match with semantic similarity
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    relation_match, similarity = self.are_relations_similar(relationS, relationQ)
                    
                    # If relation match failed, try verb-level matching
                    if not relation_match and question_verbs:
                        source_verbs = self.extract_verbs_from_text(relationS)
                        
                        for qverb in question_verbs:
                            for sverb in source_verbs:
                                verb_sim = self.get_semantic_similarity(qverb, sverb)
                                if verb_sim >= self.VERB_SIMILARITY_THRESHOLD:
                                    relation_match = True
                                    similarity = verb_sim
                                    break
                            if relation_match:
                                break
                    
                    if relation_match:
                        # Check additional constraints
                        if str(pair[5]) != "":
                            placeS = loaded[str(i)]["place"].lower()
                            place_similarity = self.get_semantic_similarity(placeQ, placeS)
                            if placeQ in placeS or placeS in placeQ or place_similarity >= self.PLACE_SIMILARITY_THRESHOLD:
                                # Return time if it exists
                                if loaded[str(i)]["time"] != '':
                                    return loaded[str(i)]["time"]
                        else:
                            # Return time if it exists
                            if loaded[str(i)]["time"] != '':
                                return loaded[str(i)]["time"]
                                
            return "None"

        # WHERE questions
        elif pair[5] in ['where']:
            subjectQ = pair[0]
            
            for i in loaded:
                subjectS = loaded[str(i)]["source"].lower()
                
                # Calculate subject similarity
                subject_similarity = self.get_semantic_similarity(subjectQ, subjectS)
                
                # Check if subject matches using semantic similarity
                subject_match = (
                    subjectQ == subjectS or 
                    subjectQ in subjectS or 
                    subjectS in subjectQ or 
                    subject_similarity >= self.SUBJECT_SIMILARITY_THRESHOLD
                )
                
                if subject_match:
                    # Check relation match with semantic similarity
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    relation_match, similarity = self.are_relations_similar(relationS, relationQ)
                    
                    # If relation match failed, try verb-level matching
                    if not relation_match and question_verbs:
                        source_verbs = self.extract_verbs_from_text(relationS)
                        
                        for qverb in question_verbs:
                            for sverb in source_verbs:
                                verb_sim = self.get_semantic_similarity(qverb, sverb)
                                if verb_sim >= self.VERB_SIMILARITY_THRESHOLD:
                                    relation_match = True
                                    similarity = verb_sim
                                    break
                            if relation_match:
                                break
                    
                    if relation_match:
                        # Check time constraint if specified
                        if str(pair[4]) != "":
                            timeS = loaded[str(i)]["time"].lower()
                            if timeQ in timeS or timeS in timeQ:
                                # Return place if it exists
                                if loaded[str(i)]["place"] != '' and loaded[str(i)]["place"] != ' ':
                                    return loaded[str(i)]["place"]
                        else:
                            # Return place if it exists
                            if loaded[str(i)]["place"] != '' and loaded[str(i)]["place"] != ' ':
                                return loaded[str(i)]["place"]
                                
            return "None"