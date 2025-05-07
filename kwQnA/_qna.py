import json
import re

# import inflect
import spacy

from kwQnA._complex import ComplexFunc
from kwQnA._getentitypair import GetEntity


class QuestionAnswer:
    """docstring for QuestionAnswer."""

    def __init__(self):
        super(QuestionAnswer, self).__init__()
        self.complex = ComplexFunc()
        self.nlp = spacy.load('en_core_web_sm')
        # self.p = inflect.engine()  # Not using inflect anymore

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

    def findanswer(self, question, c):
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
        
        # For lemmatization of objects (singular forms)
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
                
                # Clean up data
                objectS = re.sub('-', ' ', objectS)
                objectQ = re.sub('-', ' ', objectQ)
                
                # Using simple singular conversion instead of inflect
                objectS_singular = self.simple_singular(objectS)
                objectQ_singular = self.simple_singular(objectQ)
                
                # Compare both original and singular forms
                object_match = (objectS == objectQ or objectS in objectQ or objectQ in objectS or
                               objectS_singular == objectQ_singular or objectS_singular in objectQ_singular or
                               objectQ_singular in objectS_singular)
                    
                # First check if relations are similar
                relation_match = False
                
                # Direct relation match
                if relationS == relationQ:
                    relation_match = True
                # Relation contained in each other
                elif relationS in relationQ or relationQ in relationS:
                    relation_match = True
                # Lemmatized forms of relation words
                else:
                    rel_s_tokens = [t.lemma_ for t in self.nlp(relationS)]
                    rel_q_tokens = [t.lemma_ for t in self.nlp(relationQ)]
                    # Check for overlap in lemmatized tokens
                    if any(token in rel_q_tokens for token in rel_s_tokens):
                        relation_match = True
                
                if relation_match:
                    # Calculate match score (0-3)
                    score = 0
                    
                    # Object match (most important)
                    if object_match:
                        score += 2
                    
                    # Time match if specified
                    if timeQ and (timeQ in timeS or timeS in timeQ):
                        score += 1
                    
                    # Place match if specified
                    if placeQ and (placeQ in placeS or placeS in placeQ):
                        score += 1
                    
                    # If scored higher than previous matches
                    if score > best_score:
                        best_score = score
                        best_match = sourceS
                        
                    # If this is a complete match with max score, just use it
                    if score >= 2:
                        subList.append(sourceS)
                    
            # If we found good matches through scoring
            if subList:
                answer_subj = ",".join(subList)
            # Otherwise use the best match we found
            elif best_match:
                answer_subj = best_match
            else:
                return "None"
                
            return answer_subj

        # WHAT questions
        elif pair[3] in ['what']:
            subjectQ = pair[0]
            subList = []
            
            # Get all targets for the subject + relation
            for i in loaded:
                subjectS = loaded[str(i)]["source"].lower()
                
                # Check if subject matches
                subject_match = False
                if subjectQ == subjectS:
                    subject_match = True
                elif subjectQ in subjectS or subjectS in subjectQ:
                    subject_match = True
                    
                if subject_match:
                    # Check relation match
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    
                    relation_match = False
                    if relationS == relationQ:
                        relation_match = True
                    elif relationS in relationQ or relationQ in relationS:
                        relation_match = True
                    # Check key terms in relations
                    else:
                        rel_s_tokens = relationS.split()
                        rel_q_tokens = relationQ.split()
                        if any(token in rel_q_tokens for token in rel_s_tokens):
                            relation_match = True
                    
                    if relation_match:
                        # Check place and time if specified
                        place_match = True
                        if str(pair[5]) != "":
                            placeS = loaded[str(i)]["place"].lower()
                            if not (placeQ in placeS or placeS in placeQ):
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
                
                # Look for the subject
                if subjectQ == subjectS or subjectQ in subjectS or subjectS in subjectQ:
                    # Check relation match
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    
                    relation_match = False
                    if relationS == relationQ:
                        relation_match = True
                    elif relationS in relationQ or relationQ in relationS:
                        relation_match = True
                    else:
                        rel_s_tokens = relationS.split()
                        rel_q_tokens = relationQ.split()
                        if any(token in rel_q_tokens for token in rel_s_tokens):
                            relation_match = True
                            
                    if relation_match:
                        # Check additional constraints
                        if str(pair[5]) != "":
                            placeS = loaded[str(i)]["place"].lower()
                            if placeQ in placeS or placeS in placeQ:
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
                
                # Look for the subject
                if subjectQ == subjectS or subjectQ in subjectS or subjectS in subjectQ:
                    # Check relation match
                    relationS = " ".join([t.lemma_ for t in self.nlp(loaded[str(i)]["relation"])]).lower()
                    
                    relation_match = False
                    if relationS == relationQ:
                        relation_match = True
                    elif relationS in relationQ or relationQ in relationS:
                        relation_match = True
                    else:
                        rel_s_tokens = relationS.split()
                        rel_q_tokens = relationQ.split()
                        if any(token in rel_q_tokens for token in rel_s_tokens):
                            relation_match = True
                            
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