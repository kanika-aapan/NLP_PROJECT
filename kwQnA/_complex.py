import spacy


class ComplexFunc:
    # """docstring for Tenses."""

    def __init__(self):
        self.ent_pairs = list()
        self.nlp = spacy.load('en_core_web_sm')

    def get_time_place_from_sent(self, sentence):
        xdate = []
        xplace = []
        
        # Track person entities to avoid misclassifying them as places
        person_tokens = set()
        for ent in sentence.ents:
            if ent.label_ == 'PERSON':
                for i in range(ent.start, ent.end):
                    person_tokens.add(i)
        
        # First pass - look for standard NER entities
        for i in sentence.ents:
            if i.label_ in ('DATE', 'TIME'):
                xdate.append(str(i))

            if i.label_ in ('GPE', 'LOC', 'FAC'):
                xplace.append(str(i))
        
        # Second pass - look for temporal expressions missed by NER
        if not xdate:
            for token in sentence:
                # Check for years
                if token.text.isdigit() and len(token.text) == 4 and 1000 <= int(token.text) <= 2100:
                    xdate.append(token.text)
                
                # Check for time expressions
                if token.lower_ in ('today', 'yesterday', 'tomorrow', 'century', 'decade'):
                    # Look for modifiers
                    for child in token.children:
                        if child.dep_ in ('amod', 'nummod'):
                            xdate.append(f"{child.text} {token.text}")
                            break
                    else:
                        xdate.append(token.text)
        
        # Handle complex place references
        if not xplace:
            for token in sentence:
                # Skip if token is part of a PERSON entity
                if token.i in person_tokens:
                    continue
                    
                # Look for location indicators
                is_potential_place = False
                
                # Check if it's a proper noun
                if token.pos_ == 'PROPN':
                    is_potential_place = True
                
                # Check for prepositions that often indicate locations
                if token.head.lemma_ in ('in', 'at', 'from', 'to', 'near', 'by'):
                    is_potential_place = True
                
                # Check for classic place adjectives
                for child in token.children:
                    if child.dep_ == 'amod' and child.lemma_ in ('north', 'south', 'east', 'west', 'central', 'urban', 'rural'):
                        is_potential_place = True
                
                if is_potential_place:
                    # Check if it's part of a compound name
                    if any(child.dep_ == 'compound' for child in token.children):
                        place_parts = [child.text for child in token.children if child.dep_ == 'compound' and child.i not in person_tokens]
                        if place_parts:  # Only proceed if we have non-person compounds
                            place_parts.append(token.text)
                            xplace.append(" ".join(place_parts))
                    # Single word place
                    elif token.pos_ == 'PROPN' and token.ent_type_ not in ('PERSON', 'ORG'):
                        xplace.append(token.text)
        
        # Additional check to remove any potential person references that made it through
        filtered_place = []
        subject_text = None
        
        # Find the subject of the sentence
        for token in sentence:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject_tokens = [token.text.lower()]
                # Get compound parts of subject
                for child in token.children:
                    if child.dep_ == 'compound':
                        subject_tokens.append(child.text.lower())
                subject_text = " ".join(subject_tokens)
                break
        
        # Filter out places that match the subject
        for place in xplace:
            # Skip if place name is contained in the subject or vice versa
            if subject_text and (subject_text in place.lower() or place.lower() in subject_text):
                continue
            filtered_place.append(place)
        
        return xdate, filtered_place
    def find_obj(self, sentence, place, time):
            object_list = []
            # Initialize buffer_obj to prevent UnboundLocalError
            buffer_obj = None

            # First pass: Look for direct objects
            for word in sentence:
                if word.dep_ in ('obj', 'dobj', 'pobj'):
                    buffer_obj = word
                    
                    # Avoid places that are objects of prepositions like "of"
                    if str(word) in place and word.nbor(-1).dep_ in ('prep') and str(word.nbor(-1)) == "of":
                        continue
                        
                    # Standard objects (not in time or place)
                    if str(word) not in time and str(word) not in place:
                        # Get the full phrase for compound objects
                        obj_phrase = self._get_full_object_phrase(word)
                        
                        # Check for multi-word phrases with prepositions
                        if any(child.dep_ == 'prep' for child in word.children):
                            for child in word.children:
                                if child.dep_ == 'prep':
                                    # For phrases like "laws of motion"
                                    for grandchild in child.children:
                                        if grandchild.dep_ in ('pobj'):
                                            obj_phrase = f"{obj_phrase} {child} {grandchild}"
                                            
                        object_list.append(obj_phrase)
                            
                    # Handle places as objects when not part of "of" phrases
                    elif str(word) in place and str(word.nbor(-1)) != "of":
                        if not object_list:  # Only add if no other objects
                            object_list.append(str(word))
                            
                    # Handle times as objects when no other objects
                    elif str(word) in time and not object_list:
                        object_list.append(str(word))
            
            # Second pass: If no direct objects found, look for prepositional objects
            if not object_list and not buffer_obj:
                for word in sentence:
                    if word.dep_ == 'prep':
                        for child in word.children:
                            if child.dep_ == 'pobj':
                                # Special handling for time prepositions
                                if word.text.lower() in ('in', 'on', 'at', 'during', 'by'):
                                    # Handle time objects
                                    if str(child) in time:
                                        buffer_obj = child
                                        object_list.append(str(child))
                                    # Handle place objects
                                    elif str(child) in place:
                                        buffer_obj = child
                                        object_list.append(str(child))
                                    # Handle regular objects in prepositional phrases
                                    else:
                                        obj_phrase = self._get_full_object_phrase(child)
                                        buffer_obj = child
                                        object_list.append(obj_phrase)
            
            # If still no objects found, try to extract important nouns
            if not object_list and not buffer_obj:
                for word in sentence:
                    if word.pos_ == 'NOUN' and word.dep_ not in ('nsubj', 'nsubjpass'):
                        # Exclude common stop words
                        if word.text.lower() not in ('year', 'time', 'day', 'thing', 'person'):
                            obj_phrase = self._get_full_object_phrase(word)
                            buffer_obj = word
                            object_list.append(obj_phrase)
                            break
            
            return object_list, buffer_obj

    def _get_full_object_phrase(self, word):
        """Extract full noun phrase for an object"""
        phrase_parts = []
        
        # Add compound words before the object
        for child in word.lefts:
            if child.dep_ in ('compound', 'amod', 'nummod', 'quantmod'):
                phrase_parts.append(str(child))
        
        # Add the main object word
        phrase_parts.append(str(word))
        
        # Create the complete phrase
        if phrase_parts:
            return " ".join(phrase_parts)
        else:
            return str(word)

    def find_subj(self, sentence):
        subject_list = []
        # """ SUBJECT FINDING loop"""
        dep_word = [word.dep_ for word in sentence]
        word_dep_count_subj = [dep_word.index(word) for word in dep_word if word in ('nsubj', 'subj', 'nsubjpass')]
        if word_dep_count_subj:
            word_dep_count_subj = word_dep_count_subj[0] + 1
        else:
            word_dep_count_subj = 1

        subject_final = ""
        for word in sentence:
            # print(word.dep_, word)
            if word_dep_count_subj > 0:
                # in prime minister it gives compound and then nmod
                if word.dep_ in ('compound') or word.dep_ in ('nmod') or word.dep_ in ('amod') or word.dep_ in ('poss') or word.dep_ in ('case') or word.dep_ in ('nummod'):
                    if subject_final == "":
                        subject_final = str(word)
                        word_dep_count_subj = word_dep_count_subj - 1
                    elif word.dep_ in ('case'):
                        subject_final = subject_final+ "" +str(word)
                        word_dep_count_subj = word_dep_count_subj - 1
                    else:
                        subject_final = subject_final+ " " +str(word)
                        word_dep_count_subj = word_dep_count_subj - 1
                elif word.dep_ in ('nsubj', 'subj', 'nsubjpass'):
                    if subject_final == "":
                        subject_final = str(word)
                        subject_list.extend([str(a.text) for a in word.subtree if a.dep_ in ('conj')])
                        word_dep_count_subj = word_dep_count_subj - 1
                        break
                    else:
                        subject_final = subject_final+" "+str(word)
                        subject_list.extend([str(a.text) for a in word.subtree if a.dep_ in ('conj')])
                        word_dep_count_subj = word_dep_count_subj - 1
                        break
                else:
                    pass

        subject_list.append(subject_final)
        return subject_list

    def find_relation(self, buffer_obj):
        aux_relation = ""
        
        # Handle case when buffer_obj is None
        if buffer_obj is None:
            return "unknown", aux_relation
        
        # RELATION FINDING loop
        relation = [w for w in buffer_obj.ancestors if w.dep_ =='ROOT']

        if relation:
            relation = relation[0]
            sp_relation = relation
            
            # Collect all important parts of the relation
            relation_parts = [str(relation)]
            
            # Check for particles, auxiliaries or adverbs that modify the relation
            for child in relation.children:
                # Add auxiliary verbs (helped, has done, etc.)
                if child.dep_ == 'aux' and child.pos_ == 'AUX' and child.i < relation.i:
                    relation_parts.insert(0, str(child))
                
                # Add particles (grew up, turned on, etc.)
                elif child.dep_ == 'prt' and child.pos_ == 'ADP':
                    relation_parts.append(str(child))
                
                # Add important adverbs (quickly ran, etc.)
                elif child.dep_ == 'advmod' and child.pos_ == 'ADV':
                    relation_parts.append(str(child))
                    
                # Add prepositions that are part of the relation (moved to, went into)
                elif child.dep_ in ('prep') and str(child) not in ('with', 'by'):
                    relation_parts.append(str(child))
                    
                    # For phrasal verbs with prepositions (like "emigrate to")
                    for grandchild in child.children:
                        if grandchild.dep_ == 'pobj' and str(grandchild) in buffer_obj.text:
                            # We've found the object via the preposition, so record preposition as part of relation
                            break
                            
                # Handle xcomp (open complement clauses) - "began working", "helped create"
                elif child.dep_ == 'xcomp':
                    aux_relation = str(child)
                    
            # Join all relation parts
            relation = " ".join(relation_parts)
            
            # Handle edge cases with infinitives
            if str(relation) == "be" and sp_relation.nbor(1).pos_ == 'ADP' and str(sp_relation.nbor(1)) == 'to':
                relation = " ".join((str(relation), str(sp_relation.nbor(1))))
                
        else:
            relation = 'unknown'

        return relation, aux_relation

    def normal_sent(self, sentence):
        time, place = self.get_time_place_from_sent(sentence)

        subject_list, object_list = [], []
        aux_relation, child_with_comp = "", ""

        subject_list = self.find_subj(sentence)
        object_list, buffer_obj = self.find_obj(sentence, place, time)
        relation, aux_relation = self.find_relation(buffer_obj)

        self.ent_pairs = []

        if time:
            time = time[0]
        else:
            time = ""

        if place:
            place = place[0]
        else:
            place = ""

        pa, pb = [], []
        for m in subject_list:
            pa.append([m])

        for n in object_list:
            pb.append([n])

        # Only create pairs if we have both subjects and objects
        if pa and pb:
            for m in range(0, len(pa)):
                for n in range(0, len(pb)):
                    self.ent_pairs.append([str(pa[m][0]).lower(), str(relation).lower(), str(aux_relation).lower(), str(pb[n][0]).lower(), str(time), str(place)])

        return self.ent_pairs

    def question_pairs(self, question__):
        questionNLPed = self.nlp(question__)
        maybe_object = ([i for i in questionNLPed if i.dep_ in ('obj', 'pobj', 'dobj')])
        maybe_place, maybe_time = [], []
        aux_relation = ""
        maybe_time, maybe_place = self.get_time_place_from_sent(questionNLPed)
        object_list = []

        # For WHO questions
        if question__.lower().startswith("who"):
            # Extract the main verb (ROOT)
            root = None
            for token in questionNLPed:
                if token.dep_ == "ROOT":
                    root = token
                    break
                    
            if root:
                # Build the relation (verb phrase)
                relation_parts = [str(root)]
                
                # Add auxiliaries and particles
                for child in root.children:
                    if child.dep_ in ('aux', 'auxpass') and child.i < root.i:
                        relation_parts.insert(0, str(child))
                    elif child.dep_ == 'prt':
                        relation_parts.append(str(child))
                        
                relation = " ".join(relation_parts)
                
                # Extract the object
                for obj in questionNLPed:
                    if obj.dep_ in ('obj', 'dobj', 'pobj'):
                        # Check if it's a WH-word
                        if not str(obj).lower() in ("who", "what", "when", "where", "why", "how"):
                            # Build full object phrase
                            obj_phrase = self._get_full_object_phrase(obj)
                            object_list.append(obj_phrase)
                        
                # Handle prepositional phrases
                for token in questionNLPed:
                    if token.dep_ == 'prep':
                        # Check for special prepositions like "to", "in", "at"
                        if str(token).lower() in ("to", "in", "at", "on", "by"):
                            relation_parts.append(str(token))
                            
                        # Get objects of prepositions
                        for child in token.children:
                            if child.dep_ == 'pobj':
                                if not str(child).lower() in ("who", "what", "when", "where", "why", "how"):
                                    obj_phrase = self._get_full_object_phrase(child)
                                    object_list.append(obj_phrase)
                
                # Update relation with any prepositions
                relation = " ".join(relation_parts)
                
                # If no object found but there are prepositions like "to", check for hidden objects
                if not object_list:
                    for token in questionNLPed:
                        if token.dep_ == 'prep' and str(token).lower() in ("to", "in", "at", "on"):
                            for child in token.children:
                                if child.dep_ == 'pobj':
                                    obj_phrase = self._get_full_object_phrase(child)
                                    object_list.append(obj_phrase)
                    
                # If still no object, use "who" as a placeholder
                if not object_list:
                    object_list.append("who")
                    
                # Create entity pair
                self.ent_pairs = []
                if object_list:
                    obj = object_list[0]  # Use the first object found
                    
                    if maybe_time and maybe_place:
                        self.ent_pairs.append(["who", relation.lower(), aux_relation.lower(), obj.lower(), str(maybe_time[0]).lower(), str(maybe_place[0]).lower()])
                    elif maybe_time:
                        self.ent_pairs.append(["who", relation.lower(), aux_relation.lower(), obj.lower(), str(maybe_time[0]).lower(), ""])
                    elif maybe_place:
                        self.ent_pairs.append(["who", relation.lower(), aux_relation.lower(), obj.lower(), "", str(maybe_place[0]).lower()])
                    else:
                        self.ent_pairs.append(["who", relation.lower(), aux_relation.lower(), obj.lower(), "", ""])
                    
                    return self.ent_pairs
                else:
                    # Default pattern for "who" questions with no clear object
                    self.ent_pairs.append(["who", relation.lower(), "", "what", "", ""])
                    return self.ent_pairs
                    
        # For WHAT questions
        elif question__.lower().startswith("what"):
            # Extract the main verb (ROOT)
            root = None
            subject = "what"  # Default
            
            for token in questionNLPed:
                if token.dep_ == "ROOT":
                    root = token
                    break
                    
            if root:
                # Find subject (typically comes before the root)
                for token in questionNLPed:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        # Build full subject
                        subject_parts = []
                        for child in token.lefts:
                            if child.dep_ in ('compound', 'amod', 'nummod'):
                                subject_parts.append(str(child))
                        
                        subject_parts.append(str(token))
                        subject = " ".join(subject_parts)
                        break
                
                # Build relation phrase
                relation_parts = [str(root)]
                
                # Add auxiliaries and particles
                for child in root.children:
                    if child.dep_ in ('aux', 'auxpass') and child.i < root.i:
                        relation_parts.insert(0, str(child))
                    elif child.dep_ == 'prt':
                        relation_parts.append(str(child))
                        
                # Handle special prepositions
                for token in questionNLPed:
                    if token.dep_ == 'prep' and token.head == root:
                        if str(token).lower() in ("to", "in", "at", "on", "by"):
                            relation_parts.append(str(token))
                
                relation = " ".join(relation_parts)
                
                # Create entity pair
                self.ent_pairs = []
                
                if maybe_time and maybe_place:
                    self.ent_pairs.append([subject.lower(), relation.lower(), "", "what", str(maybe_time[0]).lower(), str(maybe_place[0]).lower()])
                elif maybe_time:
                    self.ent_pairs.append([subject.lower(), relation.lower(), "", "what", str(maybe_time[0]).lower(), ""])
                elif maybe_place:
                    self.ent_pairs.append([subject.lower(), relation.lower(), "", "what", "", str(maybe_place[0]).lower()])
                else:
                    self.ent_pairs.append([subject.lower(), relation.lower(), "", "what", "", ""])
                    
                return self.ent_pairs
                
        # For WHEN questions
        elif question__.lower().startswith("when"):
            # Extract the main verb (ROOT)
            root = None
            subject = "unknown"  # Default
            
            for token in questionNLPed:
                if token.dep_ == "ROOT":
                    root = token
                    break
                    
            if root:
                # Find subject (typically comes before the root)
                for token in questionNLPed:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        # Build full subject
                        subject_parts = []
                        for child in token.lefts:
                            if child.dep_ in ('compound', 'amod', 'nummod'):
                                subject_parts.append(str(child))
                        
                        subject_parts.append(str(token))
                        subject = " ".join(subject_parts)
                        break
                
                # Build relation phrase
                relation_parts = [str(root)]
                
                # Add auxiliaries and particles
                for child in root.children:
                    if child.dep_ in ('aux', 'auxpass') and child.i < root.i:
                        relation_parts.insert(0, str(child))
                    elif child.dep_ == 'prt':
                        relation_parts.append(str(child))
                
                # Handle special prepositions (especially for "when" questions)
                for token in questionNLPed:
                    if token.dep_ == 'prep' and token.head == root:
                        relation_parts.append(str(token))
                        
                        # For prepositions like "to" in "emigrate to", add the object too
                        for child in token.children:
                            if child.dep_ == 'pobj':
                                # Add the object to our list
                                obj_phrase = self._get_full_object_phrase(child)
                                object_list.append(obj_phrase)
                
                relation = " ".join(relation_parts)
                
                # Create entity pair
                self.ent_pairs = []
                
                if object_list:
                    obj = object_list[0]  # Use the first object found
                    
                    if maybe_place:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", obj.lower(), "when", str(maybe_place[0]).lower()])
                    else:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", obj.lower(), "when", ""])
                else:
                    if maybe_place:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", "", "when", str(maybe_place[0]).lower()])
                    else:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", "", "when", ""])
                    
                return self.ent_pairs
                
        # For WHERE questions
        elif question__.lower().startswith("where"):
            # Extract the main verb (ROOT)
            root = None
            subject = "unknown"  # Default
            
            for token in questionNLPed:
                if token.dep_ == "ROOT":
                    root = token
                    break
                    
            if root:
                # Find subject (typically comes before the root)
                for token in questionNLPed:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        # Build full subject
                        subject_parts = []
                        for child in token.lefts:
                            if child.dep_ in ('compound', 'amod', 'nummod'):
                                subject_parts.append(str(child))
                        
                        subject_parts.append(str(token))
                        subject = " ".join(subject_parts)
                        break
                
                # Build relation phrase
                relation_parts = [str(root)]
                
                # Add auxiliaries and particles
                for child in root.children:
                    if child.dep_ in ('aux', 'auxpass') and child.i < root.i:
                        relation_parts.insert(0, str(child))
                    elif child.dep_ == 'prt':
                        relation_parts.append(str(child))
                
                relation = " ".join(relation_parts)
                
                # Extract objects
                for obj in questionNLPed:
                    if obj.dep_ in ('obj', 'dobj', 'pobj'):
                        # Check if it's a WH-word
                        if not str(obj).lower() in ("who", "what", "when", "where", "why", "how"):
                            # Build full object phrase
                            obj_phrase = self._get_full_object_phrase(obj)
                            object_list.append(obj_phrase)
                
                # Create entity pair
                self.ent_pairs = []
                
                if object_list:
                    obj = object_list[0]  # Use the first object found
                    
                    if maybe_time:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", obj.lower(), str(maybe_time[0]).lower(), "where"])
                    else:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", obj.lower(), "", "where"])
                else:
                    if maybe_time:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", "", str(maybe_time[0]).lower(), "where"])
                    else:
                        self.ent_pairs.append([subject.lower(), relation.lower(), "", "", "", "where"])
                    
                return self.ent_pairs
                
        # Fallback for unhandled question types
        return self.ent_pairs