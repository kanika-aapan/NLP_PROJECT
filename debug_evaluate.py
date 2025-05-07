import json
import os
import spacy

from kwQnA._exportPairs import exportToJSON
from kwQnA._getentitypair import GetEntity
from kwQnA._qna import QuestionAnswer


def debug_entity_extraction():
    """Debug the entity extraction process"""
    # Load a small sample text
    sample_text = """
    Albert Einstein developed the theory of relativity in 1905.
    Marie Curie discovered radium in 1898.
    """

    # Initialize components
    getent = GetEntity()
    export = exportToJSON()
    
    # Create a file with the sample text
    with open("sample_text.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # Print available methods
    print("Available methods in GetEntity:")
    methods = [method for method in dir(getent) if not method.startswith('__')]
    for method in methods:
        print(f"  - {method}")
    
    # Try different approaches
    print("\nTrying different approaches for entity extraction...")
    
    # Process text with spaCy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sample_text)
    
    # Try different method names
    try:
        print("\nTrying method: get_entity...")
        if hasattr(getent, 'get_entity'):
            result, count = getent.get_entity(doc)
            print(f"Success! Got {count} entities.")
            return result, count
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        print("\nTrying method: _get_entity...")
        if hasattr(getent, '_get_entity'):
            result, count = getent._get_entity(doc)
            print(f"Success! Got {count} entities.")
            return result, count
    except Exception as e:
        print(f"Error: {e}")
    
    # Try the standard approach from getentitypair.py (based on context)
    print("\nTrying to read and process the file directly...")
    try:
        with open("sample_text.txt", "r") as f:
            lines = f.readlines()
        
        processed_text = [text.strip() for text in lines if text.strip() != '']
        processed_text = " ".join(processed_text)
        processed_text = nlp(processed_text)
        
        for method_name in methods:
            if "entity" in method_name.lower():
                print(f"\nTrying method: {method_name}...")
                method = getattr(getent, method_name)
                try:
                    result = method(processed_text)
                    print(f"Success with {method_name}! Got result: {result}")
                    return result
                except Exception as e:
                    print(f"Error with {method_name}: {e}")
        
        print("\nCould not find a working entity extraction method.")
        return None
    except Exception as e:
        print(f"Error with file processing: {e}")
        return None


if __name__ == "__main__":
    print("Debugging the entity extraction process...")
    result = debug_entity_extraction()
    print("\nResult:", result)