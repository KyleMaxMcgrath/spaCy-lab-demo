import spacy
from spacy.pipeline.entity_ruler import EntityRuler
from spacy.matcher import DependencyMatcher

# Load a spaCy model (you can choose a model depending on your language and requirements)
nlp = spacy.load("en_core_web_sm")

# Define patterns for lab results, lab units, and lab types for the EntityRuler
patterns = [
    {"label": "LAB_RESULT", "pattern": [{"LOWER": {"in": ["result", "value"]}}]},
    {"label": "LAB_UNIT", "pattern": [{"LOWER": {"in": ["mg/dL", "mmol/L"]}}]},
    {"label": "LAB_TYPE", "pattern": [{"LOWER": {"in": ["cholesterol", "glucose"]}}]},
]

# Create an EntityRuler and add the patterns
ruler = EntityRuler(nlp)
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

# Define dependency patterns for associating lab results with lab units and lab types
dependency_patterns = [
    {
        "RIGHT_ID": "lab_result",
        "RIGHT_ATTRS": {"label": {"in": ["LAB_RESULT"]}},
        "LEFT_ID": "lab_unit",
        "LEFT_ATTRS": {"label": {"in": ["LAB_UNIT"]}},
        "PATTERN": [{"DEP": {"in": ["prep", "pobj", "nmod"]}}],
    },
    {
        "RIGHT_ID": "lab_result",
        "RIGHT_ATTRS": {"label": {"in": ["LAB_RESULT"]}},
        "LEFT_ID": "lab_type",
        "LEFT_ATTRS": {"label": {"in": ["LAB_TYPE"]}},
        "PATTERN": [{"DEP": {"in": ["prep", "pobj", "nmod"]}}],
    },
]

# Create a DependencyMatcher and add the dependency patterns
matcher = DependencyMatcher(nlp.vocab)
matcher.add("lab_dependency", [dependency_patterns])

# Process the text
text = "The cholesterol level is 200 mg/dL. The glucose result is 120 mmol/L."
doc = nlp(text)

# Extract entities and associated lab information
lab_entities = {"LAB_RESULT": [], "LAB_UNIT": [], "LAB_TYPE": []}

for ent in doc.ents:
    lab_entities[ent.label_].append(ent.text)

for match_id, token_ids in matcher(doc):
    if "lab_result" in token_ids and "lab_unit" in token_ids:
        lab_entities["LAB_RESULT"].append(doc[token_ids["lab_result"][0]:token_ids["lab_unit"][0] + 1].text)
        lab_entities["LAB_UNIT"].append(doc[token_ids["lab_unit"][0]].text)
    elif "lab_result" in token_ids and "lab_type" in token_ids:
        lab_entities["LAB_RESULT"].append(doc[token_ids["lab_result"][0]:token_ids["lab_type"][0] + 1].text)
        lab_entities["LAB_TYPE"].append(doc[token_ids["lab_type"][0]].text)

# Print the extracted lab information
print("Lab Results:", lab_entities["LAB_RESULT"])
print("Lab Units:", lab_entities["LAB_UNIT"])
print("Lab Types:", lab_entities["LAB_TYPE"])
