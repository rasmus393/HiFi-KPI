import json
import re
import copy
from typing import List, Dict, Any

# I have not uploaded the non-cleaned dataset as I feel that is useless anyway, more for documentation of the cleaning steps.
with open(r"dummyTrain.json", "r", encoding="utf-8") as f:
    train = json.load(f)
with open(r"dummyVal.json", "r", encoding="utf-8") as f:
    validation = json.load(f)
with open(r"dummyTest.json", "r", encoding="utf-8") as f:
    test = json.load(f)

def drop_lowercase(data):
    filtered_data = [entry for entry in data if not entry['text'].lstrip()[0].islower()]
    return filtered_data

train = drop_lowercase(train)
validation = drop_lowercase(validation)
test = drop_lowercase(test)

TEXT_KEY = "text"
ENTITIES_KEY = "entities"
START_CHAR_KEY = "Start character"
END_CHAR_KEY = "End character"
CURRENCY_UNIT_KEY = "Currency / Unit"

def find_entries_starting_with_period(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in data if entry.get(TEXT_KEY, "").lstrip().startswith('.')]

def fix_all_leading_periods_in_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [remove_leading_period_and_whitespace(entry) for entry in dataset]

def remove_leading_period_and_whitespace(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes leading whitespace, a subsequent single period, and any whitespace
    immediately after that period from the entry's text. It then correctly
    shifts the character indices of all associated entities.
    """
    new_entry = copy.deepcopy(entry)
    original_text = new_entry[TEXT_KEY]

    match = re.match(r"^\s*\.\s*", original_text)
    if not match:
        new_text = original_text.lstrip()
        offset = len(original_text) - len(new_text)
    else:
        offset = match.end()
        new_text = original_text[offset:]

    # If nothing was changed, return the copy of the original entry.
    if offset == 0:
        return new_entry

    new_entry[TEXT_KEY] = new_text

    # Adjust entity indices based on the length of the removed prefix.
    for entity in new_entry.get(ENTITIES_KEY, []):
        old_start = entity[START_CHAR_KEY]
        old_end = entity[END_CHAR_KEY]
        new_start = old_start - offset
        new_end = old_end - offset

        # Sanity check: Ensure the new indices are valid for the new text length.
        if not (0 <= new_start < new_end <= len(new_text)):
            raise ValueError(
                f"Entity indices are out of bounds after shifting. "
                f"Original: '{original_text[old_start:old_end]}', "
                f"New indices: ({new_start}, {new_end}) on text of length {len(new_text)}."
            )

        # Sanity check: Ensure the text referenced by the entity remains the same.
        original_substring = original_text[old_start:old_end]
        new_substring = new_text[new_start:new_end]
        if original_substring != new_substring:
            raise ValueError(
                f"Entity text mismatch after shifting. "
                f"Expected: '{original_substring}', Got: '{new_substring}'"
            )
        entity[START_CHAR_KEY] = new_start
        entity[END_CHAR_KEY] = new_end

    return new_entry

# Find and fix the entries starting with a period (after whitespace)
entries_with_period = find_entries_starting_with_period(train)
print("Entries in train (before fix) that start with period:", len(entries_with_period))

fixed_train = fix_all_leading_periods_in_dataset(train)
fixed_validation = fix_all_leading_periods_in_dataset(validation)
fixed_test = fix_all_leading_periods_in_dataset(test)

def remove_entries_with_null_currency(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def is_entry_valid(entry: Dict[str, Any]) -> bool:
        return all(
            entity.get(CURRENCY_UNIT_KEY) is not None
            for entity in entry.get(ENTITIES_KEY, [])
        )
    return [entry for entry in dataset if is_entry_valid(entry)]

fixed_train_currency = remove_entries_with_null_currency(fixed_train)
fixed_validation_currency = remove_entries_with_null_currency(fixed_validation)
fixed_test_currency = remove_entries_with_null_currency(fixed_test)

#Define filenames
filenames = {
    "train": f"dot_fixed_train_rescraped.json",
    "validation": f"dot_fixed_validation_rescraped.json",
    "test": f"dot_fixed_test_rescraped.json",
}

# Save each dataset as a JSON file
for name, data in zip(filenames.values(), [fixed_train_currency, fixed_validation_currency, fixed_test_currency]):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
