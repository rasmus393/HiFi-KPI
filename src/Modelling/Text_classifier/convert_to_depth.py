from .bert_based_embeddings import extract_data
from .top_down_paths import find_path_to_root
import pandas as pd

def convert_to_level_with_count(json_dataset, conversion_dict):
    """
    
    Args:
        json_dataset (list): The dataset containing entity labels to convert.
        level (int): The number of levels to traverse up.
        conversion_dict (dict): The child-to-parent mapping dictionary.
    
    Returns:
        tuple: The updated dataset and the count of unconverted labels.
    """
    new_dataset = []
    for entry in json_dataset:
        updated_entry = entry.copy()
        updated_entities = []

        for entity in entry["entities"]:
            new_label = conversion_dict.get(entity["Label"])
            newEntity = entity.copy()
            if new_label:
                newEntity['Label'] = new_label
            else:
                newEntity['Label'] = entity['Label']
            updated_entities.append(newEntity)

        updated_entry["entities"] = updated_entities
        new_dataset.append(updated_entry)

    return new_dataset 

def collapse_leaves_one_level(df):
    """
    Collapses a single 'leaf' level of the graph and returns:
      (1) A DataFrame with those leaf edges removed,
      (2) A dictionary mapping child -> parent for the collapsed leaves.
      
    Root nodes (parent="") are preserved and not collapsed further.
    """
    leaves = set(df['child']) - set(df['parent'])
    all_children = set(df['child'])
    df_leaves = df[df['child'].isin(leaves)]
    df_leaves_non_root = df_leaves[df_leaves['parent'] != ""]
    child_to_parent = dict(zip(df_leaves_non_root['child'], df_leaves_non_root['parent']))
    df_collapsed = df[~df['child'].isin(child_to_parent.keys())]
    
    # For each parent that will become a new root, add a proper root entry
    for leaf, parent in child_to_parent.items():
        if parent not in all_children and parent != "":
            leaf_row = df_leaves[df_leaves['child'] == leaf].iloc[0]

            new_row = leaf_row.copy()
            new_row['child'] = parent
            new_row['parent'] = ""

            df_collapsed = pd.concat([df_collapsed, pd.DataFrame([new_row])], ignore_index=True)
    
    return df_collapsed, child_to_parent

def multi_level_collapse(
    df,                  
    train_data,
    validation_data,
    test_data,
    iterations=10
):
    """
    Performs `iterations` rounds of:
      1. Collapsing leaves in `df`.
      2. Using the resulting `child_to_parent` mappings to update train/validation/test data.
    
    Returns:
      (df_collapsed, train_data, validation_data, test_data,
       (train_sentences, train_labels),
       (val_sentences, val_labels),
       (test_sentences, test_labels))
    """
    
    df_current = df.copy()
    train_labels_level = []
    validation_labels_level = []
    test_labels_level = []
    child_path_dicts= []
    current_dfs = []
    for i in range(iterations):
        df_current, child_to_parent = collapse_leaves_one_level(df_current)
        result_df, _ = find_path_to_root(df_current)
        child_path_dict = dict(zip(result_df['child'], result_df['path_to_root']))
        # df current is used so we can collapse once more.
        print(f"Iteration {i+1} - Collapsed {len(child_to_parent)} leaves")
        
        train_data = convert_to_level_with_count(train_data, child_to_parent)
        validation_data = convert_to_level_with_count(validation_data, child_to_parent)
        test_data= convert_to_level_with_count(test_data, child_to_parent)

        _, train_labels = extract_data(train_data)
        _, val_labels     = extract_data(validation_data)
        _, test_labels   = extract_data(test_data)
        print('unique train labels', len(set(train_labels)))
        train_labels_level.append(train_labels)
        validation_labels_level.append(val_labels)
        test_labels_level.append(test_labels)
        child_path_dicts.append(child_path_dict)
        current_dfs.append(df_current)
    

    return train_labels_level, validation_labels_level, test_labels_level, child_path_dicts, current_dfs