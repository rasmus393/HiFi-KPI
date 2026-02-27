import pandas as pd
def find_path_to_root(df):
    child_to_parent = {}
    for child, parent in zip(df['child'], df['parent']):
        if pd.notna(parent):
            child_to_parent[child] = parent
    paths = {}
    
    def find_path(node):
        if node in child_to_parent and child_to_parent[node] == "":
            return [node]
        if node not in child_to_parent:
            return [node]
        parent_path = find_path(child_to_parent[node])
        return [node] + parent_path
    for child in df['child'].unique():
        paths[child] = find_path(child)

    result_df = pd.DataFrame({
        'child': list(paths.keys()), 
        'path_to_root': list(paths.values())
    })
    
    return result_df, paths