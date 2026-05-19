import csv
import os
import augment_models.atom_changer_synt as atom_changer_synt

def atom_changer_linear_mod(reaction_groups, reaction_groups_val, n_iter):
    def chunk_dict(d, chunk_size):
        items = list(d.items())
        for i in range(0, len(items), chunk_size):
            yield dict(items[i:i+chunk_size])
    
    def flatten_result_to_rows(results):
        rows = []
        for idx, entry in enumerate(results):
            generated_inputs = entry.get('generated_input', [])
            generated_outputs = entry.get('generated_output', [])
    
            for gi, go in zip(generated_inputs, generated_outputs):
                rows.append({
                    'inputs': gi,
                    'output': go,
                    'reaction_center': entry['cluster'],
                    'source_row': entry['id']
                })
        return rows
    
    def remove_same_data(data):
        seen = set()
        unique_data = []
        for entry in data:
            tup = tuple((k, str(entry[k])) for k in sorted(entry))
            if tup not in seen:
                seen.add(tup)
                unique_data.append(entry)
        return unique_data
    
    def append_csv_chunk(data, filename, write_header=True):
        mode = 'a'
        with open(filename, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'class', 'input>>output'])
            formatted_data = []
            for row in data:
                new_row = {
                    'id': row['source_row'],
                    'class': row['reaction_center'],
                    'input>>output': f"{row['inputs']}>>{row['output']}"
                }
                formatted_data.append(new_row)
    
            writer.writerows(formatted_data)
        print(f"[Appended] {len(data)} rows → {filename}")
    
    # ---------- Training ----------
    train_csv_path = 'linear_datasets/train.csv'
    print("[Start] AtomChanger augmentation for training...")
    
    for i, chunk in enumerate(chunk_dict(reaction_groups, 25)):
        print(f"[Train] Chunk {i + 1}")
        results = atom_changer_synt.atom_changer_process(chunk, n_iter=n_iter)
        if results:
            flattened = flatten_result_to_rows(results)
            chunk_result = remove_same_data(flattened)
            append_csv_chunk(chunk_result, train_csv_path)
    
    # ---------- Validation ----------
    val_csv_path = 'linear_datasets/val.csv'
    print("[Start] AtomChanger augmentation for validation...")
    
    for i, chunk in enumerate(chunk_dict(reaction_groups_val, 25)):
        print(f"[Val] Chunk {i + 1}")
        results = atom_changer_synt.atom_changer_process(chunk, n_iter=n_iter)
        if results:
            flattened = flatten_result_to_rows(results)
            chunk_result = remove_same_data(flattened)
            append_csv_chunk(chunk_result, val_csv_path)