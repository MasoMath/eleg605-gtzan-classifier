from concurrent.futures import ProcessPoolExecutor, as_completed
from GTZANClassifier import GTZANClassifier

def train_all_n_save_models(dirpath, rand_state, t_val):
    subsets = ['full', 'spectral', 'mfcc']
    saved_model_paths = {}

    for subset in subsets:
        print(
            f"\n--- Processing [Split: {t_val}s, "
            f"RNG: {rand_state}, Subset: {subset.upper()}] ---"
        )
        
        # Instantiate for each subset
        clf = GTZANClassifier(
            dirpath=dirpath,
            feature_subset=subset,
            random_state=rand_state
        )
        
        # Run pipeline steps
        clf.load_and_prepare_data()
        clf.add_baseline_models()
        clf.add_models()
        clf.fit_and_evaluate_models()
        # clf.compute_and_store_metrics()
        
        # Unique path per configuration to avoid file write collisions
        path = f'./trained_models/og_params/'
        clf.save_models(path)
        saved_model_paths[subset] = path
        
    return t_val, rand_state, saved_model_paths

if __name__ == '__main__':
    # Define your parameter grid
    splits = ['30','3'] # ,'3'
    random_states = [42, 351, 605]
    
    # Create all task arguments
    tasks = []
    for t in splits:
        for rng in random_states:
            dirpath = f'./GTZAN/Data_filtered/{t}s_split/'
            tasks.append((dirpath, rng, t))
            
    print(f"Starting parallel execution for {len(tasks)} configurations...")
    
    # Use ProcessPoolExecutor to run tasks in parallel
    # max_workers=None defaults to the number of CPU cores on your machine
    with ProcessPoolExecutor() as executor:
        # Submit all tasks to the pool
        futures = [
            executor.submit(train_all_n_save_models, dirpath, rng, t)
            for dirpath, rng, t in tasks
        ]
        
        # As each task completes, print its status
        for future in as_completed(futures):
            try:
                t_val, rng, paths = future.result()
                print(f"="*50)
                print(
                    f"Split: {t_val}s | rng: {rng} | Saved to: {paths['full']}"
                )
            except Exception as exc:
                print(f"Task generated an exception: {exc}")