import os
import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from GTZANClassifier import GTZANClassifier

class GTZANClassifierAnalysis:
    def __init__(
        self,
        pkl_filepath='./trained_models',
        data_dirpath=f'./GTZAN/Data_filtered',
        sub_folder='/og_params',
        dataset_split = 'val',
        feat_types = ['full', 'mfcc'],
        rng_states = ['42', '351', '605'],
        time_states = ['30']):
        self.pkl_filepath = f'{pkl_filepath}{sub_folder}'
        self.data_dirpath = data_dirpath
        self.dataset_split = dataset_split
        self.feat_types = feat_types
        self.rng_states = rng_states
        self.time_states = time_states
        self.classifers = {}


    def _create_master_df(self):
        master_records = []

        for config_name, clf in self.classifers.items():
            pkl_path = f'./trained_models/{config_name}.pkl'
            file_size_kb = os.path.getsize(pkl_path) / 1024 if os.path.exists(pkl_path) else 0
            
            parts = config_name.split('_')
            dataset_variant = parts[0]
            feature_subset = parts[1]
            rng_state = parts[2][3:]

            for model_name, model_dict in clf.models.items():
                if 'metrics' not in model_dict:
                    continue
                    
                m = model_dict['metrics'][self.dataset_split]
                m_train = model_dict['training metrics']
                
                model_bytes = pickle.dumps(model_dict['model'])
                model_memory_kb = len(model_bytes) / 1024

                master_records.append({
                    'Dataset': dataset_variant,
                    'Features': feature_subset,
                    'rng State': rng_state,
                    'Configuration': config_name,
                    'Model': model_name,
                    'Test Balanced Accuracy': 1.0 - m_train['test_error'],
                    'Macro F1-Score': m.get('f1_score', 0),
                    'Precision' : m['precision'],
                    'Recall' : m['recall'],
                    'Train Time (s)': m_train.get('training_time_seconds', 0),
                    'Model Memory (KB)': model_memory_kb,
                    'File Size (KB)': file_size_kb
                })

        self.df_master = pd.DataFrame(master_records)

        self.df_master = self.df_master.sort_values(by=['Test Balanced Accuracy', 'Macro F1-Score'], ascending=False).reset_index(drop=True)
        return self

    def load_data(self):
        for t in self.time_states:
            for feat in self.feat_types:
                for rng in self.rng_states:
                    name = f'{t}s_{feat}_rng{rng}'
                    self.classifers[name] = GTZANClassifier()
                    self.classifers[name].load_models(
                        pkl_filepath=f'{self.pkl_filepath}/{name}.pkl',
                        data_dirpath=f'{self.data_dirpath}/{t}s_split/')
        return self._create_master_df()

    def print_master_df(self):
        print("\n--- MASTER PERFORMANCE & RESOURCE COMPARISON ---")
        print(self.df_master.to_string(index=False))

    def print_best_config(self):
        best_row = self.df_master.iloc[0]
        print(f"\nBest Overall Model:")
        print(f"\t- Configuration: {best_row['Configuration']}")
        print(f"\t- Model Type: {best_row['Model']}")
        print(f"\t- Test Accuracy: {best_row['Test Balanced Accuracy']:.4f}")
        print(f"\t- Macro F1-Score: {best_row['Macro F1-Score']:.4f}")

    def get_total_master_memory(self, unit='KB'):
        """
        Loops through the master DataFrame and calculates the total memory footprint 
        of all stored models using the preexisting 'Model Memory (KB)' column.
        
        Parameters:
        -----------
        df_master : pd.DataFrame
            The master records DataFrame containing 'Model Memory (KB)'.
        unit : str
            The desired output unit: 'KB' or 'MB'.
            
        Returns:
        --------
        float
            The total accumulated memory size.
        """
        if 'Model Memory (KB)' not in self.df_master.columns:
            print("Error: 'Model Memory (KB)' column not found in DataFrame.")
            return 0.0

        # Sum up all model memory entries in KB
        total_kb = self.df_master['Model Memory (KB)'].sum()
        
        if unit.upper() == 'MB':
            total_value = total_kb / 1024
            unit_label = 'MB'
        else:
            total_value = total_kb
            unit_label = 'KB'
            
        print(f"\n--- TOTAL MASTER MEMORY REPORT ---")
        print(f"Total entries evaluated: {len(self.df_master)}")
        print(f"Total model memory footprint: {total_value:.2f} {unit_label}")
        print("=" * 34 + "\n")
        
        return total_value

    def plot_metric_comparison(self, metric='Macro F1-Score'):
        """
        Plots a grouped bar chart comparing all models across configurations 
        for any specified metric found in df_master.
        """
        # Pivot data so models become columns and configurations become rows
        df_pivot = self.df_master.pivot(index='Configuration', columns='Model', values=metric)

        fig, ax = plt.subplots(figsize=(12, 6))
        df_pivot.plot(kind='bar', ax=ax, width=0.8, colormap='tab10')

        ax.set_title(f'{metric} Comparison Across All Configurations and Models', fontsize=14)
        ax.set_xlabel('Dataset & Feature Configuration', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        
        # Set y-limits automatically based on metric type (probabilities/scores vs time/memory)
        if 'Error' in metric or 'Accuracy' in metric or (
                'F1' in metric or 'Precision' in metric or 'Recall' in metric):
            ax.set_ylim(0, 1.05)
            
        plt.xticks(rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_memory_footprint(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors/markers manually or let matplotlib cycle through them
        models_list = self.df_master['Model'].unique()
        colors = plt.cm.tab10.colors

        for i, model_name in enumerate(models_list):
            subset = self.df_master[self.df_master['Model'] == model_name]
            ax.scatter(
                subset['Model Memory (KB)'],
                subset['Test Balanced Accuracy'],
                label=model_name,
                color=colors[i % len(colors)],
                s=100,
                alpha=0.8,
                edgecolors='k'
            )

        ax.set_title('Model Memory Footprint vs. Balanced Accuracy', fontsize=14)
        # ax.set_xscale('log')
        ax.set_xlabel('Serialized Model Memory Size (KB)', fontsize=12)
        ax.set_ylabel('Test Balanced Accuracy', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()

    def plot_per_class_heatmap(
            self, metric_key='f1_score_per_class', target_model='SVM (RBF)'):
        """
        Creates a heatmap showing per-class scores.
        - If target_model is a specific name (e.g., 'SVM (RBF)' - default), compares all configs for that model.
        - If target_model='ALL', compares every model across every configuration.
        
        metric_key options: 'precision_per_class', 'recall_per_class', 'f1_score_per_class'
        """
        matrix_data = []
        config_labels = []
        genre_labels = None

        for config_name, clf in self.classifers.items():
            # Determine which models to evaluate for this configuration
            if str(target_model).upper() == 'ALL':
                models_to_process = list(clf.models.keys())
            else:
                # Check if the requested target model exists in this configuration
                models_to_process = [target_model] if target_model in clf.models else []

            for model_name in models_to_process:
                model_dict = clf.models.get(model_name)
                if model_dict and 'metrics' in model_dict and metric_key in model_dict['metrics'][self.dataset_split]:
                    matrix_data.append(model_dict['metrics'][self.dataset_split][metric_key])
                    config_labels.append(f"{config_name} | {model_name}")
                    if genre_labels is None:
                        genre_labels = clf.labels

        if not matrix_data:
            print(f"No per-class metrics found for target_model='{target_model}'.")
            return None, None

        matrix = np.array(matrix_data) # Shape: (rows, genres)

        # Automatically scale figure height based on how many rows we are displaying
        fig, ax = plt.subplots(figsize=(10, max(4, len(config_labels) * 0.45)))
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0.0, vmax=1.0)

        # Configure axes
        ax.set_xticks(np.arange(len(genre_labels)))
        ax.set_yticks(np.arange(len(config_labels)))
        ax.set_xticklabels(genre_labels, rotation=45, ha="right")
        ax.set_yticklabels(config_labels, fontsize=9)

        # Add text annotations inside heatmap blocks for readability
        for i in range(len(config_labels)):
            for j in range(len(genre_labels)):
                val = matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

        title_suffix = f"({target_model})" if str(target_model).upper() != 'ALL' else "(All Models & Configs)"
        metric_title = metric_key.replace('_per_class', '').replace('_', ' ').upper()
        
        ax.set_title(f"Per-Class Performance Heatmap {title_suffix}\n[{metric_title}]", fontsize=14, pad=15)
        ax.set_xlabel("GTZAN Genre Classes", fontsize=12)
        ax.set_ylabel("Configuration & Model", fontsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def top4_configs(self,displayPlot=True):
        all_models_pool = []

        for config_name, clf in self.classifers.items():
            for model_name, model_dict in clf.models.items():
                if 'metrics' not in model_dict:
                    continue
                
                m = model_dict['metrics'][self.dataset_split]
                all_models_pool.append({
                    'config_name': config_name,
                    'model_name': model_name,
                    'classifier_obj': clf,
                    'model_dict': model_dict,
                    'test_accuracy': m.get('bal_accuracy',0),
                    'macro_f1': m.get('f1_score', 0)
                })

        df_pool = pd.DataFrame(all_models_pool)
        df_top4 = df_pool.sort_values(by=['test_accuracy', 'macro_f1'], ascending=False).head(4).reset_index(drop=True)

        print("Top 4 Models Across All Configurations:")
        for idx, row in df_top4.iterrows():
            print(f"{idx+1}. [{row['config_name']}] {row['model_name']} | F1: {row['macro_f1']:.4f} | Acc: {row['test_accuracy']:.4f}")

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 12),
            sharex=True, sharey=True,
            gridspec_kw={'hspace': 0.15, 'wspace': 0.1}
        )
        axes_flat = axes.ravel()
        im1 = None

        for i, row in df_top4.iterrows():
            ax = axes_flat[i]
            clf = row['classifier_obj']
            m_name = row['model_name']
            c_name = row['config_name']

            _,_,y_true, X_to_use = clf._parse_dataset_type(self.dataset_split)
            y_pred = clf.predict(X_to_use,target_model=m_name)
            
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true, 
                y_pred, 
                display_labels=clf.labels,  
                cmap=plt.cm.Blues,
                ax=ax,
                colorbar=False
            )
            
            ax.set_title(f"{m_name}\n({c_name})", fontsize=12, fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylabel("True Label", fontsize=10)
            ax.set_xlabel("Predicted Label", fontsize=10)
            
            if i == 0:
                im1 = disp.im_

        fig.colorbar(
            im1, ax=axes_flat,
            fraction=0.046, pad=0.04
        )

        fig.suptitle(
            "Confusion Matrix Comparison of the Top 4 Global Models" +
            f"\nUsing {self.dataset_split} split",
            fontsize=18, y=0.95, fontweight='bold'
        )
        if displayPlot: plt.show()

        return df_top4, fig, ax

    def plot_metric_comparison_from_classifiers(self, metric='Macro F1-Score', data_split='test'):
        """
        Plots a grouped bar chart comparing all models across configurations 
        by calling each classifier's built-in plot_metric_comparison logic 
        (or extracting the metrics on the fly).
        """
        master_records = []

        for config_name, clf in self.classifers.items():
            # Determine the target data split for the class method
            _,_, y_true, X_data = clf._parse_dataset_type(self.dataset_split)
            # print(X_data.columns)

            if y_true is None or X_data is None:
                continue  # Skip if data isn't loaded for this classifier

            # 2. Iterate through models inside this classifier instance
            for model_name, model_dict in clf.models.items():
                estimator = model_dict.get('model') if isinstance(model_dict, dict) else model_dict
                if estimator is None:
                    continue

                # Generate predictions using the class's own predict method
                y_pred = clf.predict(X_data, target_model=model_name)

                # Compute metrics dynamically using the same logic as your class method
                acc = accuracy_score(y_true, y_pred)
                bal_acc = balanced_accuracy_score(y_true, y_pred)
                error = 1.0 - acc
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='macro', zero_division=0
                )

                master_records.append({
                    'Configuration': config_name,
                    'Model': model_name,
                    'Error': error,
                    'Balanced Accuracy': bal_acc,
                    'Accuracy': acc,
                    'Macro F1-Score': f1,
                    'Precision': prec,
                    'Recall': rec
                })

        # 3. Convert records into a master DataFrame
        if not master_records:
            raise ValueError("No valid model metrics were found across the provided classifiers dictionary.")
        
        df_master = pd.DataFrame(master_records)

        # 4. Pivot data so models become columns and configurations become rows
        # (Matches your original grouping structure)
        df_pivot = df_master.pivot(index='Configuration', columns='Model', values=metric)

        # 5. Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        df_pivot.plot(kind='bar', ax=ax, width=0.8, colormap='tab10')

        ax.set_title(f'{metric} Comparison Across All Configurations ({data_split.capitalize()} Split)', fontsize=14)
        ax.set_xlabel('Dataset & Feature Configuration', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        
        # Set y-limits automatically based on metric type
        metric_lower = metric.lower()
        if any(term in metric_lower for term in ['error', 'accuracy', 'balanced', 'f1', 'precision', 'recall', 'score']):
            ax.set_ylim(0, 1.05)
            
        plt.xticks(rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.show()
        
        return fig, ax, df_master

    def compare_ensembles_val(
        self,
        baseline_model="KNN + Linear SVM (Ensemble)",
        candidate_model="Tuned Ensemble (SVM + RF + GB)",
        split="val",
    ):
        """
        Compare the baseline voting classifier against the candidate voting classifier
        strictly on `split` (defaults to 'val') across all loaded configurations in
        `self.classifers`.
        """
        rows = []

        for config_name, clf in sorted(self.classifers.items()):
            parts = config_name.split("_")
            dataset_variant = parts[0]
            feature_subset = parts[1]
            rng_state = int(parts[2][3:])

            base_entry = clf.models.get(baseline_model, {})
            cand_entry = clf.models.get(candidate_model, {})

            if "metrics" not in base_entry or "metrics" not in cand_entry:
                continue
            if split not in base_entry["metrics"] or split not in cand_entry["metrics"]:
                continue

            m_base = base_entry["metrics"][split]
            m_cand = cand_entry["metrics"][split]

            # Rank of baseline and candidate among all models in this config on `split`
            ranked_models = sorted(
                (
                    (m_name, md["metrics"][split]["bal_accuracy"], md["metrics"][split]["f1_score"])
                    for m_name, md in clf.models.items()
                    if "metrics" in md and split in md["metrics"]
                ),
                key=lambda x: (x[1], x[2]),
                reverse=True,
            )
            rank_lookup = {m_name: idx + 1 for idx, (m_name, _, _) in enumerate(ranked_models)}

            rows.append({
                "Dataset": dataset_variant,
                "Features": feature_subset,
                "rng": rng_state,
                "Configuration": config_name,
                "Base Rank": rank_lookup.get(baseline_model),
                "Cand Rank": rank_lookup.get(candidate_model),
                "Base Acc": m_base["accuracy"],
                "Cand Acc": m_cand["accuracy"],
                "Delta Acc": m_cand["accuracy"] - m_base["accuracy"],
                "Base Bal Acc": m_base["bal_accuracy"],
                "Cand Bal Acc": m_cand["bal_accuracy"],
                "Delta Bal Acc": m_cand["bal_accuracy"] - m_base["bal_accuracy"],
                "Base Macro F1": m_base["f1_score"],
                "Cand Macro F1": m_cand["f1_score"],
                "Delta Macro F1": m_cand["f1_score"] - m_base["f1_score"],
            })

        df_comp = pd.DataFrame(rows)
        if not df_comp.empty:
            df_comp = df_comp.sort_values(
                by=["Dataset", "Features", "rng"]
            ).reset_index(drop=True)
        return df_comp


    def summarize_ensemble_comparison(self, df_comp):
        """
        Aggregates mean and std across random states (`rng`) by Dataset and Features.
        """
        return (
            df_comp.groupby(["Dataset", "Features"], as_index=False)
            .agg(
                n_seeds=("rng", "nunique"),
                mean_base_bal_acc=("Base Bal Acc", "mean"),
                std_base_bal_acc=("Base Bal Acc", "std"),
                mean_cand_bal_acc=("Cand Bal Acc", "mean"),
                std_cand_bal_acc=("Cand Bal Acc", "std"),
                mean_delta_bal_acc=("Delta Bal Acc", "mean"),
                mean_base_f1=("Base Macro F1", "mean"),
                mean_cand_f1=("Cand Macro F1", "mean"),
                mean_delta_f1=("Delta Macro F1", "mean"),
            )
        )


    def compare_per_class_val(
        self,
        config_name,
        baseline_model="KNN + Linear SVM (Ensemble)",
        candidate_model="Tuned Ensemble (SVM + RF + GB)",
        class_names=None,
        split="val",
    ):
        """
        Extracts per-genre F1, recall, precision, and confusion matrices on `split` ('val')
        for a specific configuration (e.g., '30s_full_rng605').
        """
        clf = self.classifers[config_name]
        m_base = clf.models[baseline_model]["metrics"][split]
        m_cand = clf.models[candidate_model]["metrics"][split]

        n_classes = len(m_base["f1_score_per_class"])
        labels = clf.labels

        df_genre = pd.DataFrame({
            "Genre": labels,
            "Support": m_base["support_per_class"],
            "Base F1": m_base["f1_score_per_class"],
            "Cand F1": m_cand["f1_score_per_class"],
            "Delta F1": np.asarray(m_cand["f1_score_per_class"]) - np.asarray(m_base["f1_score_per_class"]),
            "Base Recall": m_base["recall_per_class"],
            "Cand Recall": m_cand["recall_per_class"],
            "Delta Recall": np.asarray(m_cand["recall_per_class"]) - np.asarray(m_base["recall_per_class"]),
        })

        return df_genre, m_base["confusion_matrix"], m_cand["confusion_matrix"]

    def get_best_baseline_candidate_models(
        self,
        baseline_model="KNN + Linear SVM (Ensemble)",
        candidate_model="Tuned Ensemble (SVM + RF + GB)",
        split="val",
        metric_keys=("bal_accuracy", "f1_score", "accuracy"),
        duration=None,
    ):
        """
        Finds the best-performing run/model dict in `self.classifers[name].models[model_name]`
        on `split` (defaults to 'val') for:
        - "full base"
        - "full cand"
        - "mfcc base"
        - "mfcc cand"

        Returns a dict mapping each slot label to a metadata/reference dict (or directly
        to `model_dict` if you prefer—see below).
        """
        slots = [
            ("full base", "full", baseline_model),
            ("full cand", "full", candidate_model),
            ("mfcc base", "mfcc", baseline_model),
            ("mfcc cand", "mfcc", candidate_model),
        ]

        best_records = {}

        for slot_name, target_feat, target_model_name in slots:
            best_score_tuple = None
            best_payload = None

            for config_name, clf in sorted(self.classifers.items()):
                parts = config_name.split("_")
                if len(parts) < 3:
                    continue
                dataset_variant = parts[0]
                feature_subset = parts[1]
                rng_state = parts[2][3:]

                if duration is not None and dataset_variant != duration:
                    continue
                if feature_subset != target_feat:
                    continue

                model_dict = clf.models.get(target_model_name)
                if not model_dict or "metrics" not in model_dict:
                    continue
                if split not in model_dict["metrics"]:
                    continue

                m = model_dict["metrics"][split]
                score_tuple = tuple(m.get(k, -float("inf")) for k in metric_keys)

                if best_score_tuple is None or score_tuple > best_score_tuple:
                    best_score_tuple = score_tuple
                    best_payload = {
                        "config_name": config_name,
                        "dataset": dataset_variant,
                        "features": feature_subset,
                        "rng": rng_state,
                        "model_name": target_model_name,
                        "score_tuple": score_tuple,
                        "model_dict": model_dict,  # exact dict from self.classifers[name].models[model_name]
                    }

            best_records[slot_name] = best_payload

        return best_records