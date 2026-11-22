"""
Known bugs:
    _load_dfs(), predict(), and feature columns names in general:
        This code assumes that the features columns are presented
        in the exact order by the original GTZAN dataset pulled
        from kaggle. With the companion code created for this
        project this is non-issue, but this code needs to be
        refactored with a self.feature_cols that contains the
        actual feature columns being used.

        Additionally, the feature names for my dataset and the
        included dataset may differ slightly, HOWEVER, the location of
        the TEMPO column is the same in both datasets, so the behaviour
        is as desired.

    load_models() and save_models():
        Technically, if anything changes with the code in this class
        most likely every saved model needs to be regenerated as each
        time the save/load methods are used they always assume the
        data is the current version.

    save_models():
        The filepath requires the save directory be passed as precisely:
            f'./GTZAN/Data_filtered/{t}s_split/'
        etc. With an appropriate trailing '/'
    
"""
import os
import re
import time
import joblib
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import LinearSVC, SVC
# from NNClassifer import PyTorchGenreClassifier


class GTZANClassifier:
    """
    A modular object-oriented pipeline for loading, training, 
    evaluating, and visualizing models on the GTZAN dataset, 
    with built-in tracking for training time, errors, and metrics.
    """
    def __init__(
            self,
            dirpath='./GTZAN/Data_filtered/30s_split/',
            feature_subset='full',
            labels=None,
            random_state=42):
        self.dirpath = dirpath
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.labels = labels

        if self.labels is None:
            self.labels = [
                'blues', 'classical', 'country', 'disco', 'hiphop', 
                'jazz', 'metal', 'pop', 'reggae', 'rock'
            ]
        
        self.spec_end = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.best_k = None
        self.models = {}
        self.test_accuracies = {}

        self.rf_results = None

# Random Helpers:

    def _parse_dataset_type(self, dataset_type):
        """Helper to validate and normalize dataset split names."""
        split_clean = dataset_type.lower().strip()
        if split_clean in ['val', 'valid']:
            return 'val', "VALIDATION", self.y_val, self.X_val
        elif split_clean == 'train':
            return 'train', "TRAINING", self.y_train, self.X_train
        elif split_clean == 'test':
            return 'test', "TEST", self.y_test, self.X_test
        else:
            raise ValueError(
                "dataset_type must be 'train', 'val', or 'test'"
            )

####################### Preparing Models #######################
    def _load_dfs(self, dirpath, data_type):
        """Helper to load train,test,val files"""
        clean_type = data_type.lower().strip()
        if clean_type in ['val', 'valid']:
            pattern = "*val*_features.csv"
        else:
            pattern = f"{clean_type}*_features.csv"
            
        matches = list(dirpath.glob(pattern))
        
        if not matches:
            raise FileNotFoundError(
                f"No CSV file matching pattern '{pattern}'" +
                f" found in {dirpath}")
            
        if len(matches) > 1:
            print(
                f"[Warning] Multiple files match '{pattern}':" +
                f" {[m.name for m in matches]}. " +
                f"Loading: {matches[0].name}")
            
        file_path = matches[0]
        print(f"Loading {clean_type.upper()} split from: {file_path.name}")

        df = pd.read_csv(file_path)
        nans = df.isna()
        print(
            f"Dataset '{file_path}'",
            f"has {nans.sum().sum()} missing entries.")
        
        X_raw = df.drop(columns=['filename', 'length', 'label'])
        y = df['label']
        
        col_names = X_raw.columns
        self.spec_end = int(np.argmax(col_names == 'tempo')) + 1
        
        if self.feature_subset == 'spectral':
            X = X_raw.iloc[:, :self.spec_end]
            print("Feature Subset: Spectral features only.")
        elif self.feature_subset == 'mfcc':
            X = X_raw.iloc[:, self.spec_end:]
            print("Feature Subset: MFCC features only.")
        elif self.feature_subset == 'full':
            X = X_raw
            print("Feature Subset: Full feature set.")
        else:
            raise(ValueError(
                "feature_subset can only be 'spectral', 'mfcc', or 'full'"))
        
        return X, y

    def load_and_prepare_data(self):
        """
            Loads the first train, test, and validiation files found in
            self.dirpath. Handles missing checks, and filters features.
        """
        dirpath = Path(self.dirpath)

        self.X_train, self.y_train = self._load_dfs(dirpath, 'train')
        self.X_test, self.y_test = self._load_dfs(dirpath, 'test')
        self.X_val, self.y_val = self._load_dfs(dirpath, 'val')

        print(
            f"Train size: {self.X_train.shape[0]}",
            f"| Val size: {self.X_val.shape[0]}",
            f"| Test size: {self.X_test.shape[0]}"
        )
            
        return self

    def split_data(self, test_size=0.15, val_size=0.15, X=None, y=None):
        """Returns stratified 70 / 15 / 15 train/val/test split."""
        warnings.warn(
            "split_data() is obsolete and should not be called. "
            "Returns train/test/val splits instead of saving them to class.\n"
            "Load data with load_and_prepare_data()",
            DeprecationWarning,
            stacklevel=2
        )
        if X is None or y is None: raise(ValueError("must provide X and y"))
        if val_size == 0: raise(ValueError("must have a validation split"))
        temp_size = test_size + val_size
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_size,
            random_state=self.random_state, stratify=self.y
        )
        temp_size = val_size/temp_size
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=self.random_state, stratify=y_temp
        )

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _find_best_k(self, k_end=30, displayPlot=False):
        """Runs validation loop to find optimal KNN k using balanced accuracy and optionally plots."""

        k_range = range(1, k_end)
        # k_range = range(5, max(6, k_end), 2)
        val_scores = []  

        for k in k_range:
            temp_knn = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=k))
            temp_knn.fit(self.X_train, self.y_train)
            val_pred = temp_knn.predict(self.X_val)
            
            # Compute balanced accuracy for validation predictions
            score = balanced_accuracy_score(self.y_val, val_pred)
            val_scores.append(score)

        self.best_k = k_range[np.argmax(val_scores)]
        print(f"winning k chosen for KNN (Balanced Accuracy): {self.best_k}")
        
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            k_range, val_scores,
            marker='o', linestyle='-', color='b')
        ax.set_title(
            'KNN Validation Balanced Accuracy Across Different $k$ Values ' +
            f'for {self.feature_subset}',
            fontsize=14)
        ax.set_xlabel('$k$')
        ax.set_ylabel('Balanced Accuracy')
        ax.grid(True)
        fig.tight_layout()
        
        if displayPlot:
            plt.show()
        else:
            plt.close(fig)
            
        return self.best_k, fig, ax

    def add_baseline_models(self):
        """Builds, trains, and evaluates original baseline models."""
        self._find_best_k()
        knn_pipeline = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=self.best_k,
                algorithm='brute',
                metric='minkowski',
                weights='uniform'))
        svm_pipeline = make_pipeline(
            StandardScaler(),
            LinearSVC(random_state=self.random_state, max_iter=10000))
        ensemble_model = VotingClassifier(
            estimators=[('knn', knn_pipeline), ('svm', svm_pipeline)],
            voting='hard'
        )

        self.models.update({
            'KNN': {'model': knn_pipeline, 'isTrained' : False},
            'Linear SVM': {'model': svm_pipeline, 'isTrained' : False}, 
            'KNN + Linear SVM (Ensemble)': {
                'model': ensemble_model, 'isTrained' : False} 
        })

        return self

    def add_models(self):
        """
        Runs hyperparameter GridSearch for RBF SVM, Random Forest, 
        and Gradient Boosting, then creates a descriptive Voting Ensemble.
        """
        cv_cls = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.random_state)

        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('svc', SVC(kernel='rbf', random_state=self.random_state))
        ])

        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('rf', RandomForestClassifier(random_state=self.random_state))
        ])

        gb_pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('gb', GradientBoostingClassifier(random_state=self.random_state))
        ])

        param_grid_svm = {
            'svc__C': [0.01, 0.1, 1, 10],
            'svc__gamma': ['scale', 'auto', 0.01, 0.1]
        }

        param_grid_rf = {
            'rf__n_estimators': [50, 100, 200],
            'rf__max_depth': [None, 5, 10, 20],
            'rf__min_samples_leaf': [2, 5, 10]
        }

        param_grid_gb = {
            'gb__n_estimators': [50, 100, 200],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 5]
        }

        print("Running Grid Search for SVM...")
        grid_svm = GridSearchCV(
            svm_pipeline, param_grid_svm, cv=cv_cls,
            scoring='balanced_accuracy', n_jobs=-1
        )
        grid_svm.fit(self.X_train, self.y_train)
        best_svm = grid_svm.best_estimator_

        print("Running Grid Search for Random Forest...")
        grid_rf = GridSearchCV(
            rf_pipeline, param_grid_rf, cv=cv_cls,
            scoring='balanced_accuracy', n_jobs=-1
        )
        grid_rf.fit(self.X_train, self.y_train)
        best_rf = grid_rf.best_estimator_

        print("Running Grid Search for Gradient Boosting...")
        grid_gb = GridSearchCV(
            gb_pipeline, param_grid_gb, cv=cv_cls,
            scoring='balanced_accuracy', n_jobs=-1
        )
        grid_gb.fit(self.X_train, self.y_train)
        best_gb = grid_gb.best_estimator_

        ensemble_model = VotingClassifier(
            estimators=[
                ('svm', best_svm),
                ('rf', best_rf),
                ('gb', best_gb)
            ],
            voting='hard'
        )

        self.models.update({
            'SVM (RBF)': {'model': best_svm, 'isTrained' : False},
            'Random Forest': {'model': best_rf, 'isTrained' : False},
            'Gradient Boosting': {'model': best_gb, 'isTrained' : False},
            'Tuned Ensemble (SVM + RF + GB)': {
                'model': ensemble_model, 'isTrained' : False}
        })

        self.rf_results = grid_rf

        return self

    # def add_nn_model(
    #         self, name="PyTorch MLP", hidden_dim=128, 
    #         lr=0.001, epochs=50, patience=5):
    #     """Adds a PyTorch neural network model entry to self.models."""
    #     nn_classifier = PyTorchGenreClassifier(
    #         hidden_dim=hidden_dim, lr=lr, epochs=epochs,
    #         num_classes=len(self.labels), patience=patience, name=name
    #     )
    #     self.models[name] = {
    #         'model': nn_classifier,
    #         'isTrained': False,
    #         'usesVal': True
    #     }
    #     return self
    
####################### Training Models + Metrics #######################
    def fit_and_evaluate_models(self):
        """
            Helper to fit, predict, and track performance metrics
            for active models.
            Errors are based on balanced accuracy
        """
        for name, model_dict in self.models.items():
            if model_dict['isTrained']:
                print(
                    f"Model {name} is already trained.",
                    "Skipping training...")
                continue

            print(f"Training/Evaluating {name}...")

            # usesVal = model_dict['model'].get('usesVal', False)

            # if usesVal:
            #     start_time = time.time()
            #     model_dict['model'].fit(
            #         self.X_train, self.y_train,
            #         self.X_val, self.y_val
            #     )
            #     train_time = time.time() - start_time
            # else:
            start_time = time.time()
            model_dict['model'].fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            y_train_pred = model_dict['model'].predict(self.X_train)
            y_val_pred = model_dict['model'].predict(self.X_val)
            y_pred = model_dict['model'].predict(self.X_test)
            
            model_dict['y_pred'] = y_pred
            
            train_error = 1.0 - balanced_accuracy_score(self.y_train, y_train_pred)
            val_error = 1.0 - balanced_accuracy_score(self.y_val, y_val_pred)
            test_error = 1.0 - balanced_accuracy_score(self.y_test, y_pred)
            
            model_dict['training metrics'] = {
                'training_time_seconds': train_time,
                'train_error': train_error,
                'val_error': val_error,
                'test_error': test_error
            }
            
            self.test_accuracies[name] = 1.0 - test_error
            print(
                f"-> Done in {train_time:.4f}s",
                f"| Test Error: {test_error:.4f}\n")
            model_dict['isTrained'] = True

        self.compute_and_store_metrics(dataset_type='train')
        self.compute_and_store_metrics(dataset_type='val')
        self.compute_and_store_metrics(dataset_type='test')
        return self

    def compute_and_store_metrics(self, dataset_type='test'):
        """
        Computes macro and per-class metrics for a specified 
        dataset split ('train', 'val', or 'test').
        """
        split_key, *_ = self._parse_dataset_type(dataset_type)
        
        if split_key == 'val':
            y_true, X_data = self.y_val, self.X_val
        elif split_key == 'train':
            y_true, X_data = self.y_train, self.X_train
        else:
            y_true, X_data = self.y_test, self.X_test

        if y_true is None or X_data is None:
            raise ValueError(
                f"Data for split '{split_key}' is not loaded. "
                "Run load_and_prepare_data() first."
            )

        for name, model_dict in self.models.items():
            if not model_dict.get('isTrained', False):
                print(
                    f"Model {name} is not trained. "
                    "Skipping metrics calculation..."
                )
                continue

            # Generate predictions for the specified split
            y_pred = self.predict(X_data, target_model=name)
            
            if split_key == 'test':
                model_dict['y_pred'] = y_pred

            mac_prec, mac_rec, mac_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            class_pc, class_r, class_f1, cs = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            if 'metrics' not in model_dict:
                model_dict['metrics'] = {}

            model_dict['metrics'][split_key] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'bal_accuracy': balanced_accuracy_score(
                    y_true, y_pred
                ),
                'precision': mac_prec,
                'recall': mac_rec,
                'f1_score': mac_f1,
                'precision_per_class': class_pc,
                'recall_per_class': class_r,
                'f1_score_per_class': class_f1,
                'support_per_class': cs,
                'confusion_matrix': confusion_matrix(
                    y_true, y_pred
                ),
                'classification_report': classification_report(
                    y_true, y_pred, zero_division=0
                )
            }
        return self

    def print_metrics_summary(self, dataset_type='test'):
        """
        Prints overall performance dataframe including time and errors
        for a specified dataset split ('train', 'val', or 'test').
        """
        split_key, split_name, *_ = self._parse_dataset_type(dataset_type)

        summary_list = []
        for name, model_dict in self.models.items():
            train_m = model_dict.get('training metrics', {})
            metrics_dict = model_dict.get('metrics', {})
            
            # Fall back to split-specific dict in metrics, or training metrics as backup
            split_m = metrics_dict.get(split_key, {})
            
            # Determine error based on split
            if split_key == 'train':
                err = train_m.get('train_error', split_m.get('accuracy', 0.0) if not split_m else 1.0 - split_m.get('bal_accuracy', 1.0))
            elif split_key == 'val':
                err = train_m.get('val_error', 1.0 - split_m.get('bal_accuracy', 1.0) if split_m else 0.0)
            else:
                err = train_m.get('test_error', 1.0 - split_m.get('bal_accuracy', 1.0) if split_m else 0.0)

            summary_list.append({
                'Model': name,
                'Train Time (s)': train_m.get('training_time_seconds', 0.0),
                'Train Error': train_m.get('train_error', 0.0),
                'Val Error': train_m.get('val_error', 0.0),
                'Test Error': train_m.get('test_error', 0.0),
                f'{split_name} Balanced Accuracy': split_m.get('bal_accuracy', 1.0 - err),
                f'{split_name} F1-Score (Macro)': split_m.get('f1_score', 0.0)
            })
            
        df_summary = pd.DataFrame(summary_list).set_index('Model')
        print(
            f"\nPERFORMANCE & TIMING SUMMARY ({split_name} SPLIT)",
            f"| Feature Set: {self.feature_subset.upper()}"
        )
        print("="*50)
        print(df_summary.to_string())
        print("="*50 + "\n")
        return df_summary

    def print_per_class_metrics(self, model_name, dataset_type='test'):
        """
        Prints per-class metrics for a specific model 
        and dataset split ('train', 'val', or 'test').
        """
        split_key, split_name, *_ = self._parse_dataset_type(dataset_type)
        
        model_dict = self.models.get(model_name, {})
        metrics_dict = model_dict.get('metrics', {})
        
        # Fall back to flat metrics if legacy format is encountered
        split_m = metrics_dict.get(
            split_key, 
            metrics_dict if 'precision_per_class' in metrics_dict else {}
        )
        
        if not split_m or 'precision_per_class' not in split_m:
            raise ValueError(
                f"Metrics for split '{split_key}' are not available "
                f"for model '{model_name}'. Run compute_and_store_metrics() first."
            )

        df_per_class = pd.DataFrame({
            'Precision': split_m['precision_per_class'],
            'Recall': split_m['recall_per_class'],
            'F1-Score': split_m['f1_score_per_class'],
            'Support': split_m['support_per_class']
        })
        df_per_class.index = self.labels
        df_per_class.index.name = 'Class / Genre'
        
        print(
            f"\nPER-CLASS METRICS FOR: {model_name} ({split_name} SPLIT)",
            f"| Feature Set: {self.feature_subset.upper()}"
        )
        print("="*45)
        print(df_per_class.to_string())
        print("="*45 + "\n")
        return df_per_class

####################### Save/Load #######################
    def save_models(self, filepath):
            """
            Saves the trained models dictionary to a file using joblib.
            Automatically appends whether it used the 3-sec or 30-sec dataset 
            and the feature subset if not already present in the filepath.
            """
            
            if '3s' in self.dirpath:
                dataset_variant = '3s'
            elif '30s' in self.dirpath:
                dataset_variant = '30s'
            else:
                dataset_variant = 'unknown'
            
            # Insert dataset variant and feature subset into the filename cleanly
            # e.g., './trained_models.pkl' becomes './trained_models_30_sec_full.pkl'
            base_name = f"{dataset_variant}_{self.feature_subset}"
            base_name += f"_rng{self.random_state}"
            pkl_filepath = f"{filepath}/{base_name}.pkl"
            pt_filepath = f"{filepath}/{base_name}_nn_weights.pt"

            models_to_save = {}
            for name, model_dict in self.models.items():
                estimator = model_dict['model']
                models_to_save[name] = model_dict
                # if isinstance(estimator, PyTorchGenreClassifier):
                #     safe_nn_name = name.lower().replace(" ", "_")
                #     pt_filename = f"{base_name}_{safe_nn_name}_nn_weights.pt"
                #     pt_filepath = f"{filepath}/{pt_filename}"

                #     estimator.save_model(pt_filepath)
                #     dummy = PyTorchGenreClassifier(
                #         hidden_dim=estimator.hidden_dim,
                #         lr=estimator.lr,
                #         epochs=estimator.epochs,
                #         batch_size=estimator.batch_size,
                #         num_classes=estimator.num_classes,
                #         patience=estimator.patience,
                #         name=estimator.name
                #     )
                #     dummy.input_dim = estimator.input_dim
                #     dummy.train_losses_ = estimator.train_losses_
                #     dummy.val_losses_ = estimator.val_losses_

                #     models_to_save[name] = {
                #         'model': dummy,
                #         'isTrained': model_dict['isTrained'],
                #         'usesVal': model_dict.get('usesVal', False),
                #         'metrics': model_dict.get('metrics', {}),
                #         'y_pred': model_dict.get('y_pred', None)
                #     }
                # else:
                #     models_to_save[name] = model_dict


            to_save = {
                # 'models': self.models,
                'models': models_to_save,
                'rf_results': self.rf_results,
                'best_k': self.best_k,
                'feature_subset': self.feature_subset,
                'random_state': self.random_state
            }

            joblib.dump(to_save, pkl_filepath)
            # save_dir = Path(filepath)
            # save_dir.mkdir(parents=True, exist_ok=True)
            # dynamic_filename = f"{dataset_variant}_{self.feature_subset}"
            # dynamic_filename += f"_rng{self.random_state}.pkl"
            # dynamic_filepath = save_dir / dynamic_filename
            # joblib.dump(self.models, dynamic_filepath)
            print(
                f"Models and NN container successfully saved to",
                f"\n'{pkl_filepath}'",
                f"NN weights successfully saved to",
                f"\n'{pt_filepath}'")
            return self

    def load_models(self, pkl_filepath, data_dirpath):
        """Loads sklearn metadata and re-links distinct PyTorch network weights."""
        path = Path(pkl_filepath)
        filename = path.stem
        self.dirpath = data_dirpath
        
        match = re.search(r"_rng(\d+)", filename)
        if match: self.random_state = int(match.group(1))

        saved_state = joblib.load(pkl_filepath)
        self.models = saved_state['models']
        self.feature_subset = saved_state['feature_subset']
        self.rf_results = saved_state.get('rf_results', None)
        self.best_k = saved_state.get('best_k', None)

        for name, m_dict in self.models.items():
            estimator = m_dict['model']
            # if isinstance(estimator, PyTorchGenreClassifier):
            #     safe_nn_name = name.lower().replace(" ", "_")
            #     pt_filename = f"{filename}_{safe_nn_name}_nn_weights.pt"
            #     pt_filepath = path.with_name(pt_filename)
                
            #     if pt_filepath.exists():
            #         estimator.load_model(str(pt_filepath))
            #     else:
            #         print(
            #             f"Warning: Weights file '{pt_filename}' not "
            #             f"found for model '{name}'."
            #         )

            # --- Backward Compatibility Migration for Old Metrics Format ---
            if 'metrics' in m_dict and m_dict['metrics']:
                metrics_data = m_dict['metrics']
                # Check if metrics are stored flat (old format) rather than nested by split
                is_flat_old_format = not any(
                    split in metrics_data for split in ['train', 'val', 'test']
                )
                if is_flat_old_format:
                    # Wrap the old flat metrics dictionary into the 'test' slot by default
                    m_dict['metrics'] = {'test': metrics_data}

        self.test_accuracies = {
            name: 1.0 - d['training metrics']['test_error'] 
            for name, d in self.models.items()
            if 'training metrics' in d
        }
            
        try:
            self.load_and_prepare_data()
            print(
                "-> Data splits successfully",
                f"reconstructed for '{filename}'."
            )
        except Exception as e:
            print(f"Warning: Could not automatically reload CSV data: {e}")
            
        print(f"Models successfully loaded from '{pkl_filepath}'.\n")
        return self


####################### Data Analysis #######################
    def plot_confusion_matrices(
            self, model_names=None, data_split='test', displayPlot=False):
        """
        Plots a grid of confusion matrices for a given list of models 
        evaluated on a specified data split ('val', 'test', or 'train').
        """
        if model_names is None: model_names = list(self.models.keys())
        if not isinstance(model_names, list): model_names = [model_names]

        # Select the correct true labels and feature data based on the split choice
        _, split_name, y_true, X_data = self._parse_dataset_type(data_split)

        num_models = len(model_names)
        
        # Dynamically determine grid dimensions (e.g., 2 columns wide, rows as needed)
        cols = 2 if num_models > 1 else 1
        rows = (num_models + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 6 * rows),
            sharex=True, sharey=True,
            gridspec_kw={'hspace': 0.15, 'wspace': 0.15}
        )
        
        # Ensure axes is always a flat array even if 1x1 or 1xD
        axes_flat = np.array([axes]).ravel() if num_models == 1 else axes.ravel()

        def _draw_matrix(model_name, sub_ax):
            # Generate predictions using self.predict() as requested
            y_pred = self.predict(X_data, target_model=model_name)
            
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true, 
                y_pred, 
                display_labels=self.labels,  
                cmap=plt.cm.Blues,
                ax=sub_ax,
                colorbar=False
            )
            sub_ax.set_title(f"{model_name}", fontsize=14)
            plt.setp(sub_ax.get_xticklabels(), rotation=45, ha="right")
            sub_ax.set_ylabel("True Label", fontsize=12)
            sub_ax.set_xlabel("Predicted Label", fontsize=12)
            return disp.im_  

        im1 = None
        for i, name in enumerate(model_names):
            if i < len(axes_flat):
                im = _draw_matrix(name, axes_flat[i])
                if i == 0:
                    im1 = im

        # Turn off any unused empty subplots if grid has extra slots
        for j in range(num_models, len(axes_flat)):
            axes_flat[j].axis('off')

        if im1 is not None:
            fig.colorbar(
                im1, ax=axes_flat[:num_models],
                fraction=0.046, pad=0.04
            )
            
        fig.suptitle(
            f"Confusion Matrices ({split_name} Split)\n" + 
            f"for {self.feature_subset} and random state {self.random_state}",
            fontsize=18, y=0.98
        )
        
        if displayPlot:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_metric_comparison(
            self, metric='Macro F1-Score',
            data_split='test', displayPlot=False):
        """
        Plots a bar chart comparing all models within this classifier instance 
        for a specified metric across a chosen data split ('train', 'val', or 'test').
        """

        _, split_name, y_true, X_data = self._parse_dataset_type(data_split)

        if y_true is None or X_data is None:
            raise ValueError(f"Data for split '{split_name}' is not loaded. Run load_and_prepare_data() first.")

        # 2. Compute the requested metric for each model on the fly
        model_metrics = {}
        metric_lower = metric.lower().strip()

        for name, model_dict in self.models.items():
            # Ensure the model estimator is present
            estimator = model_dict.get('model') if isinstance(model_dict, dict) else model_dict
            if estimator is None:
                continue

            # Generate predictions using self.predict()
            y_pred = self.predict(X_data, target_model=name)

            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            error = 1.0 - acc
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            if 'error' in metric_lower:
                val = error
            elif 'bal' in metric_lower or 'balanced' in metric_lower:
                val = bal_acc
            elif 'acc' in metric_lower:
                val = acc
            elif 'f1' in metric_lower:
                val = f1
            elif 'prec' in metric_lower:
                val = prec
            elif 'rec' in metric_lower:
                val = rec
            else:
                raise(ValueError(
                    'metric must be err, bal acc,' \
                    'acc, f1, prec, or rec'))

            model_metrics[name] = val

        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(model_metrics.keys())
        values = list(model_metrics.values())

        colors = [
            'skyblue', 'salmon', 'lightgreen', 'plum',
            'orange', 'orchid', 'khaki', 'coral']
        bars = ax.bar(names, values, color=colors[:len(names)], width=0.6)

        if "30s" in self.dirpath:
            t = "30s"
        else:
            t = "3s"
        ax.set_title(
            f'{metric} Comparison ({split_name} Split) | ' +
            f'Config: {self.feature_subset} + {t} + rng{self.random_state}',
            fontsize=14)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)

        max_val = max(values) if values else 1.0
        if any(term in metric_lower for term in [
                'error', 'accuracy', 'balanced', 'f1', 'precision', 'recall', 'score']):
            ax.set_ylim(0, min(1.15, max_val * 1.15))
        else:
            ax.set_ylim(0, max_val * 1.15)

        plt.xticks(rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.02, f"{yval:.3f}",
                ha='center', va='bottom', fontsize=10
            )

        fig.tight_layout()
        if displayPlot:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_resource_comparison(
            self, metric='train time',
            memory_unit='KB', displayPlot=False):
        """
        Plots a bar chart comparing resource metrics (Train Time, Model Memory, File Size) 
        across all models within this classifier instance.
        
        Parameters:
        -----------
        metric : str
            The resource metric to plot: 'train time', 'model memory', or 'file size'.
        memory_unit : str
            Units for memory/file size: 'KB' or 'MB'.
        displayPlot : bool
            Whether to display the plot immediately via plt.show().
        """

        metric_lower = metric.lower().strip()
        resource_values = {}

        # 1. Gather resource values for each model
        for name, model_dict in self.models.items():
            estimator = model_dict.get('model') if isinstance(model_dict, dict) else model_dict
            if estimator is None:
                continue

            if 'time' in metric_lower or 'train' in metric_lower:
                # Update this line to check 'training metrics' first
                train_m = model_dict.get('training metrics', {})
                val = train_m.get('training_time_seconds', 0.0)
                
                # Fallback check to legacy 'metrics' if needed
                if val == 0.0:
                    m = model_dict.get('metrics', {})
                    val = m.get('training_time_seconds', 0.0)
                    
                ylabel = "Training Time (seconds)"
                title_prefix = "Training Time"
            elif 'memory' in metric_lower:
                # Calculate model memory using pickle in-memory size
                model_bytes = pickle.dumps(estimator)
                size_kb = len(model_bytes) / 1024
                val = size_kb / 1024 if memory_unit.upper() == 'MB' else size_kb
                unit_label = memory_unit.upper()
                ylabel = f"Model Memory ({unit_label})"
                title_prefix = f"Model Memory ({unit_label})"
            elif 'file' in metric_lower:
                # Estimate or check file size on disk for the saved pkl
                # Uses the naming convention generated by save_models()
                dataset_variant = '3s' if '3s' in self.dirpath else ('30s' if '30s' in self.dirpath else 'unknown')
                pkl_name = f"{dataset_variant}_{self.feature_subset}_rng{self.random_state}.pkl"
                # Search common directories or use self.dirpath parent
                possible_paths = [
                    Path(self.dirpath) / pkl_name,
                    Path('./trained_models') / pkl_name,
                    Path('.') / pkl_name
                ]
                
                size_kb = 0
                for path in possible_paths:
                    if path.exists():
                        size_kb = os.path.getsize(path) / 1024
                        break
                
                val = size_kb / 1024 if memory_unit.upper() == 'MB' else size_kb
                unit_label = memory_unit.upper()
                ylabel = f"File Size ({unit_label})"
                title_prefix = f"Saved File Size ({unit_label})"
            else:
                raise ValueError(f"Unrecognized resource metric: '{metric}'. Use 'train time', 'model memory', or 'file size'.")

            resource_values[name] = val

        if not resource_values:
            print("No models found to plot resource comparison.")
            return None, None

        # 2. Create the bar chart visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(resource_values.keys())
        values = list(resource_values.values())

        colors = ['coral', 'gold', 'lightseagreen', 'skyblue', 'plum', 'salmon']
        bars = ax.bar(names, values, color=colors[:len(names)], width=0.6)

        
        if "30s" in self.dirpath:
            t = "30s"
        else:
            t = "3s"
        ax.set_title(
            f'{title_prefix} Comparison | ' +
            f'Config: {self.feature_subset} + {t} + rng{self.random_state}',
            fontsize=14)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        max_val = max(values) if values else 1.0
        ax.set_ylim(bottom=0, top=max_val * 1.20)  # Gives 20% headroom above the highest bar

        plt.xticks(rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate bar heights cleanly with appropriate decimal formatting
        for bar in bars:
            yval = bar.get_height()
            format_str = f"{yval:.2f}" if yval < 10 else f"{yval:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + (max(values) * 0.02 if max(values) > 0 else 0.02), 
                format_str,
                ha='center', va='bottom', fontsize=10
            )

        fig.tight_layout()
        if displayPlot:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_per_class_heatmap(
            self, metric_key='f1_score_per_class',
            data_split='test', displayPlot=False):
        """
        Creates a heatmap showing per-class scores across all models 
        contained within this classifier instance for a specified data split.
        
        metric_key options: 'precision', 'recall', 'f1_score' (or 'precision_per_class', etc.)
        """
        # 1. Determine the target data split using the helper function
        _, split_name, y_true, X_data = self._parse_dataset_type(data_split)

        if y_true is None or X_data is None:
            raise ValueError(f"Data for split '{split_name}' is not loaded. Run load_and_prepare_data() first.")

        matrix_data = []
        model_labels = []
        genre_labels = self.labels

        # Clean metric_key mapping for scikit-learn's precision_recall_fscore_support
        base_metric = metric_key.replace('_per_class', '').strip().lower()

        # 2. Iterate through each model in the instance and compute per-class metrics on the fly
        for model_name, model_dict in self.models.items():
            estimator = model_dict.get('model') if isinstance(model_dict, dict) else model_dict
            if estimator is None:
                continue

            # Generate predictions using self.predict()
            y_pred = self.predict(X_data, target_model=model_name)

            # Calculate per-class metrics via precision_recall_fscore_support
            class_pc, class_r, class_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

            if 'prec' in base_metric:
                scores = class_pc
            elif 'rec' in base_metric:
                scores = class_r
            else:
                scores = class_f1  # Default to F1

            matrix_data.append(scores)
            model_labels.append(model_name)

        if not matrix_data:
            print("No models found to plot per-class heatmap.")
            return None, None

        matrix = np.array(matrix_data)  # Shape: (number of models, number of genres)

        # 3. Create the heatmap figure
        fig, ax = plt.subplots(figsize=(10, max(4, len(model_labels) * 0.6)))
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=0.0, vmax=1.0)

        # Configure axes
        ax.set_xticks(np.arange(len(genre_labels)))
        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_xticklabels(genre_labels, rotation=45, ha="right")
        ax.set_yticklabels(model_labels, fontsize=10)

        # Add text annotations inside heatmap blocks for readability
        for i in range(len(model_labels)):
            for j in range(len(genre_labels)):
                val = matrix[i, j]
                text_color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

        metric_title = base_metric.replace('_', ' ').upper()
        if "30s" in self.dirpath:
            t = "30s"
        else:
            t = "3s"
        ax.set_title(
            f'Per-Class {metric_title} Heatmap ({split_name} Split) | ' +
            f'Config: {self.feature_subset} + {t} + rng{self.random_state}',
            fontsize=14, pad=15)
        ax.set_xlabel("GTZAN Genre Classes", fontsize=12)
        ax.set_ylabel("Models", fontsize=12)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        
        if displayPlot:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax
####################### Predict Function #######################
    def predict(self, X, target_model=None):
        """
        Takes unnormalized X data (DataFrame or numpy array) with required
        features, applies feature subset filtering/preprocessing,
        and predicts labels.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Input features.
        target_model : str, optional
            Specific model name. If provided, returns a pandas Series
            of shape (n,).
            If None, returns a dictionary of {model_name: pd.Series
            of shape (n,)}.
            
        Returns:
        --------
        pd.Series or dict of pd.Series
            Predicted genre labels as pandas Series of shape (n,).
        """
        # 2. Helper function to generate predictions wrapped in a pandas Series
        def _get_prediction_series(estimator):
            raw_preds = estimator.predict(X)
            # Ensure output is a pandas Series of shape (n,) keeping original DataFrame indices if available
            index = X.index if isinstance(X, pd.DataFrame) else None
            return pd.Series(raw_preds, index=index, name='predicted_genre')

        # 3. Handle single model vs all models return behavior
        if target_model is not None:
            if target_model not in self.models:
                raise ValueError(
                    f"Model '{target_model}' not found." + 
                    f"Available models: {list(self.models.keys())}")
            
            model_item = self.models[target_model]
            # Extract pipeline estimator safely whether stored raw or inside a metrics dictionary wrapper
            pipeline = model_item['model'] if isinstance(
                model_item, dict) and 'model' in model_item else model_item
            return _get_prediction_series(pipeline)
        else:
            predictions_dict = {}
            for name, model_item in self.models.items():
                pipeline = model_item['model'] if isinstance(
                    model_item, dict) and 'model' in model_item else model_item
                predictions_dict[name] = _get_prediction_series(pipeline)
            return predictions_dict