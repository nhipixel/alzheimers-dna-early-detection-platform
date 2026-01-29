from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib, os
import numpy as np

class XGBoostModel:
    def __init__(self, params=None, use_feature_selection=True, n_features=10000):
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'learning_rate': 0.05,
                'max_depth': 8,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'gamma': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
                'early_stopping_rounds': 50,
                'verbose': False
            }
        self.params = params
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features

        if use_feature_selection:
            feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, 50000))
            
            preprocessor = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('feature_selection', feature_selector),
                ('scaler', RobustScaler())
            ])
        else:
            preprocessor = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

        # Define Final Model Pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**self.params))
        ])

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            
            preprocessor = Pipeline(steps=self.model.steps[:-1])
            preprocessor.fit(X_train, y_train)
            X_val_transformed = preprocessor.transform(X_val)
            
            classifier = self.model.steps[-1][1]
            X_train_transformed = preprocessor.transform(X_train)
            classifier.fit(
                X_train_transformed, y_train,
                eval_set=[(X_train_transformed, y_train), (X_val_transformed, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Return prediction probabilities"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Enhanced evaluation with multiple metrics"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def search_cv(self, param_grid, X_train, y_train, cv=5, scoring='f1_weighted'):
        """Enhanced grid search with stratified CV"""
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            self.model, 
            param_grid, 
            cv=cv_splitter, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance from trained model"""
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importance = self.model.named_steps['classifier'].feature_importances_
            
            if feature_names is not None:
                # Handle feature selection - get selected features
                if hasattr(self.model.named_steps['preprocessor'], 'named_steps'):
                    if 'feature_selection' in self.model.named_steps['preprocessor'].named_steps:
                        selector = self.model.named_steps['preprocessor'].named_steps['feature_selection']
                        selected_features = selector.get_support(indices=True)
                        feature_names = [feature_names[i] for i in selected_features]
                
                return list(zip(feature_names[:len(importance)], importance))
            else:
                return importance
        return None
    
    def save_model(self, path):
        """Save model with enhanced metadata"""
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, 'xgboost_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save model metadata
        metadata = {
            'params': self.params,
            'use_feature_selection': self.use_feature_selection,
            'n_features': self.n_features,
            'model_type': 'XGBoostModel'
        }
        metadata_path = os.path.join(path, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        return model_path
    
    def load_model(self, path):
        """Load model with metadata"""
        model_path = os.path.join(path, 'xgboost_model.pkl')
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = os.path.join(path, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.params = metadata.get('params', self.params)
            self.use_feature_selection = metadata.get('use_feature_selection', True)
            self.n_features = metadata.get('n_features', 10000)
        
        return self.model