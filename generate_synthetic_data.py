# Generate Synthetic DNA Methylation Data for Model Training Demo
import numpy as np
import pandas as pd
import h5py
import os

def generate_synthetic_methylation_data(n_samples=500, n_features=50000, save_path='./model/data/'):
    """
    Generate synthetic DNA methylation data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of CpG sites (features) 
        save_path: Path to save the generated data
    """
    
    # Create directories
    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Generate synthetic methylation data (beta values between 0 and 1)
    np.random.seed(42)
    
    # Training data
    train_samples = int(n_samples * 0.8)
    test_samples = n_samples - train_samples
    
    # Generate realistic methylation patterns
    # Control samples: more variation, centered around 0.5
    n_control = train_samples // 3
    control_data = np.random.beta(2, 2, size=(n_control, n_features))
    
    # MCI samples: slight hypomethylation in some regions
    n_mci = train_samples // 3
    mci_data = np.random.beta(1.8, 2.2, size=(n_mci, n_features))
    
    # Alzheimer's samples: more pronounced hypomethylation
    n_alzheimers = train_samples - n_control - n_mci
    alzheimers_data = np.random.beta(1.5, 2.5, size=(n_alzheimers, n_features))
    
    # Combine training data
    train_data = np.vstack([control_data, mci_data, alzheimers_data])
    train_labels = ['control'] * n_control + ['MCI'] * n_mci + ["Alzheimer's"] * n_alzheimers
    
    # Generate test data similarly
    n_control_test = test_samples // 3
    n_mci_test = test_samples // 3
    n_alzheimers_test = test_samples - n_control_test - n_mci_test
    
    control_test = np.random.beta(2, 2, size=(n_control_test, n_features))
    mci_test = np.random.beta(1.8, 2.2, size=(n_mci_test, n_features))
    alzheimers_test = np.random.beta(1.5, 2.5, size=(n_alzheimers_test, n_features))
    
    test_data = np.vstack([control_test, mci_test, alzheimers_test])
    test_labels = ['control'] * n_control_test + ['MCI'] * n_mci_test + ["Alzheimer's"] * n_alzheimers_test
    
    # Create sample IDs
    train_ids = [f'TRAIN_{i:04d}' for i in range(train_samples)]
    test_ids = [f'TEST_{i:04d}' for i in range(test_samples)]
    
    # Create mapping DataFrames
    train_mapping = pd.DataFrame({
        'sample_id': train_ids,
        'disease_state': train_labels,
        'series_id': ['GSE123456'] * train_samples,
        'sex': np.random.choice(['M', 'F'], train_samples),
        'age': np.random.normal(75, 10, train_samples).clip(50, 95)
    })
    
    test_mapping = pd.DataFrame({
        'sample_id': test_ids,
        'disease_state': test_labels,
        'series_id': ['GSE123456'] * test_samples,
        'sex': np.random.choice(['M', 'F'], test_samples),
        'age': np.random.normal(75, 10, test_samples).clip(50, 95)
    })
    
    # Create feature names (CpG sites)
    feature_names = [f'cg{i:08d}' for i in range(n_features)]
    
    # Save training data
    print(f"Saving training data to {train_path}")
    
    # Save as CSV
    train_df = pd.DataFrame(train_data, columns=feature_names, index=train_ids)
    train_df.to_csv(os.path.join(train_path, 'methylation.csv'))
    train_mapping.to_csv(os.path.join(train_path, 'idmap.csv'), index=False)
    
    # Save as HDF5 for faster loading
    with h5py.File(os.path.join(train_path, 'methylation.h5'), 'w') as f:
        f.create_dataset('methylation', data=train_data)
        f.create_dataset('sample_ids', data=[s.encode() for s in train_ids])
        f.create_dataset('feature_names', data=[s.encode() for s in feature_names])
    
    # Save test data
    print(f"Saving test data to {test_path}")
    
    test_df = pd.DataFrame(test_data, columns=feature_names, index=test_ids)
    test_df.to_csv(os.path.join(test_path, 'methylation.csv'))
    test_mapping.to_csv(os.path.join(test_path, 'idmap.csv'), index=False)
    
    with h5py.File(os.path.join(test_path, 'methylation.h5'), 'w') as f:
        f.create_dataset('methylation', data=test_data)
        f.create_dataset('sample_ids', data=[s.encode() for s in test_ids])
        f.create_dataset('feature_names', data=[s.encode() for s in feature_names])
    
    print(f"Generated synthetic data:")
    print(f"  Training: {train_samples} samples x {n_features} features")
    print(f"  Test: {test_samples} samples x {n_features} features")
    print(f"  Classes: Control ({n_control + n_control_test}), MCI ({n_mci + n_mci_test}), Alzheimer's ({n_alzheimers + n_alzheimers_test})")

if __name__ == "__main__":
    generate_synthetic_methylation_data(
        n_samples=500, 
        n_features=10000,  # Reduced for faster training
        save_path='./model/data/'
    )