import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def preprocess_time_series(parquet_dir, output_dir, batch_size=100):
    os.makedirs(output_dir, exist_ok=True)

    # Traverse subdirectories and collect parquet files
    patient_dirs = [d for d in os.listdir(parquet_dir) if os.path.isdir(os.path.join(parquet_dir, d))]

    for i in range(0, len(patient_dirs), batch_size):
        batch_dirs = patient_dirs[i:i + batch_size]
        
        for patient_dir in batch_dirs:
            patient_id = patient_dir.split('=')[-1]  # Extract patient ID from the folder name
            patient_path = os.path.join(parquet_dir, patient_dir, 'part-0.parquet')  # Path to the parquet file
            
            if os.path.exists(patient_path):
                # Load and process time-series data
                df = pd.read_parquet(patient_path)
                
                # Specify the time-series features to extract
                time_series_features = ['X', 'Y', 'Z', 'enmo', 'anglez', 'non-wear_flag', 'light', 'battery_voltage']
                
                # Filter the dataframe for relevant features
                df = df[time_series_features]
                
                # Convert to numpy array (with float32 for memory efficiency)
                ts_data = df.to_numpy(dtype=np.float32)
                
                # Save processed data as .npy file
                np.save(os.path.join(output_dir, f"{patient_id}.npy"), ts_data)


# Data Preprocessing
def data_pre_processing(file_name):
    df = pd.read_csv(file_name)
    TARGET_COLS = ["sii"]
    
    FEATURE_COLS = [
        "Basic_Demos-Age",
        "Basic_Demos-Sex",
        "CGAS-CGAS_Score",
        "Physical-BMI",
        "Physical-Height",
        "Physical-Weight",
        "Physical-Waist_Circumference",
        "Physical-Diastolic_BP",
        "Physical-HeartRate",
        "Physical-Systolic_BP",
        "Fitness_Endurance-Max_Stage",
        "Fitness_Endurance-Time_Mins",
        "Fitness_Endurance-Time_Sec",
        "FGC-FGC_CU",
        "FGC-FGC_CU_Zone",
        "FGC-FGC_GSND",
        "FGC-FGC_GSND_Zone",
        "FGC-FGC_GSD",
        "FGC-FGC_GSD_Zone",
        "FGC-FGC_PU",
        "FGC-FGC_PU_Zone",
        "FGC-FGC_SRL",
        "FGC-FGC_SRL_Zone",
        "FGC-FGC_SRR",
        "FGC-FGC_SRR_Zone",
        "FGC-FGC_TL",
        "FGC-FGC_TL_Zone",
        "BIA-BIA_Activity_Level_num",
        "BIA-BIA_BMC",
        "BIA-BIA_BMI",
        "BIA-BIA_BMR",
        "BIA-BIA_DEE",
        "BIA-BIA_ECW",
        "BIA-BIA_FFM",
        "BIA-BIA_FFMI",
        "BIA-BIA_FMI",
        "BIA-BIA_Fat",
        "BIA-BIA_Frame_num",
        "BIA-BIA_ICW",
        "BIA-BIA_LDM",
        "BIA-BIA_LST",
        "BIA-BIA_SMM",
        "BIA-BIA_TBW",
        "PAQ_A-PAQ_A_Total",
        "PAQ_C-PAQ_C_Total",
        "SDS-SDS_Total_Raw",
        "SDS-SDS_Total_T",
        "PreInt_EduHx-computerinternet_hoursday"]

    data = df[FEATURE_COLS]
    target = df[TARGET_COLS].fillna(-1).values.flatten()
    patient_ids = df["id"]

    iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
    data_imputed = pd.DataFrame(iterative_imputer.fit_transform(data), columns=FEATURE_COLS)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)

    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=5)
    label_prop_model.fit(data_scaled, target)
    target_imputed = label_prop_model.transduction_

    return pd.DataFrame(data_scaled, columns=FEATURE_COLS), target_imputed, patient_ids


# Time-Series Dataset
# class LazyPatientDataset(Dataset):
#     def __init__(self, patient_ids, static_features, labels, ts_dir, pad_value=0.0):
#         self.patient_ids = patient_ids
#         self.static_features = static_features
#         self.labels = labels
#         self.ts_dir = ts_dir
#         self.pad_value = pad_value

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         patient_id = self.patient_ids[idx]
#         static_feat = torch.tensor(self.static_features.iloc[idx].to_numpy(), dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)

#         ts_filepath = os.path.join(self.ts_dir, f"{patient_id}.npy")
#         if os.path.exists(ts_filepath):
#             time_series = torch.tensor(np.load(ts_filepath), dtype=torch.float32)
#         else:
#             time_series = None

#         return time_series, static_feat, label

class LazyPatientDataset(Dataset):
    def __init__(self, patient_ids, static_features, labels, ts_dir, pad_value=0.0):
        self.patient_ids = patient_ids
        self.static_features = static_features
        self.labels = labels
        self.ts_dir = ts_dir
        self.pad_value = pad_value

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        static_feat = torch.tensor(self.static_features.iloc[idx].to_numpy(), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        ts_filepath = os.path.join(self.ts_dir, f"{patient_id}.npy")
        if os.path.exists(ts_filepath):
            time_series = torch.tensor(np.load(ts_filepath), dtype=torch.float32)
        else:
            time_series = None

        return time_series, static_feat, label  # Return only 3 elements

    
def collate_fn(batch):
    time_series, static_features, labels = zip(*batch)  # Unpack into 3 variables

    # Filter out None time-series entries and their corresponding static features and labels
    valid_indices = [i for i, ts in enumerate(time_series) if ts is not None]
    valid_time_series = [time_series[i] for i in valid_indices]
    static_features = [static_features[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    # Pad the time series if there are valid ones
    if valid_time_series:
        time_series_padded = pad_sequence(valid_time_series, batch_first=True, padding_value=0)
        
        # Create mask: True for non-zero elements along the last dimension
        ts_masks = (time_series_padded != 0).any(dim=-1).float()
    else:
        # Handle empty batch gracefully
        time_series_padded = torch.zeros(len(batch), 1, valid_time_series[0].size(-1))
        ts_masks = torch.zeros(len(batch), 1, dtype=torch.float)

    # Stack static features and labels
    static_features = torch.stack(static_features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return time_series_padded, static_features, labels, ts_masks


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, max_len):
        super(TimeSeriesTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))  # Shape: [1, max_len, embed_dim]
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first = True),
            num_layers,
        )
        self.fc = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        seq_len = x.size(1)
    
        # Project to embedding if necessary
        x = self.input_projection(x)  # [batch_size, seq_len, embed_dim]
    
        # Add positional encoding for the current sequence length
        x = x + self.positional_encoding[:, :seq_len, :]  # [batch_size, seq_len, embed_dim]
    
        # Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=mask)  # [batch_size, seq_len, embed_dim]
    
        # Pool over time (mean pooling)
        x = x.mean(dim=1)  # [batch_size, embed_dim]
    
        return self.fc(x)  # [batch_size, embed_dim]



class StaticFeatureEmbedder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)


class FeedForwardClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class CombinedModel(nn.Module):
    def __init__(self, time_series_model, static_model, classifier, embed_dim):
        super().__init__()
        self.time_series_model = time_series_model
        self.static_model = static_model
        self.classifier = classifier
        self.embed_dim = embed_dim

    def forward(self, time_series, static_features, mask=None):
        ts_embedding = self.time_series_model(time_series, mask)
        static_embedding = self.static_model(static_features)
        combined = torch.cat([ts_embedding, static_embedding], dim=1)
        return self.classifier(combined)


# Training
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for time_series, static_features, labels, ts_masks in dataloader:
        time_series, static_features, labels, ts_masks = (
            time_series.to(device),
            static_features.to(device),
            labels.to(device),
            ts_masks.to(device),
        )

        optimizer.zero_grad()
        outputs = model(time_series, static_features, ts_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    import os
    from torch.nn.utils.rnn import pad_sequence

    # File paths and parameters
    TS_DIR = "./dataset/series_train.parquet"  # Directory where time-series data is stored
    CSV_FILE = "./dataset/train.csv"  # Path to CSV file with static features and target
    PROCESSED_TS_DIR = './dataset/processed_time_series'
    
    print("Preprocessing time-series data...")
    preprocess_time_series(TS_DIR, PROCESSED_TS_DIR)
    
    # Step 1: Preprocess the data
    print("Pre process static data")
    static_features, labels, patient_ids = data_pre_processing(CSV_FILE)
    
    
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIME_SERIES_INPUT_DIM = 8  # Adjust based on your data
    STATIC_FEATURE_DIM = static_features.shape[1]  # Adjust based on your data
    EMBED_DIM = 16
    NUM_CLASSES = 4
    NUM_HEADS = 4
    NUM_LAYERS = 2

    # Step 2: Initialize Dataset and Dataloader
    dataset = LazyPatientDataset(
        patient_ids=patient_ids,
        static_features=static_features,
        labels=labels,
        ts_dir=processed_ts_dir,
    )
    print("Creating dataset and dataloader...")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Step 3: Initialize Models
    time_series_model = TimeSeriesTransformer(
        input_dim=TIME_SERIES_INPUT_DIM,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=800000,  # Adjust based on your max sequence length
    )

    static_model = StaticFeatureEmbedder(
        input_dim=STATIC_FEATURE_DIM,
        embed_dim=EMBED_DIM,
    )

    classifier = FeedForwardClassifier(
        input_dim=2 * EMBED_DIM,
        num_classes=NUM_CLASSES,
    )

    model = CombinedModel(
        time_series_model=time_series_model,
        static_model=static_model,
        classifier=classifier,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    # Step 4: Define Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Step 5: Train the Model
    for epoch in range(EPOCHS):
        loss = train_model(model, dataloader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss:.4f}")

    # Step 6: Save the Model
    model_path = "./combined_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

