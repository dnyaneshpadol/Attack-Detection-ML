import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

def load_nsl_kdd(path):
    df = pd.read_csv(path, names=COLUMNS, header=None)
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    return df

def exploratory_data_analysis(df):
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    print(df.info())
    print(df.describe())
    print("\nClass distribution:\n", df['label'].value_counts())

def plot_class_distribution(df, output_dir=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Class Distribution (All Attack Types)")
    
    if output_dir:
        plt.savefig(f"{output_dir}/class_distribution.png")
    plt.show()

def prepare_data(df):
    df = df.copy()
    
    y = df["label"]  # multi-class
    
    X = df.drop(columns=["label"])
    
    categorical_cols = ["protocol_type", "service", "flag"]
    X = pd.get_dummies(X, columns=categorical_cols)
    
    return X, y

def get_feature_lists(X):
    return list(X.columns)