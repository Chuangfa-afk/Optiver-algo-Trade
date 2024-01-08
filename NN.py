# %% Import the Necesary libraries
import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
import pandas as pd
import gc
import numpy as np
import logging
import matplotlib.pyplot as plt
import math;
import seaborn as sns;

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__);

# %% impoprt the data
df = pd.read_csv('train.csv')

# %% FUNCTION TO check and change the datatype in order to reduce the memory
def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    
    # 📏 Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # 🔄 Iterate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    # Provide memory optimization information if 'verbose' is True
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    # Return the DataFrame with optimized memory usage
    return df

# %% Apply the memory reduciton 

df = reduce_mem_usage(df, 1)



# %% NORMALIZATION TO THE TARGET
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(-1, 1))

# # Normalize the 'target' column and replace it in the original DataFrame
# df['target'] = scaler.fit_transform(df['target'].values.reshape(-1, 1))


# %% FUnction fill with mean for target, prices and volumne (size)

def fill_missing_with_stock_mean(df, target_column='target', stock_id_column='stock_id'):
    """
    Fills missing values in the target column based on the mean of the corresponding stock ID.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the column with missing values to fill. Default is 'target'.
    stock_id_column (str): The name of the column containing the stock IDs. Default is 'stock_id'.

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """

    # Ensure the stock_id_column is in the DataFrame
    if stock_id_column not in df.columns:
        raise ValueError(f"Column '{stock_id_column}' not found in DataFrame.")

    # Calculate mean for each stock
    means = df.groupby(stock_id_column)[target_column].mean()

    # Function to apply to each row
    def fill_with_mean(row):
        if pd.isna(row[target_column]):
            return means[row[stock_id_column]]
        else:
            return row[target_column]

    # Apply function across the DataFrame
    df[target_column] = df.apply(fill_with_mean, axis=1)
    return df

def fill_columns_with_one(df, columns):
    """
    Fill specified columns in a DataFrame with the value 1.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list of str): The names of the columns to fill with 1.

    Returns:
    pd.DataFrame: The modified DataFrame with specified columns filled with 1.
    """
    for column in columns:
        if column in df.columns:
            df[column] = 1
        else:
            print(f"Column '{column}' not found in DataFrame.")
    return df

import pandas as pd

def fill_missing_with_stock_median(df, columns, stock_id_column='stock_id'):
    """
    Fills missing values in specified columns based on the median of the corresponding stock ID.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): List of columns for which to fill missing values.
    stock_id_column (str): The name of the column containing the stock IDs.

    Returns:
    pd.DataFrame: DataFrame with missing values filled.
    """

    # Calculate median for each stock for the specified columns
    medians = df.groupby(stock_id_column)[columns].median()

    # Fill missing values with corresponding median
    for column in columns:
        if column in df.columns:
            # Define a function for filling with median
            def fill_with_median(row):
                if pd.isna(row[column]):
                    return medians.loc[row[stock_id_column], column]
                else:
                    return row[column]

            # Apply the function to the DataFrame
            df[column] = df.apply(fill_with_median, axis=1)
        else:
            print(f"Column '{column}' not found in DataFrame.")

    return df



def add_info_columns(raw_df):

    df = raw_df
    
    df['imbalance_ratio'] = df['imbalance_size'] / (df['matched_size'] + 1.0e-8)
    df["imbalance"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    
    df['ordersize_imbalance'] = (df['bid_size']-df['ask_size']) / ((df['bid_size']+df['ask_size'])+1.0e-8)
    df['matching_imbalance'] = (df['imbalance_size']-df['matched_size']) / ((df['imbalance_size']+df['matched_size'])+1.0e-8) 
    gc.collect()  
    return df

def sort_by_time(df):
    df = df.sort_values(by=['date_id', 'seconds_in_bucket'])
    gc.collect()
    return df

def inspect_columns(df):
    
    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 2),
        'max': df.max(),
        'min': df.min(),
        'median': df.median(),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })
    return result

# %% Data Understanding
#Inspect every column
print(inspect_columns(df))

# Correlation map to see the connection between each variable
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# %% Imputation for the dataset
df = fill_missing_with_stock_mean(df)
df = fill_columns_with_one(df, ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"])
df = fill_missing_with_stock_median(df, ['matched_size', 'imbalance_size'])
df = add_info_columns(df)
# %% CHECK for NA, we should see a FALSE meaning there is no NAN values
df.isna().any().any()
# %% DROP ROW_ID
df = df.drop('row_id', axis=1)


# %% CHECK THE shape of the df
df.shape

# %% SORT THE DATA BY TIme

df = sort_by_time(df)
print(df)
# %% DATA splitting function with lookback argument

# def split_data(stock, lookback):
#     data_raw = stock
#     data = []


#     # create a list of time series of window size lookback
#     for index in range(len(data_raw) - lookback):
#         data.append(data_raw[index: index + lookback])

#     data = np.array(data)
#     # The last 20% of the data as test set
#     test_set_size = int(np.round(0.2*data.shape[0]))
#     train_set_size = data.shape[0] - test_set_size

#     # Assign the first lookback -1 time steps as input x
#     x_train = data[:train_set_size,:-1,:]
#     x_test = data[train_set_size:,:-1,:]

#     # The last time step as target y (the 'target' column)
#     y_train = data[:train_set_size,-1,11]  # 11 is the index of the 'target' column
#     y_test = data[train_set_size:,-1,11]

#     return [x_train, y_train, x_test, y_test]

def split_data(stock, lookback, target_col_index):
    data_raw = stock  # Assuming stock is already a NumPy array
    data = []

    # Create a list of time series of window size lookback
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    # The last 20% of the data as test set
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    # Assign the first lookback -1 time steps as input x
    x_train = data[:train_set_size, :-1, :]
    x_test = data[train_set_size:, :-1, :]

    # The last time step as target y (the target column)
    y_train = data[:train_set_size, -1, target_col_index]
    y_test = data[train_set_size:, -1, target_col_index]

    return [x_train, y_train, x_test, y_test]



# %% EXTRACT the only train features in other words, exclude stock_id, date_id and seconds in bucket for training

feature_columns = [col for col in df.columns if col not in ['stock_id', 'date_id', 'seconds_in_bucket']]

# Selecting the relevant columns
data_for_model = df[feature_columns]

# %% Remove the target variable and put it to the end column
cols = list(data_for_model.columns)
cols.append(cols.pop(cols.index('target')))  # Remove 'target' and add it at the end

# Now reorder the dataframe
data_for_model = data_for_model[cols]

for i, column in enumerate(data_for_model.columns):
    print(f"Index {i}: Column name '{column}'")

# %% Converting to numpy array as required by split_data function
data_for_model_np = data_for_model.to_numpy()


# %% SPLIT the data and see the shape of x, y - train, test
# Lookback period
lookback = 20

# Splitting the data
x_train, y_train, x_test, y_test = split_data(data_for_model_np, lookback, target_col_index=-1)
gc.collect()
# Printing the shapes
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# %% Convert the training and testing data into Tensor
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)
gc.collect()

# %% DATALoader
from torch.utils.data import TensorDataset, DataLoader
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

batch_size = 512  # You can adjust this

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = TensorDataset(x_test, y_test)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

# %% Define models -> RNN, LSTM, GRU

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, time_step, input_size)
        # r_out shape: (batch, time_step, output_size)
        # h_n shape: (n_layers, batch, hidden_size)
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        r_out, hidden = self.rnn(x, hidden)  
        out = self.fc(r_out[:, -1, :])
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.01):
        """
        Initialize the model by setting up the layers.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        ##TODO: Define LSTM model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        ## TODO: Define the final, fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # x (batch_size, seq_length, input_dim)
        # hidden (num_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)

        # Here we initialize the hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  #short term memory
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()  #long term memory

        ## Get the outputs and the new hidden state from the lstm
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        ## apply Dropout
        out = self.dropout(out)
        ## Put out through the fully-connected layer
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the GRU model layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, nhead, num_layers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


# %%SELECT WHICH MODEL to train??? Just set it to TRUE
RNN = False
LSTM_MODEL = True
GRU_MODEL = False
TRANSFORMER = False
LGBM = False ### NOT disclosure yet because the competition is still active

# Hyperparameters
INPUT_SIZE = 17  # number of features
HIDDEN_SIZE = 32  # you can change this
NUM_LAYERS = 3    # number of RNN layers
OUTPUT_SIZE = 1   # predicting a single value
LEARNING_RATE = 0.001;
EPOCHS = 1 # HYPER -> 582 But be carefull, it takes more than 10 hours to run it
L2_REG = 0.00001 # Tunned
DROPOUT = 0.01

if RNN: 
    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
elif LSTM_MODEL:
    model = LSTM(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=OUTPUT_SIZE, num_layers=NUM_LAYERS)
elif GRU_MODEL:
    model = GRU(input_dim=INPUT_SIZE, hidden_dim=HIDDEN_SIZE, output_dim=OUTPUT_SIZE, num_layers=NUM_LAYERS)
elif TRANSFORMER:
    model = TransformerModel(input_dim=INPUT_SIZE, output_dim=OUTPUT_SIZE, hidden_dim=HIDDEN_SIZE, nhead=8, num_layers=NUM_LAYERS, dropout=0.01)
else:
    print("NO training is happining... Try to set one boolean above to be True")
    

# %% Train and validate
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)  # learning rate can be adjusted

# %% Define training and validation fucntion
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, print_every=100):
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


        # Validation phase
        val_loss = validate_model(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_loader)
    return val_loss

train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# %% Training the model with hyperparameter tunning
import itertools
import os

HYPER = False

def hyperparameter_tuning(model, train_loader, val_loader, criterion, optimizer, num_epochs, hyperparameters):
    best_val_loss = float('inf')
    best_hyperparams = None

    for params in itertools.product(*hyperparameters.values()):
        # Set the hyperparameters
        input_size, hidden_size, num_layers, output_size, learning_rate = params
        model = SimpleRNN(input_size, hidden_size, num_layers, output_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, print_every=100)

        # Validate the model
        val_loss = validate_model(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_hyperparams = params

    # Save the best model
    save_path = 'best_model.pth'
    torch.save(model.state_dict(), save_path)

    return best_hyperparams, best_val_loss, save_path

# Define hyperparameters to search
hyperparameters = {
    'input_size': [13], # 17
    'hidden_size': [32, 64, 128], # 32
    'num_layers': [3,4,5,6], # 3
    'output_size': [1], # 1
    'learning_rate': [0.0001, 0.01, 0.001] # 0.001
}

# Perform hyperparameter tuning
if HYPER:
    best_hyperparams, best_val_loss, best_model_path = hyperparameter_tuning(model, train_loader, val_loader, criterion, optimizer, EPOCHS, hyperparameters)

    print(f'Best Hyperparameters: {best_hyperparams}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Best Model Saved at: {best_model_path}')

'''
RNN OPTIMAL
Best Hyperparameters: (17, 32, 3, 1, 0.0001)
Best Validation Loss: 6.055213364417763
Best Model Saved at: best_model.pth
'''





# %% Visualize the perfomance of the model After all the records and hyperparameter
# Data for plotting
models = ["Linear Regression", "RNN", "RNN + Add features", "RNN Hyper Tuned",
          "LSTM + Add Features", "LSTM Hyper Tuned", "GRU + Add Features",
          "GRU Hyper Tuned", "LGBM (Highest Score)", "Transformer"]
mae_values = [6.3531, 6.2834, 6.0552, 5.8527, 5.6445, 5.4554, 5.8944, 5.6753, 5.3367, 5.5683]

# Finding indexes of the lowest and second lowest MAE
lowest_mae_idx = mae_values.index(min(mae_values))
second_lowest_mae = sorted(mae_values)[1]
second_lowest_mae_idx = mae_values.index(second_lowest_mae)

# Colors for the bars
colors = ['skyblue' for _ in models]
colors[lowest_mae_idx] = 'green'  # Color for the lowest MAE
colors[second_lowest_mae_idx] = 'orange'  # Color for the second lowest MAE

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.barh(models, mae_values, color=colors)
plt.xlabel('MAE (nn.L1Loss)')
plt.title('Model Performance Comparison')
plt.gca().invert_yaxis() 

# Adding the text on the bars
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.4f}',
             va='center', ha='left', fontsize=8)

plt.show()



# %% SUBMISSION

###Uncomment the below code and change the variable kaggle_submission to True if you want to
###submit the result
kaggle_submission = False

# if kaggle_submission:
    # import optiver 2023;
    # env = optiver2023.make_env()
    # iter_test = env.iter_test()

    # counter = 0
    # for (test, revealed_targets, sample_prediction) in iter_test:
    #     if counter == 0:
    #         print(test.head(3))
    #         print(revealed_targets.head(3))
    #         print(sample_prediction.head(3))
    #     test = fill_missing_with_stock_mean(test)
    #     test = fill_columns_with_one(test, ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"])
    #     test = fill_missing_with_stock_median(test, ['matched_size', 'imbalance_size'])
    #     test = add_info_columns(test)
    #     sample_prediction['target'] = model.predict(test)
    #     env.predict(sample_prediction)
    #     counter += 1;
