import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
def standardize_tensor(tensor, T_min=22., T_max=100.):
  """Standardizes a tensor to [0, 1] using min-max scaling."""
  return (tensor - T_min) / (T_max - T_min)

def destandardize_tensor(tensor, T_min = 22., T_max = 100.):
  """Destandardizes a tensor from [0, 1] to its original range."""
  return tensor * (T_max - T_min) + T_min

def load_dataset(data_config):
    """
    Input: data_config - edit in configs/config.yaml
    Return: Train loader & Val loader 70:30
    """
    #Load configs
    x_path = data_config["x_path"]
    y_path = data_config["y_path"]
    BATCH_SIZE = data_config["batch_size"]
    L = data_config["L"]
    dt = data_config["dt"]
    T = data_config["T"]
    nt = T / dt
    T_MIN = data_config["T_MIN"]
    T_MAX = data_config["T_MAX"]
    CP_MIN = data_config["CP_MIN"]
    CP_MAX = data_config["CP_MAX"]
    PS_MIN = data_config["PS_MIN"]
    PS_MAX = data_config["PS_MAX"]
    # Load tensor file
    X_data= torch.load(x_path,weights_only=False)
    y_data= torch.load(y_path,weights_only=False)
    print(f'X_data shape: {X_data.shape}')
    print(f'y_data shape: {y_data.shape}')
    # split data to time_step
    y_subset = y_data[:, :, ::dt]
    y_subset = y_subset[:,:,:T]
    y_subset = y_subset.clone().detach()
    print(f'y_data shape after slice: {y_subset.shape}')
    #Standerdize data and reshape data
    X = X_data.permute(0, 2, 1)
    Y = y_subset.permute(0, 2, 1)
    X[:,0:2,:] = torch.abs(X[:,0:2,:]/1.)
    X[:,2:3,:] = standardize_tensor(X[:,2:3,:], 0.8, 2.5)
    X[:,3:4,:] = standardize_tensor(X[:,3:4,:], 1.1, 2.5)
    X[:,4:5,:] = standardize_tensor(X[:,4:5,:], PS_MIN, PS_MAX)
    X[:,5:6,:] = standardize_tensor(X[:,5:6,:], CP_MIN, CP_MAX)
    Y = standardize_tensor(Y, T_MIN, T_MAX)
    print("*"*50)
    print(f'X train shape: {X.shape}')
    print(f'Y train shape: {Y.shape}')
    print("*"*50)
    num_channels = X.shape[1]
    for i in range(num_channels):
      channel_min = X[:, i, :].min()
      channel_max = X[:, i, :].max()
      print(f'Channel {i}: Min = {channel_min.item()}, Max = {channel_max.item()}')

    print("*"*50)
    X = X.float()
    Y = Y.float()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=BATCH_SIZE)

    print(f'X_train shape: {X_train.shape}')
    print(f'Y_train shape: {Y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'Y_test shape: {Y_test.shape}')
    return train_loader, val_loader