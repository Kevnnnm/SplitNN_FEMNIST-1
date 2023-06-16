import numpy as np
import pandas as pd
from time import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import seaborn as sns
from skimage.metrics import mean_squared_error as mse

import datas
import classes

transform = transforms.Normalize((-0.5), (0.5))
device = 'mps'


np.random.seed(42)

client_train_ds = datas.FEMNIST(train=True, transform = transform, client_num = 1)
client_train_nums = datas.only_numbers(client_train_ds)
client_loader = DataLoader(client_train_nums, batch_size=1, shuffle = True)
client_model = classes.SplitNN().to(device)
client_model.load_state_dict(torch.load("Trained_models/FEMNIST_SplitNN_client_nums.pth"))

shadow_model = classes.ShadowNN().to(device)
shadow_model.load_state_dict(torch.load("Trained_models/FEMNIST_SplitNN_shadow_nums.pth"))

def main(client_model, client_loader, model_type):
    N = 10000
    size = 500
    smash_data = [None] * size
    smash_label = [None] * size
    row_nums = [None] * size

    for i, (images, labels) in enumerate(client_loader):
        if i == size: break
        images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
        smash_data[i] = client_model.first_part(images).detach().cpu().numpy()[0]
        smash_label[i] = labels.item()
        row_nums[i] = i

    # print(smash_data[0]) #array of dimension 500 data points, require 0th index of each one
    # print(smash_label[0]) #label for given data point

    df = pd.DataFrame(smash_data, row_nums)
    df['labels'] = smash_label
    #df.columns = pd.RangeIndex(1, len(df.columns)+1) 

    rndperm = np.random.permutation(df.shape[0])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[row_nums].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    print('PCA: Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        xs=df.loc[rndperm,:]["pca-one"], 
        ys=df.loc[rndperm,:]["pca-two"], 
        zs=df.loc[rndperm,:]["pca-three"], 
        c=df.loc[rndperm,:]["labels"], 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.title('PCA-3D')
    plt.show()

    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[row_nums].values

    time_start = time()
    tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    #df_subset['tsne-3d-three'] = tsne_results[:,2]

    plt.figure(figsize=(16,7))

    ax1 = plt.subplot(1, 2, 1)
    if model_type == 0: plt.title("Client Model 2D PCA")
    if model_type == 1: plt.title("Shadow Model 2D PCA")
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax1
    )


    ax2 = plt.subplot(1, 2, 2)
    if model_type == 0: plt.title("Client Model 2D t-SNE || 2 Components")
    if model_type == 1: plt.title("Shadow Model 2D t-SNE || 2 Components")
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="labels",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax2
    )


    if model_type == 0: plt.savefig("Data_Visualizations/Client_Model_2-Components.png")
    if model_type == 1: plt.savefig("Data_Visualizations/Shadow_Model_2-Components.png")
    plt.show()
    

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    return smash_data

# data_cl = main(client_model, client_loader, 0)
# data_sh = main(shadow_model, client_loader, 1)

# avg_mse = 0
# for i in range(len(data_cl)):
#     avg_mse += mse(data_cl[i], data_sh[i])
# avg_mse /= len(data_cl)
# print(f"Average MSE: {avg_mse}")







N = 10000
size = 500
cl_smash_data = [None] * size
cl_smash_label = [None] * size
cl_row_nums = [None] * size

sh_smash_data = [None] * size
sh_smash_label = [None] * size
sh_row_nums = [None] * size

for i, (images, labels) in enumerate(client_loader):
    if i == size: break
    images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    cl_smash_data[i] = client_model.first_part(images).detach().cpu().numpy()[0]
    cl_smash_label[i] = labels.item()
    cl_row_nums[i] = i

    sh_smash_data[i] = shadow_model.first_part(images).detach().cpu().numpy()[0]
    sh_smash_label[i] = labels.item()
    sh_row_nums[i] = i

# print(smash_data[0]) #array of dimension 500 data points, require 0th index of each one
# print(smash_label[0]) #label for given data point

cl_df = pd.DataFrame(cl_smash_data, cl_row_nums)
cl_df['labels'] = cl_smash_label
sh_df = pd.DataFrame(sh_smash_data, sh_row_nums)
sh_df['labels'] = sh_smash_label
#df.columns = pd.RangeIndex(1, len(df.columns)+1) 

cl_rndperm = np.random.permutation(sh_df.shape[0])
cl_pca = PCA(n_components=3)
cl_pca_result = cl_pca.fit_transform(sh_df[sh_row_nums].values)

sh_rndperm = np.random.permutation(cl_df.shape[0])
sh_pca = PCA(n_components=3)
sh_pca_result = sh_pca.fit_transform(cl_df[cl_row_nums].values)

cl_df['pca-one'] = cl_pca_result[:,0]
cl_df['pca-two'] = cl_pca_result[:,1] 
cl_df['pca-three'] = cl_pca_result[:,2]

sh_df['pca-one'] = sh_pca_result[:,0]
sh_df['pca-two'] = sh_pca_result[:,1] 
sh_df['pca-three'] = sh_pca_result[:,2]

print('PCA: Explained variation per principal component: {}'.format(cl_pca.explained_variance_ratio_))
print('PCA: Explained variation per principal component: {}'.format(sh_pca.explained_variance_ratio_))


fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    xs=cl_df.loc[cl_rndperm,:]["pca-one"], 
    ys=cl_df.loc[cl_rndperm,:]["pca-two"], 
    zs=cl_df.loc[cl_rndperm,:]["pca-three"], 
    #c=cl_df.loc[cl_rndperm,:]["labels"], 
    c = 'blue',
    cmap='tab10'
)

# Plot shadow data
ax.scatter(
    xs=sh_df.loc[sh_rndperm, "pca-one"],
    ys=sh_df.loc[sh_rndperm, "pca-two"],
    zs=sh_df.loc[sh_rndperm, "pca-three"],
    #c=sh_df.loc[sh_rndperm, "labels"],
    c = 'red',
    cmap='tab10',
    alpha=0.5,  # Adjust transparency for shadow data
    label='Shadow Data'
)

ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.title('PCA-3D')
plt.show()

cl_df_subset = cl_df.loc[cl_rndperm[:N],:].copy()
cl_data_subset = cl_df_subset[cl_row_nums].values

sh_df_subset = cl_df.loc[sh_rndperm[:N],:].copy()
sh_data_subset = cl_df_subset[sh_row_nums].values

time_start = time()
cl_tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=300)
cl_tsne_results = cl_tsne.fit_transform(cl_data_subset)

sh_tsne = TSNE(n_components=2, verbose=2, perplexity=40, n_iter=300)
sh_tsne_results = sh_tsne.fit_transform(sh_data_subset)

print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))

cl_df_subset['cl_tsne-2d-one'] = cl_tsne_results[:, 0]
cl_df_subset['cl_tsne-2d-two'] = cl_tsne_results[:, 1]
sh_df_subset['sh_tsne-2d-one'] = sh_tsne_results[:, 0]
sh_df_subset['sh_tsne-2d-two'] = sh_tsne_results[:, 1]
#df_subset['tsne-3d-three'] = tsne_results[:,2]

plt.figure(figsize=(16,7))

ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="labels",
    #palette=sns.color_palette("hls", 10),
    palette = sns.color_palette(["blue", "red"]),
    data=cl_df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="labels",
    #palette=sns.color_palette("hls", 10),
    palette = sns.color_palette(["blue", "red"]),
    data=sh_df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax1.legend(['Client Data', 'Shadow Data'])

ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="cl_tsne-2d-one", y="cl_tsne-2d-two",
    hue="labels",
    #palette=sns.color_palette("hls", 10),
    palette = sns.color_palette(["blue", "red"]),
    data=cl_df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
sns.scatterplot(
    x="sh_tsne-2d-one", y="sh_tsne-2d-two",
    hue="labels",
    #palette=sns.color_palette("hls", 10),
    palette = sns.color_palette(["blue", "red"]),
    data=sh_df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax2.legend(['Client Data', 'Shadow Data'])


plt.savefig("Data_Visualizations/Comparison.png")
plt.show()


cl_pca_50 = PCA(n_components=50)
cl_pca_result_50 = cl_pca_50.fit_transform(cl_data_subset)

sh_pca_50 = PCA(n_components=50)
sh_pca_result_50 = sh_pca_50.fit_transform(sh_data_subset)

print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(cl_pca_50.explained_variance_ratio_)))
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(sh_pca_50.explained_variance_ratio_)))

avg_mse = 0
for i in range(len(cl_smash_data)):
    avg_mse += mse(cl_smash_data[i], sh_smash_data[i])
avg_mse /= len(cl_smash_data)
print(f"Average MSE: {avg_mse}")