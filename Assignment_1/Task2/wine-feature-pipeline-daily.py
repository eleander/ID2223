
# from ydata_synthetic.synthesizers.regular import RegularSynthesizer
# from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
# import pandas as pd

# # Load data
# df = pd.read_csv("./dataset/wine.csv")
# # drop rows that contain missing values
# df.dropna(inplace=True)
# print(df.isnull().sum())
# print(df.dtypes)
# # transform type column to unique numbers
# df["type"] = df["type"].astype('category')
# print(df.dtypes)

# num_cols = ["quality","fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"]
# cat_cols = ['type']

# # Define model and training parameters
# ctgan_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
# train_args = TrainParameters(epochs=3)

# # Train the generator model
# synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
# synth.fit(data=df, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# # Generate 1000 new synthetic samples
# synth_data = synth.sample(1) 
# print(synth_data)

import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "ydata-synthetic==1.1.0"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_wine(wine_df):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine flower
    """
    import pandas as pd
    import random

    virginica_df = generate_wine("Virginica", 8, 5.5, 3.8, 2.2, 7, 4.5, 2.5, 1.4)
    versicolor_df = generate_wine("Versicolor", 7.5, 4.5, 3.5, 2.1, 3.1, 5.5, 1.8, 1.0)
    setosa_df =  generate_wine("Setosa", 6, 4.5, 4.5, 2.3, 1.2, 2, 0.7, 0.3)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.uniform(0,3)
    if pick_random >= 2:
        wine_df = virginica_df
        print("Virginica added")
    elif pick_random >= 1:
        wine_df = versicolor_df
        print("Versicolor added")
    else:
        wine_df = setosa_df
        print("Setosa added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
