
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import pandas as pd

# Load data
df = pd.read_csv("./dataset/wine.csv")
# drop rows that contain missing values
df.dropna(inplace=True)
print(df.isnull().sum())
print(df.dtypes)
# transform type column to unique numbers
df["type"] = df["type"].astype('category')
print(df.dtypes)

num_cols = ["quality","fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides"]
cat_cols = ['type']

# Define model and training parameters
ctgan_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
train_args = TrainParameters(epochs=3)

# Train the generator model
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(data=df, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# Generate 1000 new synthetic samples
synth_data = synth.sample(1) 
print(synth_data)
