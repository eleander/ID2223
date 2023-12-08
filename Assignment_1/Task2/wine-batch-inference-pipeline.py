import os
import modal
from PIL import Image, ImageDraw
import numpy as np

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_batch_inference")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image", "Pillow"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()
    batch_data = batch_data[['alcohol', 'density', 'volatile_acidity', 'chlorides']]

    y_pred = model.predict(batch_data)
    offset = 1
    wine = y_pred[y_pred.size-offset]
    print("Wine predicted: " + str(wine))
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Quality: " + str(wine), fill=(255,255,0))
    img.save("./latest_wine.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)

    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read() 
    # print(df)

    label = df.iloc[-offset]["quality"]
    print("Wine actual: " + str(label))
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Quality: " + str(label), fill=(255,255,0))
    img.save("./actual_wine.png")
    dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine],
        'label': [label],
        'datetime': [now]
        }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])


    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_wine.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of all wines
    columns = sorted(list(set(np.unique(labels)) | set(np.unique(predictions))))
    wine_count = len(columns)
    print("Number of different wine quality predictions or truth up to date: " + str(wine_count))
    # We modified the code so that the confusion matrix is generated dynamically depending on the selected labels and predictions
    if wine_count < 2:
        # Create an empty image to avoid deployment errors in the monitor app
        empty_image = Image.new('RGB', (100, 100), color = (73, 109, 137))
        empty_image.save("./confusion_matrix_wine.png")
    else:
        results = confusion_matrix(labels, predictions)
        true_cols = [f'True {col}' for col in columns]
        pred_cols = [f'Pred {col}' for col in columns]

        df_cm = pd.DataFrame(results, true_cols, pred_cols)

        cm = sns.heatmap(df_cm, annot=True)

        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_wine.png")

    dataset_api.upload("./confusion_matrix_wine.png", "Resources/images", overwrite=True)



if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

