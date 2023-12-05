import gradio as gr
from PIL import Image, ImageDraw
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(alcohol, density, volatile_acidity, chlorides):
    print("Calling function")
    df = pd.DataFrame([[alcohol, density, volatile_acidity, chlorides]], columns=['alcohol', 'density', 'volatile_acidity', 'chlorides'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    img = Image.new('RGB', (100, 30), color = (73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "Quality: " + str(res[0]), fill=(255,255,0))
    return img
        
demo = gr.Interface(
    fn=wine,
    title="Wine Predictive Analytics",
    description="Experiment with different wine configurations.",
    allow_flagging="never",
    # default values are the mean values of the training dataset
    inputs=[
        gr.inputs.Number(default=10, label="alcohol"),
        gr.inputs.Number(default=0.99, label="density"),
        gr.inputs.Number(default=0.33,label="volatile_acidity"),
        gr.inputs.Number(default=0.056, label="chlorides"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

