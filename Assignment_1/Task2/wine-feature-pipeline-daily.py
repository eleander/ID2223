import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks", "ydata-synthetic==1.1.0"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_random_wine(project):
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    mr = project.get_model_registry()
    model = mr.get_model("wine_generator", version=1)
    model_dir = model.download()
    model = RegularSynthesizer.load(model_dir + '/wine_generator.pkl')
    sample = model.sample(1)
    return sample

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_sample = generate_random_wine(project)

    wine_fg = fs.get_feature_group(name="wine",version=1)
    wine_fg.insert(wine_sample)

if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
