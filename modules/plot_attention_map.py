import matplotlib.pyplot as plt
from .predict import predict

def plot_attention_map(original_img, attention_map, classnames, model):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.set_tight_layout(True)
    model_predict = predict(model, original_img, classnames)
    original_img = original_img.resize((224, 224))
    fig.suptitle(f"Predicted: {model_predict[0]} - {model_predict[1]}", fontsize=16)
    ax[0].imshow(original_img)
    ax[1].imshow(attention_map)
    plt.show()