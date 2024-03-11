from core.processors.cg_image_classification import paths
from util import config

config.load_env(paths.get_cg_image_classification_env())

from collections import OrderedDict

import torch
import torch.utils.data
import matplotlib.pyplot as plt

from core.processors.cg_image_classification.dataset import Datasets, ImgDataset
from core.processors.cg_image_classification.nn_model.bagnet_heatmaps import generate_heatmap_pytorch, plot_heatmap
from core.processors.cg_image_classification.train_definitions import get_model
from core.processors.cg_image_classification.dataset.dataloader import create_torch_bodmas_dataset_loader


def get_state_dict(state_dict) -> OrderedDict:
    """
    Models trained with DataParallel have a "module." prefix before each layer name. This function removes it.
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def plot_heatmap_on_model(model: torch, checkpoint_path: str, dataset: ImgDataset,
                          dataloader: torch.utils.data.DataLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(get_state_dict(checkpoint["state_dict"]))
    model.to(device)
    model.eval()

    for i, (images, target, details) in enumerate(dataloader):
        image = images.numpy()
        target_num = target.detach().cpu().numpy()[0].item()
        images = images.to(device, non_blocking=True)
        output = model(images)

        pred = output.detach().cpu().numpy()[0].argmax()
        pred = int(pred)

        original_image = image[0].transpose([1, 2, 0])

        heatmap_pred = generate_heatmap_pytorch(model, dataset, image, pred)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(131)
        ax.set_title(f"original\n{details['md5']}")
        plt.imshow(original_image)

        if target_num == pred:
            ax = plt.subplot(132)
            ax.set_title(f"heatmap: according to Predicted=GT {dataset.data_index2class[target_num]}")
            plot_heatmap(heatmap_pred, original_image, ax)
        else:
            heatmap_gt = generate_heatmap_pytorch(model, dataset, image, target_num)

            ax = plt.subplot(132)
            ax.set_title(f"heatmap: according to Predicted {dataset.data_index2class[pred]}")
            plot_heatmap(heatmap_pred, original_image, ax)
            ax = plt.subplot(133)
            ax.set_title(f"heatmap: according to GT {dataset.data_index2class[target_num]}")
            plot_heatmap(heatmap_gt, original_image, ax)

        plt.show()


if __name__ == "__main__":
    chkpt = "./nn_model/0309-2032_Bagnet-9_False_Bodmas-30x30x3_32_100-train-38980-val-12994_model_best.pth.tar"
    bodmas = Datasets.BODMAS.value
    bodmas.filter_ground_truth(100)
    ds, dl = create_torch_bodmas_dataset_loader(bodmas, bodmas.data_df_gt_filtered, 1)
    ds.set_iter_details(True)
    model = get_model()
    plot_heatmap_on_model(model, chkpt, bodmas, dl)
