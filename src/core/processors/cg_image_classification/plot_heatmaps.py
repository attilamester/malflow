from core.processors.cg_image_classification import paths
from util import config

config.load_env(paths.get_cg_image_classification_env())

from collections import OrderedDict

import torch
import torch.utils.data
import matplotlib.pyplot as plt

from core.processors.cg_image_classification.dataset import ImgDataset
from core.processors.cg_image_classification.nn_model.bagnet_heatmaps import generate_heatmap_pytorch, plot_heatmap
from core.processors.cg_image_classification.train_definitions import get_model, get_dataset
from core.processors.cg_image_classification.dataset.dataloader import create_torch_bodmas_dataset_loader, BodmasDataset


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


def plot_heatmap_on_model(model: torch.nn.Module, checkpoint_path: str, dataset: ImgDataset,
                          dataloader: torch.utils.data.DataLoader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(get_state_dict(checkpoint["state_dict"]))
    model.to(device)
    model.eval()

    def show_heatmap(model, dataset, images: torch.tensor, details: BodmasDataset.ItemDetails, image_title: str,
                     subplot_index: int):
        image = images.numpy()
        image_to_plot = image[0].transpose([1, 2, 0])
        output = model(images.to(device, non_blocking=True))
        pred = output.detach().cpu().numpy()[0].argmax()
        pred = int(pred)
        target_num = target.detach().cpu().numpy()[0].item()

        heatmap_pred = generate_heatmap_pytorch(model, dataset, image, pred)
        ax = plt.subplot(231 + subplot_index)
        ax.set_title(image_title)
        plt.imshow(image_to_plot)

        ax = plt.subplot(232 + subplot_index)
        ax.set_title(
            f"heatmap: according to Predicted (correct: {target_num == pred}) {dataset._data_index2class[pred]}")
        plot_heatmap(heatmap_pred, image_to_plot, ax)
        if target_num != pred:
            heatmap_gt = generate_heatmap_pytorch(model, dataset, image, target_num)
            ax = plt.subplot(233 + subplot_index)
            ax.set_title(f"heatmap: according to GT {dataset._data_index2class[target_num]}")
            plot_heatmap(heatmap_gt, image_to_plot, ax)

    details: BodmasDataset.ItemDetails
    for i, (images, target, details) in enumerate(dataloader):
        # To exam only the packed samples
        if not details.packed.packed:
            continue

        fig = plt.figure(figsize=(10, 10))
        show_heatmap(model, dataset, images, details, f"Sample\n{details.md5}", 0)

        if details.packed.packed:
            show_heatmap(model, dataset, details.packed.orig_image, details,
                         f"Packed original\n{details.packed.orig_md5}", 3)

        plt.show()


if __name__ == "__main__":
    chkpt = "./nn_model/0309-2032_Bagnet-9_False_Bodmas-30x30x3_32_100-train-38980-val-12994_model_best.pth.tar"
    MODEL = get_model()
    bodmas = get_dataset()
    ds, dl = create_torch_bodmas_dataset_loader(bodmas, bodmas._data_df_gt_filtered, 1)
    ds.set_iter_details(True)

    plot_heatmap_on_model(MODEL, chkpt, bodmas, dl)
