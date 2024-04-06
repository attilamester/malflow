import os
from collections import OrderedDict
from typing import Type, List

import cv2
import numpy as np
import plotly
import plotly.graph_objects as go
import torch
import torch.utils.data
from plotly.subplots import make_subplots

from core.data import DatasetProvider
from core.processors.cg_image_classification import paths
from util import config

config.load_env()
config.load_env(paths.get_cg_image_classification_env())

from core.model.call_graph_image import CallGraphImage
from core.processors.cg_image_classification.dataset import ImgDataset
from core.processors.cg_image_classification.nn_model.bagnet_heatmaps import generate_heatmap_pytorch
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


def evaluate_model(model: torch.nn.Module, checkpoint_path: str, dataloader: torch.utils.data.DataLoader, pred_topk=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(get_state_dict(checkpoint["state_dict"]))
    model.to(device)
    model.eval()

    details: BodmasDataset.ItemDetails
    for i, (images, target, details) in enumerate(dataloader):
        output = model(images.to(device, non_blocking=True))
        np_target = target.detach().cpu().numpy()
        target = np_target.tolist()

        # Keep the top 5 predicted classes in DESC orted
        prediction = torch.topk(output, k=pred_topk, dim=1, largest=True, sorted=True)[1].numpy().tolist()
        # ########
        # Or: np_output = output.detach().cpu().numpy(); prediction = [int(np_output[k].argmax()) for k in range(len(output))]
        # Or: prediction = [int(v) for v in torch.max(output.data, 1)[1]]
        # ########

        prediction_on_packed_images_ = [None] * len(images)

        if any(details.packed):
            original_packed_images = [(k, img) for k, img in enumerate(details.image_packed_tf) if
                                      torch.is_tensor(img)]
            torch_images = torch.stack([img for _, img in original_packed_images])
            output = model(torch_images.to(device, non_blocking=True))
            prediction_on_packed_images = torch.topk(output, k=pred_topk, dim=1, largest=True, sorted=True)[
                1].numpy().tolist()

            j = 0
            for k, img in original_packed_images:
                prediction_on_packed_images_[k] = prediction_on_packed_images[j]
                j += 1

        yield images, target, details, prediction, prediction_on_packed_images_


def plotly_heatmap_on_model(model: torch.nn.Module, checkpoint_path: str, dataset: ImgDataset,
                            dataloader: torch.utils.data.DataLoader):
    details: BodmasDataset.ItemDetails
    for (images, target, details,
         prediction,
         prediction_on_packed_images) in evaluate_model(model, checkpoint_path, dataloader, pred_topk=5):

        batch_size = len(images)
        for i in range(batch_size):
            plotly_html_of_dataloader_item(model, dataset, images[i], details.image_disk[i],
                                           target[i], prediction[i], details.index[i].item(), details.packed[i].item(),
                                           details.image_packed_tf[i], details.image_packed_disk[i])


def plotly_html_of_dataloader_item(model: torch.nn.Module, dataset: ImgDataset, image: torch.tensor,
                                   image_disk: np.ndarray,
                                   target: int, prediction: List[int], index: int, packed: bool,
                                   image_packed: torch.tensor, image_packed_disk: np.ndarray):
    def add_trace_image_heatmap(fig, img, img_disk, row):
        heatmaps_classes = generate_heatmap_pytorch(model, np.stack([img.numpy()]), classes)
        fig.add_trace(plotly_create_image_of_cg(img_disk), row=row, col=1)
        fig.add_trace(plotly_create_image_of_cg(img_disk), row=row, col=2)
        for i, heatmap_class in enumerate(heatmaps_classes):
            trace_name = f"Class {dataset._data_index2class[classes[i]]}"
            if classes[i] == target:
                trace_name += " (== GT)"
            fig.add_trace(
                plotly_create_image_of_heatmap(heatmap_class, trace_name=trace_name, visible=classes[i] == target),
                row=row, col=2)

    classes = prediction[:]
    if target not in set(prediction):
        classes.append(target)

    md5 = dataset.get_row_id(dataset._data_df_gt_filtered.iloc[index])

    rows = 1 if not packed else 2
    subplot_titles = [f"Sample {md5}", "Heatmap of class activation"]
    if packed:
        packed_md5 = dataset._data_df_gt_filtered.iloc[index]["md5"]
        subplot_titles.extend([f"Orig. packed sample {packed_md5}", "Heatmap of class activation"])
    fig = make_subplots(rows=rows, cols=2, subplot_titles=subplot_titles)

    # heatmaps_classes = generate_heatmap_pytorch(model, np.stack([image.numpy()]), classes)
    add_trace_image_heatmap(fig, image, image_disk, 1)

    if packed:
        add_trace_image_heatmap(fig, image_packed, image_packed_disk, 2)

    plotly.offline.plot(fig, filename=f"{md5}_plotly.html", auto_open=False)

    if packed:
        exit(0)


def read_image(filepath) -> np.ndarray:
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def plotly_create_image_of_cg(np_img: np.ndarray) -> go.Image:
    hover_text = [[str(CallGraphImage.decode_rgb(r=np_img[i, j, 0],
                                                 g=np_img[i, j, 1],
                                                 b=np_img[i, j, 2])) for j in
                   range(np_img.shape[1])] for i in range(np_img.shape[0])]
    return go.Image(z=np_img, hoverinfo="x,y,text", text=hover_text, name="Original image")


def plotly_create_image_of_heatmap(np_img: np.ndarray, trace_name: str, visible: bool = True) -> go.Heatmap:
    height, width = np_img.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    # return go.Scatter(
    #     x=x_coords.flatten(),
    #     y=y_coords.flatten(),
    #     mode="markers",
    #     marker=dict(size=10, color=np_img.flatten(), colorscale="Viridis", showscale=True))
    return go.Heatmap(
        x=x_coords.flatten(),
        y=y_coords.flatten(),
        z=np_img.flatten(),
        name=trace_name,
        showscale=False,
        showlegend=True,
        opacity=0.8,
        colorscale="oranges",
        visible=None if visible else "legendonly"
    )


def plotly_html_of_sample(dset: Type[DatasetProvider], md5: str):
    """
    # TODO: add test for this
    # for i_data in INSTRUCTIONS:
    #     i = Instruction(i_data.disasm, b"0", [])
    #     enc = CallGraphImage.encode_instruction_rgb(i)
    #     print(f"Instruction: {i_data.disasm} | RGB: {enc}\n"
    #           f"Decoded    : {CallGraphImage.decode_rgb(rgb=enc)}")
    # Example output:
    # Instruction: ljmp 4:0xc2811a31 | RGB: b'"8\t'
    # Decoded    : [jmp] ADDR_FAR
    # Instruction: notrack jmp 0xfb7508c5 | RGB: b'"<\x19'
    # Decoded    : [bnd] [notrack] [jmp] CONST
    """
    plotly_images = []
    for dim in [(30, 30), (100, 100), (224, 224)]:
        img_path = os.path.join(dset.get_dir_images(), f"images_{dim[0]}x{dim[1]}",
                                f"{md5}_{dim[0]}x{dim[1]}_True_True.png")
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            continue
        np_img = read_image(img_path)
        plotly_images.append((plotly_create_image_of_cg(np_img), f"{dim[0]}x{dim[1]}"))

    fig = make_subplots(rows=1, cols=len(plotly_images), subplot_titles=[i[1] for i in plotly_images])
    for i, (plotly_img, dim) in enumerate(plotly_images):
        fig.add_trace(plotly_img, row=1, col=i + 1)

    plotly.offline.plot(fig, filename=f"{md5}_plotly.html", auto_open=False)


if __name__ == "__main__":
    chkpt = "./nn_model/0309-2032_Bagnet-9_False_Bodmas-30x30x3_32_100-train-38980-val-12994_model_best.pth.tar"
    MODEL = get_model()
    bodmas = get_dataset()
    ds, dl = create_torch_bodmas_dataset_loader(bodmas, bodmas._data_df_gt_filtered, 5)
    ds.set_iter_details(True)

    plotly_heatmap_on_model(MODEL, chkpt, bodmas, dl)
