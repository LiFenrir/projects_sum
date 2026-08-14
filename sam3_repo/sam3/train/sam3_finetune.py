import torch

from torch.utils.data import DataLoader

from transformers import Sam3Model, Sam3Processor

from data.sam3_text_dataset import SAM3TextDataset

from loss.finetune_loss import segmentation_loss



DEVICE="cuda"



# ======================
# load SAM3
# ======================


model = Sam3Model.from_pretrained(
    "facebook/sam3"
)


model.to(
    DEVICE
)



# ======================
# freeze image encoder
# ======================


for name, param in model.named_parameters():

    if "vision_encoder" in name:

        param.requires_grad=False



# ======================
# dataset
# ======================


dataset = SAM3TextDataset(

    root="./sam3_dataset",

    annotation_file=
    "./sam3_dataset/annotations.json"

)


loader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=True,

    num_workers=8,

    pin_memory=True
)



# ======================
# optimizer
# ======================


train_params = [

    p for p in model.parameters()

    if p.requires_grad

]


optimizer=torch.optim.AdamW(

    train_params,

    lr=1e-5,

    weight_decay=0.05

)



# ======================
# training
# ======================


epochs=100



model.train()



for epoch in range(epochs):


    total_loss=0


    for batch in loader:


        images=batch["image"].to(
            DEVICE
        )


        texts=batch["text"]


        masks=batch["mask"].to(
            DEVICE
        )


        outputs=model(

            pixel_values=images,

            input_text=texts

        )


        pred_masks = (
            outputs.pred_masks
        )


        loss=segmentation_loss(

            pred_masks,

            masks

        )


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    print(
        f"epoch {epoch}:",
        total_loss/len(loader)
    )


    torch.save(

        model.state_dict(),

        f"sam3_ft_{epoch}.pth"

    )