from dataloaders.kospeech import KoSpeechDataModule
from models.basic import CNNLayer


model = CNNLayer(1)

dm = KoSpeechDataModule(batch_size=4)
dm.setup()

train = dm.train_dataloader()

for a, b, c in train:
    print(type(b))

    # outputs = model(b[0])

    break
