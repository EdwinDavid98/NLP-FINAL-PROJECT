import json
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Definir el path donde están los archivos guardados
path = '/home/emontenegrob/Labs_NLP/prueba2_chatbot/'

# PyTorch Lightning Module
class RobertaClassifier(pl.LightningModule):
    def __init__(self, num_labels, learning_rate=5e-5, weight_decay=0.01):
        super(RobertaClassifier, self).__init__()
        self.save_hyperparameters()  # Guardar los hiperparámetros para reproducir el modelo
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels, hidden_dropout_prob=0.4)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        val_loss = outputs.loss
        preds = torch.argmax(outputs.logits, dim=-1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Ajusta según las épocas
        return [optimizer], [scheduler]

# Cargar los datos desde el archivo JSON
with open(f'{path}intents_universidad2_augmented_clean.json', 'r') as file:
    data = json.load(file)

# Extraer patrones y etiquetas
patterns = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])

# Convertir las etiquetas en números
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Usar el tokenizer de RoBERTa para convertir texto en tokens
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenizar y codificar las secuencias de entrada
inputs = tokenizer(patterns, padding=True, truncation=True, return_tensors="pt", max_length=64)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(inputs['input_ids'], y, test_size=0.2, random_state=42)
attention_masks_train, attention_masks_test = train_test_split(inputs['attention_mask'], test_size=0.2, random_state=42)

# Convertir a tensores
train_data = TensorDataset(X_train, torch.tensor(attention_masks_train).clone().detach(), torch.tensor(y_train))
test_data = TensorDataset(X_test, torch.tensor(attention_masks_test).clone().detach(), torch.tensor(y_test))

# Cargar los datos en DataLoader para ser usados por el modelo
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=32, num_workers=4)
val_dataloader = DataLoader(test_data, batch_size=32, num_workers=4)

# Callback de EarlyStopping para detener el entrenamiento si la pérdida de validación deja de mejorar
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitorea la pérdida de validación
    patience=3,  # Si no mejora en 3 épocas consecutivas, detiene el entrenamiento
    verbose=True,
    mode='min'  # Queremos que la pérdida disminuya
)

# Callback de ModelCheckpoint para guardar el mejor modelo
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Monitorea la pérdida de validación
    dirpath=f'{path}checkpoints/',  # Carpeta para guardar el modelo
    filename='best-checkpoint',  # Nombre del archivo
    save_top_k=1,  # Guardar solo el mejor modelo
    mode='min',  # Guardar el modelo con la pérdida más baja
    save_weights_only=True  # Guardar solo los pesos del modelo
)

# Instanciar el modelo
model = RobertaClassifier(num_labels=len(set(labels)))

# PyTorch Lightning Trainer con Early Stopping y Checkpoint
trainer = pl.Trainer(
    max_epochs=30,
    devices=1,  # Número de dispositivos (GPUs o CPUs)
    accelerator="gpu",  # Acelerador: GPU o CPU
    precision=16,  # Mixed precision
    callbacks=[early_stopping, checkpoint_callback],  # Añadir early stopping y checkpoints
)

# Entrenamiento del modelo
trainer.fit(model, train_dataloader, val_dataloader)

# Guardar el modelo completo (pesos + configuración)
model.model.save_pretrained(f'{path}roberta_complete_model')
tokenizer.save_pretrained(f'{path}roberta_complete_model')

# Evaluar el modelo
trainer.validate(model, dataloaders=val_dataloader)
