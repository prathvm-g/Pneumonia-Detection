
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.config import EPOCHS

def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2
    )
    return [early_stop, reduce_lr]


def train_model(model, train_gen, val_gen, epochs=EPOCHS, label='Model'):
    print(f'\nTraining: {label}')
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=get_callbacks(),
        verbose=1
    )
    return history
