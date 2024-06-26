def fine_tune_model(model, train_dataset, val_dataset, output_dir='./results', num_train_epochs=3):
    from transformers import Trainer, TrainingArguments
    # so we have some training aruguments 
    training_args = TrainingArguments(
        output_dir= output_dir,
        num_train_epochs= num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy='epoch',
        save_strategy="epoch",
        logging_dir='./logs'
    )