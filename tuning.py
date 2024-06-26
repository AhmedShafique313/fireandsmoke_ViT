from transformers import ViTImageProcessor, ViTForImageClassification

def fine_tune_model(model, train_dataset, val_dataset, output_dir='./results', num_train_epochs=3):
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()