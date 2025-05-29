# Helper function, inspired from lecture code: https://colab.research.google.com/drive/1yWMwSUNhuoy-85KzR1th5e4aN8lR-UaY?usp=sharing

import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def train_model(use_release=True, use_dance=True, epochs=5, description = "train", model_setup_encoder=None):
    """
    Train a model using the provided setup function and metadata configuration.

    Args:
        use_release (bool): Whether to use release year as metadata.
        use_dance (bool): Whether to use danceability as metadata.
        epochs (int): Number of training epochs.
        description (str): Identifier for saved model and metadata files.
        model_setup_encoder (callable): Function that returns model, optimizer, loss_fn, train/val loaders.

    Returns:
        model: The trained model with best weights loaded.
    """
    
    print("\n------ Metadata Usage Configuration ------")
    print(f"Using release_year: {use_release}")
    print(f"Using danceability: {use_dance}")

    model, optimizer, loss_fn, train_loader, val_loader = model_setup_encoder(use_release, use_dance)
    best_val_loss = float('inf')
    best_epoch = None
    best_save_path = "run_results/models/"+description+".pth"


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            metadata = batch.get('metadata', torch.zeros(input_ids.size(0), 1)).to(model.device)
            labels = batch['labels'].to(model.device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, metadata)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

        train_accuracy = correct / total
        print(f"\nEpoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_accuracy, val_loss = evaluate_model(model, val_loader, loss_fn=loss_fn, label="validation")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_save_path)
            with open(best_save_path.replace(".pth", "_meta.json"), "w") as f:
                json.dump({"best_epoch": best_epoch, "best_val_loss": best_val_loss}, f)
            print(f"[Best Model Saved] Epoch {best_epoch} with val loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load(best_save_path))
    model.best_epoch = best_epoch
    model.best_score = 1 - best_val_loss
    print(f"\n[Model Restored] Loaded best model from epoch {best_epoch} with val loss: {best_val_loss:.4f}")
    return model




def evaluate_model(model, dataloader, loss_fn = None, label="Test", folder = "", description = "", file_name = "", label_encoder=None):
    """
    Evaluate a trained model on a dataset and optionally save performance reports.

    Args:
        model: Trained model to evaluate.
        DataLoader: DataLoader for evaluation set.
        loss_fn: Loss function for computing validation loss.
        label: Label to distinguish between "Test", "Validation", etc.
        description: Identifier to locate metadata for restored model.
        folder: Subfolder under results directory for saving outputs. - bert or distilbert
        file_name: Name used to create result output paths.
        LabelEncoder: Fitted encoder for converting label indices to strings.

    Returns:
        float: Validation accuracy and loss (only if label is 'validation').
    """
    show_details = label.lower() != "validation"
    if description:
        meta_path = f"run_results/models/{folder}{description}_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                print(f"[Evaluation] Model loaded from file '{description}' (epoch {meta['best_epoch']}, val_loss: {meta['best_val_loss']:.4f})")
        else:
            print(f"[Evaluation] Warning: No metadata file found for '{description}'")

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            metadata = batch.get('metadata', torch.zeros(input_ids.size(0), 1)).to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids, attention_mask, metadata)
            if loss_fn is not None:
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
  
    if show_details:
        accuracy = correct / total
        print(f"{label} Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, digits=2)
        print(report)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"{label} Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()
        
        #save the results
        save_test_results(
            folder = folder,
            file_name=file_name,
            accuracy=accuracy,
            classification_rep=report,
            confusion_matrix_data=cm,
            label_encoder = label_encoder
        )
    else:
        accuracy = correct / total
        val_loss = val_loss / len(dataloader)
        print(f"\n{label} score: Val Loss: {val_loss:.2f}, Val Accuracy: {accuracy:.2f}")

        return accuracy, val_loss




def save_test_results(folder, file_name, accuracy, classification_rep, confusion_matrix_data, label_encoder):
    """
    Save test results including accuracy, classification report, and confusion matrix plot.

    Args:
        folder: Subdirectory path within results/distilbert to save files.
        file_name: Name of the specific test run.
        accuracy: Accuracy score from evaluation.
        classification_rep: Formatted classification report.
        confusion_matrix_data: Confusion matrix array.
        LabelEncoder: Encoder to decode class labels.
    """
    
    save_dir = f"run_results/results/{folder}{file_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Save accuracy and report
    results = {
        "test_accuracy": accuracy,
        "classification_report": classification_rep
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    if folder != "llama/":
        label_names = label_encoder.classes_
    else:
        label_names = label_encoder
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    

    print(f"[Saved] Test results saved to: {save_dir}")