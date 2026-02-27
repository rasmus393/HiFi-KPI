import torch
import torch.nn as nn

class MulticlassClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Initialize the multiclass classifier with distance-aware loss.
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of classes to predict
        """
        super(MulticlassClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.linear(x)
    
    def compute_loss(self, outputs, targets):
        return self.loss_fn(outputs, targets)
    
class ClassifierTrainer:
    def __init__(self, model, learning_rate=0.0001, batch_size=32):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.batch_size = batch_size

    def train_epoch(self, train_loader, device):
        """
        Train for one epoch.
        Args:
            train_loader (DataLoader): DataLoader for training data
        Returns:
            tuple: (average total loss, average classification loss, average l1 loss, average l2 loss)
        """
        self.model.train()
        for batch  in train_loader:
            batch_X, batch_y = batch  
            outputs = self.model(batch_X.to(device))
            total_loss = self.model.compute_loss(outputs, batch_y.to(device))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        return total_loss
    
    def evaluate(self, test_loader, device):
        """
        Evaluate the model on test/validation data.
        Args:
            test_loader (DataLoader): DataLoader for test/validation data
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total_samples = 0

        with torch.no_grad():  # No gradient tracking
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = self.model(batch_X.to(device))
                loss = self.model.compute_loss(outputs, batch_y.to(device))
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)

        # Compute average loss and accuracy
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        print(avg_loss, accuracy)
        return avg_loss, accuracy
    
def run_classifier(train_embeddings, train_labels_encoded, 
                   val_embeddings, val_labels_encoded, 
                   label_encoder, batch_size=384, distance_matrix=None, input_dim = 384,  device="cuda"):
    
    num_classes = len(label_encoder.classes_)

    device = torch.device(device)
    X_train = torch.as_tensor(train_embeddings)
    y_train = torch.as_tensor(train_labels_encoded).long().to(device)
    X_val = torch.as_tensor(val_embeddings).to(device)
    y_val = torch.as_tensor(val_labels_encoded).long().to(device)

    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_val, y_val)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MulticlassClassifier(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    trainer = ClassifierTrainer(model, learning_rate=0.0001, batch_size=batch_size)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_losses = trainer.train_epoch(train_loader, device)
        eval_metrics = trainer.evaluate(test_loader, device=device)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training losses - Total: {train_losses.item():.4f}")
        print(f"Evaluation - Total Loss: {eval_metrics[0]:.4f}, "
              f"Classification ACC: {eval_metrics[1]:.4f}")
        print("-" * 30)
    
    return model
